"""app.run end to end with a real Store on a tmp dir and the real brain.

Faked: the camera (the test publishes its facts and stills), the vision backend (a
script of observations), the notifier (records deliveries, can fail), the audio
classifier (the model file is not in CI). Skipped: recorder, ticker and HTTP server.
Time is driven by the Ticks the test publishes, so nothing sleeps for real.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from babymon import app, factlog
from babymon.app import Backends
from babymon.camera import CameraOutputs
from babymon.config import CameraConfig, PushoverConfig, Settings, VisionConfig
from babymon.domain import (
    Alert,
    AlertKind,
    AlertSent,
    AudioSample,
    CameraStatus,
    ControlsChanged,
    Instant,
    MotionSample,
    Observation,
    Priority,
    SetControls,
    Still,
    Tick,
    VisionObserved,
)
from babymon.notify import DeliveryError
from babymon.store import Store
from tests.conftest import observation

JPEG = b"\xff\xd8fake-frame\xff\xd9"
STOMACH = observation(body="stomach")
HOUR = timedelta(hours=1)


def utcnow() -> Instant:
    return datetime.now(UTC)


class FakeCamera:
    """Hands its outputs to the test, which then plays the camera."""

    def __init__(self) -> None:
        self.outputs: asyncio.Future[CameraOutputs] = asyncio.get_running_loop().create_future()

    async def run(self, out: CameraOutputs) -> None:
        self.outputs.set_result(out)
        await asyncio.Event().wait()


@dataclass
class ScriptedVision:
    script: list[Observation]
    frames: list[Still] = field(default_factory=list)

    async def observe(self, frame: Still, reference: Still | None) -> Observation:
        self.frames.append(frame)
        return self.script[len(self.frames) - 1]


@dataclass
class RecordingNotifier:
    fail_first: int = 0
    delivered: list[tuple[Alert, Still | None]] = field(default_factory=list)

    async def deliver(self, alert: Alert, snapshot: Still | None) -> None:
        if self.fail_first:
            self.fail_first -= 1
            raise DeliveryError("phone unreachable")
        self.delivered.append((alert, snapshot))


class StubClassifier:
    def cry_probability(self, samples: np.ndarray) -> float:
        return 0.0


async def idle(*_: object) -> None:
    await asyncio.Event().wait()


def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        camera=CameraConfig(user="u", password="p"),
        vision=VisionConfig(api_key="k"),
        pushover=PushoverConfig(app_token="t", user_keys=("k",)),
        data_dir=tmp_path / "data",
        yamnet_path=tmp_path / "absent.onnx",
    )


def backends(camera: FakeCamera, vision: ScriptedVision, notifier: RecordingNotifier) -> Backends:
    return Backends(
        camera=camera.run,
        recorder=None,
        classifier=StubClassifier(),
        vision=vision,
        notifier=notifier,
        serve_http=None,
        ticker=idle,
    )


@contextlib.asynccontextmanager
async def running(
    cfg: Settings,
    camera: FakeCamera,
    vision: ScriptedVision,
    notifier: RecordingNotifier,
    initial_controls: SetControls | None = None,
    record_facts: Path | None = None,
) -> AsyncIterator[CameraOutputs]:
    """app.run as a task; on exit cancel it and re-raise anything an actor raised."""
    task = asyncio.create_task(
        app.run(cfg, initial_controls, record_facts, backends(camera, vision, notifier))
    )
    try:
        yield await camera.outputs
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def until(check: Callable[[], bool], timeout_s: float = 5.0) -> None:
    async with asyncio.timeout(timeout_s):
        while not check():
            await asyncio.sleep(0.01)


def rows[E](store: Store, base: Instant, kind: type[E]) -> list[E]:
    found = [s.event for s in store.events(base - HOUR, base + HOUR)]
    return [e for e in reversed(found) if isinstance(e, kind)]


async def two_stomach_checks(out: CameraOutputs, store: Store, base: Instant, start_s: int) -> None:
    """Camera up, a periodic check, a confirm check 20 s later, then the Tick that gates.
    Each Tick waits for its vision row so the next one sees the updated streak."""
    seen = len(rows(store, base, VisionObserved))
    out.publish(CameraStatus(base + timedelta(seconds=start_s), up=True, detail="fake"))
    out.stills.set(Still(JPEG, utcnow()))
    out.publish(Tick(base + timedelta(seconds=start_s)))
    await until(lambda: len(rows(store, base, VisionObserved)) == seen + 1)
    out.stills.set(Still(JPEG, utcnow()))
    out.publish(Tick(base + timedelta(seconds=start_s + 20)))
    await until(lambda: len(rows(store, base, VisionObserved)) == seen + 2)
    out.publish(Tick(base + timedelta(seconds=start_s + 21)))


async def test_two_stomach_checks_send_one_emergency_with_a_snapshot_and_a_row(tmp_path: Path):
    cfg, base = settings(tmp_path), utcnow()
    store = Store.open(cfg.data_dir)
    camera, vision, notifier = FakeCamera(), ScriptedVision([STOMACH] * 2), RecordingNotifier()
    facts_file = tmp_path / "night.jsonl"

    async with running(cfg, camera, vision, notifier, record_facts=facts_file) as out:
        await two_stomach_checks(out, store, base, start_s=60)
        await until(lambda: len(rows(store, base, AlertSent)) == 1)

    [(alert, attached)] = notifier.delivered
    assert (alert.kind, alert.priority) == (AlertKind.ON_STOMACH, Priority.EMERGENCY)
    assert alert.at == base + timedelta(seconds=81)
    assert attached is not None and attached.jpeg == JPEG
    [row] = rows(store, base, AlertSent)
    assert row.delivered and row.alert == alert and row.snapshot is not None
    assert store.snapshot_path(row.snapshot).read_bytes() == JPEG
    assert [type(f).__name__ for f in factlog.read(facts_file)] == [
        "CameraStatus",
        "Tick",
        "VisionDone",
        "Tick",
        "VisionDone",
        "Tick",
        "DeliveryResult",
    ]


async def test_restart_within_cooldown_does_not_resend(tmp_path: Path):
    cfg, base = settings(tmp_path), utcnow()
    store = Store.open(cfg.data_dir)
    vision, notifier = ScriptedVision([STOMACH] * 4), RecordingNotifier()

    async with running(cfg, FakeCamera(), vision, notifier) as out:
        await two_stomach_checks(out, store, base, start_s=60)
        await until(lambda: len(rows(store, base, AlertSent)) == 1)

    async with running(cfg, FakeCamera(), vision, notifier) as out:
        await two_stomach_checks(out, store, base, start_s=100)
        out.publish(SetControls(base + timedelta(seconds=122), nap_mode=True))
        await until(lambda: len(rows(store, base, ControlsChanged)) == 1)
        await asyncio.sleep(0.2)

    assert len(vision.frames) == 4
    assert len(notifier.delivered) == 1
    assert [r.delivered for r in rows(store, base, AlertSent)] == [True]


async def test_restart_after_a_failed_delivery_resends(tmp_path: Path):
    cfg, base = settings(tmp_path), utcnow()
    store = Store.open(cfg.data_dir)
    vision, notifier = ScriptedVision([STOMACH] * 4), RecordingNotifier(fail_first=1)

    async with running(cfg, FakeCamera(), vision, notifier) as out:
        await two_stomach_checks(out, store, base, start_s=60)
        await until(lambda: len(rows(store, base, AlertSent)) == 1)
    assert notifier.delivered == []

    async with running(cfg, FakeCamera(), vision, notifier) as out:
        await two_stomach_checks(out, store, base, start_s=100)
        await until(lambda: len(rows(store, base, AlertSent)) == 2)

    [(alert, _)] = notifier.delivered
    assert alert.at == base + timedelta(seconds=121)
    assert [(r.delivered, r.detail) for r in rows(store, base, AlertSent)] == [
        (False, "phone unreachable"),
        (True, ""),
    ]


async def test_startup_flag_persists_as_the_controls_row(tmp_path: Path):
    cfg, base = settings(tmp_path), utcnow()
    store = Store.open(cfg.data_dir)
    nap = SetControls(base, nap_mode=True)

    async with running(cfg, FakeCamera(), ScriptedVision([]), RecordingNotifier(), nap):
        await until(lambda: len(rows(store, base, ControlsChanged)) == 1)

    assert store.restore(utcnow(), 600).controls.nap_mode is True


async def test_a_second_process_on_the_same_data_dir_exits(tmp_path: Path):
    cfg = settings(tmp_path)
    async with running(cfg, FakeCamera(), ScriptedVision([]), RecordingNotifier()):
        second = backends(FakeCamera(), ScriptedVision([]), RecordingNotifier())
        with pytest.raises(SystemExit, match="another babymon process"):
            await app.run(cfg, None, None, second)


async def test_sigterm_stops_every_actor_and_run_returns(tmp_path: Path):
    cfg, camera = settings(tmp_path), FakeCamera()
    task = asyncio.create_task(
        app.run(cfg, None, None, backends(camera, ScriptedVision([]), RecordingNotifier()))
    )
    await camera.outputs
    os.kill(os.getpid(), signal.SIGTERM)
    async with asyncio.timeout(5):
        assert await task is None


async def test_ticks_strictly_increase_even_when_the_clock_does_not(monkeypatch):
    startup = utcnow()
    monkeypatch.setattr(app, "TICK_S", 0)
    monkeypatch.setattr(app, "utcnow", lambda: startup)
    ticks: list[Tick] = []
    task = asyncio.create_task(app.run_ticker(ticks.append, startup))
    await until(lambda: len(ticks) >= 3)
    task.cancel()
    instants = [t.at for t in ticks[:3]]
    assert instants[0] > startup
    assert instants == sorted(set(instants))


async def test_samples_reach_the_store_as_finished_minutes(tmp_path: Path):
    cfg, base = settings(tmp_path), utcnow().replace(second=0, microsecond=0)
    store = Store.open(cfg.data_dir)

    async with running(cfg, FakeCamera(), ScriptedVision([]), RecordingNotifier()) as out:
        out.publish(MotionSample(base + timedelta(seconds=5), 0.04))
        out.publish(AudioSample(base + timedelta(seconds=6), 2.0, cry_probability=0.9))
        out.publish(Tick(base + timedelta(seconds=61)))
        await until(lambda: len(store.minutes(base, base + HOUR)) == 1)

    [m] = store.minutes(base, base + HOUR)
    assert (m.at, m.motion_samples, m.audio_samples, m.cry_seconds) == (base, 1, 1, 1)
    assert store.events(base - HOUR, base + HOUR) == []
