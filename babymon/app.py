"""Wiring. The only module that knows every actor. No logic beyond plumbing.

Actors and what each owns:

    camera      ffmpeg process, pipes, MotionMeter        -> facts, stills, pcm
    recorder    its own ffmpeg writing MP4 segments       -> files on disk
    audio       classifier, NoiseFloor                    -> facts
    vision      one in-flight model call                  -> facts
    brain       MonitorState (sole owner)                 -> effects, published Situation
    writer      ordered event appends                     -> Store
    rollup      per-minute motion/audio fold, a brain tap -> minute writer -> Store
    alerter     delivery + AlertSent rows                 -> phones, Store
    retention   periodic Store.sweep                      -> disk
    ticker      1 Hz Tick                                 -> facts
    web         HTTP                                      -> facts (SetControls), Store (reference)

Channels: one asyncio.Queue[Fact] into the brain (many writers, append-only, the
structural serialiser); Latest cells for newest-wins data; two queues out of the
brain (events to the writer, alerts to the alerter).

Failure policy: actors handle their expected failures (ffmpeg exits, model errors,
HTTP errors). An unexpected exception ends the TaskGroup and the process; Docker's
restart policy brings it back, and startup converges from disk (Store.restore,
Store.sweep, fresh ffmpeg). Restart is the recovery path, so it is the tested path.

SIGINT ends the process through KeyboardInterrupt (the CLI maps it to exit 130).
SIGTERM, what `docker stop` sends, cancels every actor and `run` returns. Both paths
cancel the camera and recorder tasks, whose cleanup kills their ffmpeg process group.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
from collections.abc import Awaitable, Callable, Generator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Never, TextIO, assert_never

import uvicorn
from fastapi import FastAPI

from babymon import brain, factlog, rollup
from babymon.audio import AudioClassifier, YamnetClassifier, run_audio
from babymon.brain import MonitorState
from babymon.camera import Camera, CameraOutputs, PcmWindow
from babymon.config import Settings
from babymon.domain import (
    Alert,
    Effect,
    Event,
    Fact,
    Instant,
    MinuteRollup,
    Notify,
    Record,
    RequestVision,
    SetControls,
    Situation,
    Still,
    Tick,
    Tuning,
)
from babymon.latest import Latest
from babymon.notify import Notifier, notifier_for, run_alerter
from babymon.recorder import Recorder
from babymon.rules import Decider, RulesDecider
from babymon.store import InstanceLockError, Store, acquire_instance_lock
from babymon.vision import ClaudeVision, VisionBackend, run_vision
from babymon.web import WebDeps, create_app

log = logging.getLogger(__name__)

TICK_S = 1.0
SWEEP_EVERY_S = 300.0
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def utcnow() -> Instant:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class Backends:
    """The swap points, one per actor that touches the outside world. `from_settings`
    builds the production set; the integration test passes fakes for the same slots and
    keeps the brain, writer, alerter and retention real. None skips that actor."""

    camera: Callable[[CameraOutputs], Awaitable[Never]]
    recorder: Callable[[Path], Awaitable[Never]] | None
    classifier: AudioClassifier
    vision: VisionBackend | None
    notifier: Notifier
    serve_http: Callable[[FastAPI], Awaitable[Never]] | None
    ticker: Callable[[Callable[[Fact], None], Instant], Awaitable[Never]]

    @classmethod
    def from_settings(cls, settings: Settings) -> Backends:
        vision = (
            ClaudeVision(
                settings.vision.api_key.get_secret_value(),
                settings.vision.model,
                api_frame_width=settings.vision.api_frame_width,
            )
            if settings.vision
            else None
        )
        return cls(
            camera=Camera(settings.camera).run,
            recorder=Recorder(settings.camera).run,
            classifier=YamnetClassifier(settings.yamnet_path),
            vision=vision,
            notifier=notifier_for(settings.pushover),
            serve_http=lambda app: serve_http(app, settings.http_port),
            ticker=run_ticker,
        )


class _Stop(Exception):
    """Raised inside the TaskGroup by the SIGTERM watcher so every actor is cancelled."""


async def run(
    settings: Settings,
    initial_controls: SetControls | None = None,
    record_facts: Path | None = None,
    backends: Backends | None = None,
) -> None:
    """Start every actor and block until SIGTERM (returns), SIGINT (KeyboardInterrupt)
    or an actor's unexpected exception (ExceptionGroup, non-zero exit)."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    try:
        lock = acquire_instance_lock(settings.data_dir)
    except InstanceLockError as e:
        raise SystemExit(f"babymon: {e}") from None
    with lock, contextlib.ExitStack() as resources:
        backends = backends or Backends.from_settings(settings)
        tuning = settings.tuning
        capabilities = settings.capabilities()
        now = utcnow()
        store = Store.open(settings.data_dir)
        swept = store.sweep(now)
        log.info("data dir %s: swept %s", settings.data_dir, swept)
        state = brain.initial_state(store.restore(now, tuning.alert_cooldown_s), capabilities, now)
        log.info(
            "controls armed=%s nap_mode=%s; sleep %s; cooldowns %s",
            state.controls.armed,
            state.controls.nap_mode,
            state.sleep.state,
            sorted(k.value for k in state.ledger),
        )
        if not capabilities.vision:
            log.warning("VISION disabled: no API key (set ANTHROPIC_API_KEY)")
        if not capabilities.push:
            log.warning("PUSH disabled: no Pushover keys (set BABYMON_PUSHOVER__*)")

        facts: asyncio.Queue[Fact] = asyncio.Queue()
        events: asyncio.Queue[Event] = asyncio.Queue()
        alerts: asyncio.Queue[Alert] = asyncio.Queue()
        stills: Latest[Still] = Latest()
        pcm: Latest[PcmWindow] = Latest()
        vision_requests: Latest[RequestVision] = Latest()
        published: Latest[Situation] = Latest()
        publish = facts.put_nowait

        minutes: asyncio.Queue[MinuteRollup] = asyncio.Queue()
        taps = [_rollup_tap(tuning, minutes.put_nowait)]
        if record_facts is not None:
            taps.append(
                _fact_recorder(resources.enter_context(record_facts.open("a", buffering=1)))
            )
        if initial_controls is not None:
            publish(initial_controls)

        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGTERM, stop.set)
        resources.callback(loop.remove_signal_handler, signal.SIGTERM)

        async def stop_on_sigterm() -> Never:
            await stop.wait()
            log.info("SIGTERM: stopping")
            raise _Stop

        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(backends.camera(CameraOutputs(publish, stills, pcm)))
                if backends.recorder is not None:
                    tg.create_task(backends.recorder(store.segment_dir))
                tg.create_task(run_audio(pcm, backends.classifier, publish))
                if backends.vision is not None:
                    tg.create_task(
                        run_vision(backends.vision, vision_requests, stills, store, publish)
                    )
                tg.create_task(
                    run_brain(
                        state,
                        RulesDecider(tuning),
                        tuning,
                        facts,
                        published,
                        events,
                        alerts,
                        vision_requests,
                        _tee(taps),
                    )
                )
                tg.create_task(run_writer(store, events))
                tg.create_task(run_minute_writer(store, minutes))
                tg.create_task(run_alerter(alerts, backends.notifier, stills, store, publish))
                tg.create_task(backends.ticker(publish, now))
                tg.create_task(run_retention(store))
                if backends.serve_http is not None:
                    deps = WebDeps(publish=publish, situation=published, stills=stills, store=store)
                    tg.create_task(backends.serve_http(create_app(deps)))
                tg.create_task(stop_on_sigterm())
        except* _Stop:
            pass


def _fact_recorder(sink: TextIO) -> Callable[[Fact], None]:
    def record(fact: Fact) -> None:
        sink.write(factlog.encode(fact) + "\n")

    return record


def _rollup_tap(tuning: Tuning, sink: Callable[[MinuteRollup], None]) -> Callable[[Fact], None]:
    acc: rollup.Accumulator | None = None

    def tap(fact: Fact) -> None:
        nonlocal acc
        acc, finished = rollup.step(acc, fact, tuning)
        if finished is not None:
            sink(finished)

    return tap


def _tee(taps: list[Callable[[Fact], None]]) -> Callable[[Fact], None]:
    def tap(fact: Fact) -> None:
        for each in taps:
            each(fact)

    return tap


async def run_brain(
    state: MonitorState,
    decider: Decider,
    tuning: Tuning,
    facts: asyncio.Queue[Fact],
    published: Latest[Situation],
    events: asyncio.Queue[Event],
    alerts: asyncio.Queue[Alert],
    vision_requests: Latest[RequestVision],
    tap: Callable[[Fact], None] | None = None,
) -> Never:
    """The only place the async Decider meets the pure core. `tap` sees every fact as it
    is dequeued: the minute rollup and `run --record-facts` share it."""
    while True:
        fact = await facts.get()
        if tap is not None:
            tap(fact)
        state, effects = brain.observe(state, fact, tuning)
        if isinstance(fact, Tick):
            situation = brain.situation(state, tuning)
            state, notifies = brain.gate(state, await decider.decide(situation), tuning)
            effects += notifies
            published.set(situation)
        for effect in effects:
            dispatch(effect, events, alerts, vision_requests)


def dispatch(
    effect: Effect,
    events: asyncio.Queue[Event],
    alerts: asyncio.Queue[Alert],
    vision_requests: Latest[RequestVision],
) -> None:
    """Never awaits, so the brain never waits on I/O."""
    match effect:
        case Record(event=event):
            events.put_nowait(event)
        case Notify(alert=alert):
            alerts.put_nowait(alert)
        case RequestVision():
            vision_requests.set(effect)
        case _:
            assert_never(effect)


async def run_writer(store: Store, events: asyncio.Queue[Event]) -> Never:
    while True:
        event = await events.get()
        await asyncio.to_thread(store.append, event)


async def run_minute_writer(store: Store, minutes: asyncio.Queue[MinuteRollup]) -> Never:
    while True:
        minute = await minutes.get()
        await asyncio.to_thread(store.append_minute, minute)


async def run_ticker(publish: Callable[[Fact], None], after: Instant) -> Never:
    """1 Hz. Instants strictly increase and the first is strictly after `after` (the
    startup instant), which is what makes every Tick count in the brain."""
    last = after
    while True:
        await asyncio.sleep(TICK_S)
        at = max(utcnow(), last + timedelta(microseconds=1))
        publish(Tick(at))
        last = at


async def run_retention(store: Store, every_s: float = SWEEP_EVERY_S) -> Never:
    while True:
        await asyncio.sleep(every_s)
        swept = await asyncio.to_thread(store.sweep, utcnow())
        if swept.segments or swept.snapshots or swept.events or swept.minutes:
            log.info("swept %s", swept)


class _Server(uvicorn.Server):
    """uvicorn's own SIGINT/SIGTERM handlers would stop only the HTTP task and leave the
    other actors running. The TaskGroup owns shutdown, so the handlers are not installed."""

    @contextlib.contextmanager
    def capture_signals(self) -> Generator[None]:
        yield


async def serve_http(app: FastAPI, port: int) -> Never:
    config = uvicorn.Config(
        app, host="0.0.0.0", port=port, log_config=None, access_log=False, lifespan="off"
    )
    await _Server(config).serve()
    raise RuntimeError("HTTP server stopped")
