"""Camera and recorder: pure helpers, then the supervised processes against a stub ffmpeg."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
from datetime import UTC
from pathlib import Path

import numpy as np
import pytest

from babymon import camera as camera_module
from babymon.camera import (
    PCM_WINDOW_BYTES,
    Camera,
    CameraOutputs,
    MotionMeter,
    SessionFds,
    ffmpeg_argv,
    redact,
    split_jpegs,
    supervise,
)
from babymon.config import CameraConfig
from babymon.domain import CameraStatus, Fact, MotionSample
from babymon.latest import Latest
from babymon.recorder import Recorder, recorder_argv

SECRET = "hunter2"
CFG = CameraConfig(user="user", password=SECRET, stall_s=0.6)
STUB = Path(__file__).parent / "stub_ffmpeg.py"


def gray(fill: int = 0, block: tuple[int, int, int, int] | None = None, value: int = 255):
    frame = np.full((360, 640), fill, dtype=np.uint8)
    if block:
        x, y, w, h = block
        frame[y : y + h, x : x + w] = value
    return frame


def test_ffmpeg_argv_names_one_pipe_per_output_and_nothing_for_recording():
    argv = ffmpeg_argv(CFG, SessionFds(frames=11, pcm=12, stills=13))
    joined = " ".join(argv)
    assert "fps=4,scale=640:360,format=gray -f rawvideo" in joined
    assert "pipe:11" in argv and "pipe:12" in argv and "pipe:13" in argv
    assert "-ar 16000 -f s16le" in joined
    assert "fps=1,scale=1280:720 -q:v 4 -f mjpeg" in joined
    assert "-rtsp_transport tcp" in joined
    assert "segment" not in joined


def test_recorder_argv_writes_utc_named_fragmented_segments(tmp_path):
    argv = recorder_argv(CFG, tmp_path)
    joined = " ".join(argv)
    assert "-c:v copy -c:a aac -b:a 32k" in joined
    assert "-segment_time 300 -segment_atclocktime 1 -reset_timestamps 1 -strftime 1" in joined
    assert "frag_keyframe+empty_moov+default_base_moof" in joined
    assert argv[-1] == str(tmp_path / "%Y%m%d-%H%M%S.mp4")


def test_split_jpegs_leaves_a_partial_tail_for_the_next_chunk():
    one, two = b"\xff\xd8aaa\xff\xd9", b"\xff\xd8bbbbbb\xff\xd9"
    stream = one + two
    buffer = bytearray(stream[: len(one) + 4])
    assert split_jpegs(buffer) == [one]
    assert bytes(buffer) == two[:4]
    buffer += stream[len(one) + 4 :]
    assert split_jpegs(buffer) == [two]
    assert buffer == b""


def test_split_jpegs_drops_bytes_before_the_first_start_marker():
    buffer = bytearray(b"junk\xff\xd8x\xff\xd9")
    assert split_jpegs(buffer) == [b"\xff\xd8x\xff\xd9"]


def test_motion_is_zero_for_the_first_frame_and_for_a_still_scene():
    meter = MotionMeter(None, 12)
    assert meter.level(gray(50)) == 0.0
    assert meter.level(gray(50)) == 0.0


def test_motion_is_the_changed_fraction_of_the_frame():
    meter = MotionMeter(None, 12)
    meter.level(gray(0))
    level = meter.level(gray(0, block=(0, 0, 320, 180)))
    assert level == pytest.approx(0.25, abs=0.02)


def test_motion_ignores_changes_outside_the_roi():
    meter = MotionMeter((0, 0, 100, 100), 12)
    meter.level(gray(0))
    assert meter.level(gray(0, block=(300, 200, 100, 100))) == 0.0
    assert meter.level(gray(0, block=(0, 0, 100, 100))) > 0.5


def test_motion_ignores_sensor_noise_below_the_threshold():
    meter = MotionMeter(None, 12)
    meter.level(gray(100))
    noisy = gray(100) + np.random.default_rng(0).integers(0, 6, (360, 640), dtype=np.uint8)
    assert meter.level(noisy) == 0.0


def test_redact_removes_url_credentials_and_a_bare_password():
    line = f"rtsp://user:{SECRET}@host/s: refused, token {SECRET}"
    cleaned = redact(line, CFG)
    assert SECRET not in cleaned
    assert "refused" in cleaned


async def test_supervisor_backs_off_per_failure_and_resets_after_a_healthy_run(monkeypatch):
    delays: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(camera_module.asyncio, "sleep", fake_sleep)
    statuses: list[tuple[bool, str]] = []
    runs = 0

    async def attempt(on_up):
        nonlocal runs
        runs += 1
        if runs == 6:
            on_up()
        if runs == 7:
            raise asyncio.CancelledError
        return f"run {runs}"

    with pytest.raises(asyncio.CancelledError):
        await supervise(attempt, lambda up, why: statuses.append((up, why)), healthy_s=0.0)
    assert delays[:5] == [1.0, 2.0, 5.0, 10.0, 30.0]
    assert delays[5:] == [1.0]
    assert statuses[-1] == (False, "run 6")


@pytest.fixture
def stub(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    wrapper = bin_dir / "ffmpeg"
    wrapper.write_text(f'#!/bin/sh\nexec {sys.executable} {STUB} "$@"\n')
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "invocations.jsonl"
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("STUB_LOG", str(log))
    monkeypatch.setattr(camera_module, "BACKOFF_S", (0.05,))

    class Stub:
        def mode(self, mode: str) -> None:
            monkeypatch.setenv("STUB_MODE", mode)

        def invocations(self) -> list[dict]:
            if not log.exists():
                return []
            return [json.loads(line) for line in log.read_text().splitlines()]

    return Stub()


async def until(predicate, timeout: float = 8.0):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.02)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class Sink:
    def __init__(self) -> None:
        self.facts: list[Fact] = []
        self.out = CameraOutputs(publish=self.facts.append, stills=Latest(), pcm=Latest())

    def of(self, kind: type) -> list:
        return [f for f in self.facts if isinstance(f, kind)]


async def test_camera_publishes_status_motion_pcm_and_stills_then_dies_with_its_task(stub):
    stub.mode("camera_ok")
    sink = Sink()
    task = asyncio.create_task(Camera(CFG).run(sink.out))
    await until(lambda: len(sink.of(MotionSample)) == 2 and sink.out.stills.get())

    assert sink.of(CameraStatus)[0].up is True
    first, second = sink.of(MotionSample)
    assert first.level == 0.0 and second.level > 0.9
    assert first.at.tzinfo is UTC and second.at >= first.at
    pcm = sink.out.pcm.get()
    assert pcm is not None and pcm.samples.shape == (PCM_WINDOW_BYTES // 2,)
    assert pcm.samples.dtype == np.float32
    still = sink.out.stills.get()
    assert still is not None and still.jpeg == b"\xff\xd8stub-image\xff\xd9"

    (pid,) = [i["pid"] for i in stub.invocations()]
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert not alive(pid)


async def test_camera_reports_down_with_a_redacted_reason_and_respawns(stub):
    stub.mode("fail")
    sink = Sink()
    task = asyncio.create_task(Camera(CFG).run(sink.out))
    await until(lambda: len(stub.invocations()) >= 2)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    down = sink.of(CameraStatus)
    assert down and all(not s.up for s in down)
    assert "Connection refused" in down[0].detail
    assert SECRET not in repr(sink.facts)


async def test_camera_kills_a_stalled_child_and_restarts(stub):
    stub.mode("camera_stall")
    sink = Sink()
    task = asyncio.create_task(Camera(CFG).run(sink.out))
    await until(lambda: len(stub.invocations()) >= 2)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    statuses = sink.of(CameraStatus)
    assert [s.up for s in statuses[:2]] == [True, False]
    assert "stalled" in statuses[1].detail
    first_pid = stub.invocations()[0]["pid"]
    assert not alive(first_pid)


async def test_recorder_runs_in_utc_and_respawns_after_the_child_is_killed(stub, tmp_path):
    stub.mode("recorder_ok")
    segments = tmp_path / "segments"
    task = asyncio.create_task(Recorder(CFG, segment_s=10).run(segments))
    await until(lambda: len(stub.invocations()) == 1)
    first = stub.invocations()[0]
    assert first["tz"] == "UTC"
    assert first["argv"][-1] == str(segments / "%Y%m%d-%H%M%S.mp4")
    assert segments.is_dir()

    os.kill(first["pid"], 9)
    await until(lambda: len(stub.invocations()) == 2)
    second_pid = stub.invocations()[1]["pid"]
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert not alive(second_pid)


async def test_recorder_restarts_when_output_time_stops_advancing(stub, tmp_path):
    stub.mode("recorder_stall")
    task = asyncio.create_task(Recorder(CFG).run(tmp_path / "segments"))
    await until(lambda: len(stub.invocations()) >= 2)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert not alive(stub.invocations()[0]["pid"])
