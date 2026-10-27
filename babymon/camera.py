"""Camera ingest: one supervised ffmpeg process on stream2 with three analysis pipes.

    stream2 --+-- fps=4, 640x360 gray rawvideo -> fd A -> MotionMeter -> MotionSample facts
              +-- 16 kHz s16le mono          -> fd B -> PcmWindow (1 s) -> Latest
              +-- fps=1, 1280x720 MJPEG      -> fd C -> Still -> Latest

Recording is a second ffmpeg (recorder.py): the C210 serves concurrent sessions, so a
recorder fault never blinds the analysis and a stall here never gaps the recording.
Both supervise their child with `supervise` and `reap` below.

Invariant: pipe readers never await a consumer. They drain bytes and publish into
Latest cells or the fact sink, so a slow classifier or vision call cannot back-pressure
ffmpeg.
"""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import logging
import os
import re
import signal
import tempfile
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Never

import cv2
import numpy as np
import numpy.typing as npt

from babymon.config import CameraConfig, Settings
from babymon.domain import CameraStatus, Fact, Instant, MotionSample, Still
from babymon.latest import Latest

log = logging.getLogger(__name__)

FRAME_W, FRAME_H = 640, 360
FRAME_BYTES = FRAME_W * FRAME_H
PCM_RATE = 16_000
PCM_WINDOW_BYTES = PCM_RATE * 2

BACKOFF_S = (1.0, 2.0, 5.0, 10.0, 30.0)
HEALTHY_S = 60.0
RTSP_TIMEOUT_US = 10_000_000


@dataclass(frozen=True, slots=True)
class PcmWindow:
    """Exactly one second of 16 kHz mono audio, float32 in -1..1."""

    samples: npt.NDArray[np.float32]
    at: Instant


@dataclass(frozen=True, slots=True)
class CameraOutputs:
    publish: Callable[[Fact], None]
    """Non-blocking append to the brain inbox. Receives MotionSample and CameraStatus."""
    stills: Latest[Still]
    pcm: Latest[PcmWindow]


@dataclass(frozen=True, slots=True)
class SessionFds:
    frames: int
    pcm: int
    stills: int


def now() -> Instant:
    return datetime.now(UTC)


_URL_CREDENTIALS = re.compile(r"//[^/@\s]*@")


def redact(line: str, cfg: CameraConfig) -> str:
    """Strip the camera credentials from text that may quote the RTSP URL."""
    line = _URL_CREDENTIALS.sub("//***@", line)
    secret = cfg.password.get_secret_value()
    return line.replace(secret, "***") if secret else line


def rtsp_input(cfg: CameraConfig) -> list[str]:
    return ["-rtsp_transport", "tcp", "-timeout", str(RTSP_TIMEOUT_US), "-i", cfg.rtsp_url()]


def ffmpeg_argv(cfg: CameraConfig, fds: SessionFds) -> list[str]:
    return [
        "ffmpeg", "-nostdin", "-v", "error",
        *rtsp_input(cfg),
        "-map", "0:v", "-vf", f"fps=4,scale={FRAME_W}:{FRAME_H},format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray", "-flush_packets", "1", f"pipe:{fds.frames}",
        "-map", "0:a", "-ac", "1", "-ar", str(PCM_RATE),
        "-f", "s16le", "-flush_packets", "1", f"pipe:{fds.pcm}",
        "-map", "0:v", "-vf", "fps=1,scale=1280:720", "-q:v", "4",
        "-f", "mjpeg", "-flush_packets", "1", f"pipe:{fds.stills}",
    ]  # fmt: skip


class MotionMeter:
    """Frame differencing. Owns the previous frame; nothing else does.

    level = fraction of ROI pixels whose |blurred(cur) - blurred(prev)| exceeds
    noise_threshold. The blur and threshold absorb infrared sensor noise. The ROI
    (crib area, from config) keeps a parent walking past from reading as baby motion.
    """

    def __init__(self, roi: tuple[int, int, int, int] | None, noise_threshold: int) -> None:
        self._roi = roi
        self._threshold = noise_threshold
        self._previous: npt.NDArray[np.uint8] | None = None

    def level(self, gray: npt.NDArray[np.uint8]) -> float:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        previous, self._previous = self._previous, blurred
        if previous is None:
            return 0.0
        diff = cv2.absdiff(blurred, previous)
        if self._roi is not None:
            x, y, w, h = self._roi
            diff = diff[y : y + h, x : x + w]
        return float(np.count_nonzero(diff > self._threshold)) / diff.size


def split_jpegs(buffer: bytearray) -> list[bytes]:
    """Pure. Pop every complete SOI..EOI JPEG off the front of buffer, leaving any
    partial tail in place.

    ffmpeg's MJPEG output carries no 0xFF 0xD9 inside entropy-coded data (0xFF is
    stuffed with 0x00) and no embedded thumbnails, so the first EOI after an SOI ends
    the image."""
    images: list[bytes] = []
    while True:
        start = buffer.find(b"\xff\xd8")
        if start < 0:
            buffer.clear()
            return images
        end = buffer.find(b"\xff\xd9", start + 2)
        if end < 0:
            del buffer[:start]
            return images
        images.append(bytes(buffer[start : end + 2]))
        del buffer[: end + 2]


async def supervise(
    attempt: Callable[[Callable[[], None]], Awaitable[str]],
    on_status: Callable[[bool, str], None],
    backoff: Sequence[float] | None = None,
    healthy_s: float = HEALTHY_S,
) -> Never:
    """Run `attempt` forever. An attempt calls its argument once it is healthy and
    returns why it ended. Backoff steps up per consecutive failure and resets after an
    attempt stayed healthy for `healthy_s`."""
    backoff = BACKOFF_S if backoff is None else backoff
    step = 0
    while True:
        up_since: float | None = None

        def on_up() -> None:
            nonlocal up_since
            up_since = time.monotonic()
            on_status(True, "")

        detail = await attempt(on_up)
        on_status(False, detail)
        if up_since is not None and time.monotonic() - up_since >= healthy_s:
            step = 0
        await asyncio.sleep(backoff[min(step, len(backoff) - 1)])
        step += 1


async def spawn(
    argv: list[str],
    *,
    env: dict[str, str] | None = None,
    pass_fds: tuple[int, ...] = (),
    stdout: int | None = asyncio.subprocess.DEVNULL,
) -> asyncio.subprocess.Process:
    """Start a child in its own process group so `reap` takes down anything it forks.

    The child also asks the kernel to SIGKILL it when this process dies, so a kill -9
    of babymon cannot leave a recorder ffmpeg writing segments on its own (measured
    2026-10-27: the recorder's only pipe is -progress, whose write errors ffmpeg ignores).
    """
    return await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=stdout,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, **(env or {})},
        pass_fds=pass_fds,
        start_new_session=True,
        preexec_fn=_die_with_parent,
    )


def _die_with_parent() -> None:
    PR_SET_PDEATHSIG = 1
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGKILL)


async def reap(proc: asyncio.subprocess.Process) -> None:
    """Idempotent: kill the process group and wait. Safe when the child already exited."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)
    await proc.wait()


class LastLine:
    """Drains a child's stderr (a full pipe would block ffmpeg) and keeps the last line."""

    def __init__(self, redact_line: Callable[[str], str]) -> None:
        self.text = ""
        self._redact = redact_line

    async def drain(self, stream: asyncio.StreamReader) -> None:
        async for raw in stream:
            line = raw.decode(errors="replace").strip()
            if line:
                self.text = self._redact(line)


async def first_done(tasks: Sequence[asyncio.Task[object]]) -> None:
    """Wait for the first task to finish, cancel the rest, re-raise a task's exception."""
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    for task in done:
        if not task.cancelled() and (error := task.exception()) is not None:
            raise error


async def watch_stall(last_progress: Callable[[], float], stall_s: float) -> str:
    """Returns when `last_progress()` (a time.monotonic value) is older than stall_s."""
    while True:
        await asyncio.sleep(min(1.0, stall_s / 4))
        idle = time.monotonic() - last_progress()
        if idle > stall_s:
            return f"stalled: no data for {idle:.0f}s"


async def exit_detail(
    proc: asyncio.subprocess.Process, last: LastLine, drain: asyncio.Task[None]
) -> str:
    """Reap the child, let the stderr drain reach EOF so its last line is complete."""
    await reap(proc)
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(drain, 1.0)
    code = f"exit code {proc.returncode}"
    return f"{last.text} ({code})" if last.text else f"ffmpeg exited, {code}"


async def _open_reader(fd: int) -> tuple[asyncio.StreamReader, asyncio.ReadTransport]:
    reader = asyncio.StreamReader(limit=2 * FRAME_BYTES)
    loop = asyncio.get_running_loop()
    transport, _ = await loop.connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader), os.fdopen(fd, "rb", buffering=0)
    )
    return reader, transport


class _Session:
    """Everything one ffmpeg attempt owns: readers, the meter, the progress clock."""

    def __init__(self, cfg: CameraConfig, out: CameraOutputs, on_up: Callable[[], None]) -> None:
        self.cfg = cfg
        self.out = out
        self.on_up = on_up
        self.meter = MotionMeter(cfg.motion_roi, cfg.motion_noise_threshold)
        self.last_frame = time.monotonic()
        self.up = False

    async def read_frames(self, reader: asyncio.StreamReader) -> str:
        while True:
            try:
                data = await reader.readexactly(FRAME_BYTES)
            except asyncio.IncompleteReadError:
                return "video stream ended"
            self.last_frame = time.monotonic()
            if not self.up:
                self.up = True
                self.on_up()
                self.out.publish(CameraStatus(at=now(), up=True, detail=""))
            gray = np.frombuffer(data, dtype=np.uint8).reshape(FRAME_H, FRAME_W)
            self.out.publish(MotionSample(at=now(), level=self.meter.level(gray)))

    async def read_pcm(self, reader: asyncio.StreamReader) -> str:
        while True:
            try:
                data = await reader.readexactly(PCM_WINDOW_BYTES)
            except asyncio.IncompleteReadError:
                return "audio stream ended"
            samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
            self.out.pcm.set(PcmWindow(samples=samples, at=now()))

    async def read_stills(self, reader: asyncio.StreamReader) -> str:
        buffer = bytearray()
        while chunk := await reader.read(65536):
            buffer += chunk
            for jpeg in split_jpegs(buffer):
                self.out.stills.set(Still(jpeg=jpeg, at=now()))
        return "still stream ended"


class Camera:
    def __init__(self, cfg: CameraConfig) -> None:
        self._cfg = cfg

    async def run(self, out: CameraOutputs) -> Never:
        """Supervise forever. Idempotent by construction: each attempt starts from
        nothing (new pipes, new process, new MotionMeter).

        Publishes CameraStatus(up=True) on the first frame and
        CameraStatus(up=False, detail=last stderr line) when an attempt ends. A stall
        (no frame bytes for cfg.stall_s) kills the child. Cancellation kills the
        child's process group; no orphan ffmpeg survives.
        """

        def on_status(up: bool, detail: str) -> None:
            if not up:
                log.warning("camera down: %s", detail)
                out.publish(CameraStatus(at=now(), up=False, detail=detail))

        async def attempt(on_up: Callable[[], None]) -> str:
            return await self._attempt(out, on_up)

        await supervise(attempt, on_status)

    async def _attempt(self, out: CameraOutputs, on_up: Callable[[], None]) -> str:
        cfg = self._cfg
        pipes = [os.pipe() for _ in range(3)]
        read_fds = [r for r, _ in pipes]
        write_fds = [w for _, w in pipes]
        transports: list[asyncio.ReadTransport] = []
        proc: asyncio.subprocess.Process | None = None
        try:
            argv = ffmpeg_argv(cfg, SessionFds(*write_fds))
            try:
                proc = await spawn(argv, pass_fds=tuple(write_fds))
            finally:
                for fd in write_fds:
                    os.close(fd)
            assert proc.stderr is not None
            readers = []
            while read_fds:
                reader, transport = await _open_reader(read_fds.pop(0))
                readers.append(reader)
                transports.append(transport)
            session = _Session(cfg, out, on_up)
            last = LastLine(lambda line: redact(line, cfg))
            drain = asyncio.create_task(last.drain(proc.stderr))
            stall = asyncio.create_task(watch_stall(lambda: session.last_frame, cfg.stall_s))
            tasks: list[asyncio.Task[object]] = [
                asyncio.create_task(session.read_frames(readers[0])),
                asyncio.create_task(session.read_pcm(readers[1])),
                asyncio.create_task(session.read_stills(readers[2])),
                stall,
                asyncio.create_task(proc.wait()),
            ]
            try:
                await first_done(tasks)
                if stall.done() and not stall.cancelled():
                    await reap(proc)
                    return stall.result()
                return await exit_detail(proc, last, drain)
            finally:
                drain.cancel()
        finally:
            if proc is not None:
                await asyncio.shield(reap(proc))
            for transport in transports:
                transport.close()
            for fd in read_fds:
                os.close(fd)


@dataclass(frozen=True, slots=True)
class CameraProbe:
    seconds: float
    frames: int
    pcm_windows: int
    stills: int
    motion_levels: list[float]
    statuses: list[CameraStatus]

    def summary(self) -> str:
        levels = self.motion_levels
        peak = max(levels, default=0.0)
        mean = sum(levels) / len(levels) if levels else 0.0
        return (
            f"{self.seconds:.0f}s: {self.frames / self.seconds:.2f} frames/s, "
            f"{self.pcm_windows / self.seconds:.2f} pcm windows/s, "
            f"{self.stills / self.seconds:.2f} stills/s, "
            f"motion mean {mean:.4f} peak {peak:.4f}, "
            f"status changes: {[(s.up, s.detail) for s in self.statuses]}"
        )


GRAB_TIMEOUT_S = 30.0


async def grab_still(settings: Settings, timeout_s: float = GRAB_TIMEOUT_S) -> Still:
    """One fresh 1280x720 frame from a short-lived camera session."""
    out = CameraOutputs(publish=lambda fact: None, stills=Latest(), pcm=Latest())
    task = asyncio.create_task(Camera(settings.camera).run(out))
    try:
        async with asyncio.timeout(timeout_s):
            _, still = await out.stills.newer_than(0)
    except TimeoutError:
        raise SystemExit(f"camera: no frame within {timeout_s:.0f} s") from None
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    return still


async def probe(settings: Settings, seconds: float) -> None:
    """`babymon probe camera --seconds N`: the camera actor and a recorder, side by
    side, into a temporary segment directory. Prints rates, motion and the segments."""
    from babymon.recorder import Recorder

    with tempfile.TemporaryDirectory(prefix="babymon-probe-") as tmp:
        segment_dir = Path(tmp)
        recorder = asyncio.create_task(Recorder(settings.camera).run(segment_dir))
        try:
            report = await probe_camera(settings.camera, seconds)
            segments = sorted(segment_dir.glob("*.mp4"))
            print(report.summary())
            for path in segments:
                print(f"segment {path.name}: {path.stat().st_size} bytes")
        finally:
            recorder.cancel()
            await asyncio.gather(recorder, return_exceptions=True)
    if report.frames == 0:
        last = report.statuses[-1].detail if report.statuses else "no status reported"
        raise SystemExit(f"camera probe: no frames in {seconds:.0f} s: {last}")


async def probe_camera(cfg: CameraConfig, seconds: float) -> CameraProbe:
    """Run the camera actor alone for `seconds` and count what it produced."""
    facts: list[Fact] = []
    out = CameraOutputs(publish=facts.append, stills=Latest(), pcm=Latest())
    task = asyncio.create_task(Camera(cfg).run(out))
    try:
        await asyncio.sleep(seconds)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    motion = [f.level for f in facts if isinstance(f, MotionSample)]
    return CameraProbe(
        seconds=seconds,
        frames=len(motion),
        pcm_windows=out.pcm.version,
        stills=out.stills.version,
        motion_levels=motion,
        statuses=[f for f in facts if isinstance(f, CameraStatus)],
    )
