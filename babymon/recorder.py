"""Recorder: its own supervised ffmpeg writing fixed-length MP4 segments.

Independent of the camera actor (its own RTSP session), so a fault in one never gaps
the other. The store owns the segment directory's layout and retention; this module
only fills it.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Never

from babymon.camera import (
    LastLine,
    exit_detail,
    first_done,
    reap,
    redact,
    rtsp_input,
    spawn,
    supervise,
    watch_stall,
)
from babymon.config import CameraConfig

log = logging.getLogger(__name__)

SEGMENT_S = 300


def recorder_argv(cfg: CameraConfig, segment_dir: Path, segment_s: int = SEGMENT_S) -> list[str]:
    """Pure. Video copied, audio to AAC (the camera's pcm_alaw cannot go in MP4).
    Fragmented MP4 with per-packet flushing, so the segment being written, or one cut
    short by a crash or SIGKILL, still plays (measured; plain MP4 leaves no moov).
    Names are strftime patterns, so the child must run with TZ=UTC. `-progress` writes
    key=value lines to stdout, which the supervisor reads as the liveness signal."""
    return [
        "ffmpeg", "-nostdin", "-v", "error", "-nostats", "-progress", "pipe:1",
        *rtsp_input(cfg),
        "-map", "0", "-c:v", "copy", "-c:a", "aac", "-b:a", "32k",
        "-f", "segment", "-segment_time", str(segment_s), "-segment_atclocktime", "1",
        "-reset_timestamps", "1", "-strftime", "1",
        "-segment_format_options",
        "movflags=+frag_keyframe+empty_moov+default_base_moof:flush_packets=1",
        str(segment_dir / "%Y%m%d-%H%M%S.mp4"),
    ]  # fmt: skip


class Recorder:
    def __init__(self, cfg: CameraConfig, segment_s: int = SEGMENT_S) -> None:
        self._cfg = cfg
        self._segment_s = segment_s

    async def run(self, segment_dir: Path) -> Never:
        """Supervise forever; same restart shape as the camera. Stalled means the
        muxer's output time stopped advancing for cfg.stall_s."""
        segment_dir.mkdir(parents=True, exist_ok=True)

        def on_status(up: bool, detail: str) -> None:
            if up:
                log.info("recorder up")
            else:
                log.warning("recorder down: %s", detail)

        async def attempt(on_up: Callable[[], None]) -> str:
            return await self._attempt(segment_dir, on_up)

        await supervise(attempt, on_status)

    async def _attempt(self, segment_dir: Path, on_up: Callable[[], None]) -> str:
        cfg = self._cfg
        argv = recorder_argv(cfg, segment_dir, self._segment_s)
        proc = await spawn(argv, env={"TZ": "UTC"}, stdout=asyncio.subprocess.PIPE)
        try:
            assert proc.stdout is not None and proc.stderr is not None
            last = LastLine(lambda line: redact(line, cfg))
            progress = _Progress(on_up)
            drain = asyncio.create_task(last.drain(proc.stderr))
            stall = asyncio.create_task(watch_stall(lambda: progress.advanced_at, cfg.stall_s))
            tasks: list[asyncio.Task[object]] = [
                asyncio.create_task(progress.read(proc.stdout)),
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
            await asyncio.shield(reap(proc))


class _Progress:
    """Tracks ffmpeg's `out_time_us=` lines; the clock moves only when the value does."""

    def __init__(self, on_up: Callable[[], None]) -> None:
        self.advanced_at = time.monotonic()
        self._out_time = 0
        self._on_up = on_up
        self._up = False

    async def read(self, stream: asyncio.StreamReader) -> None:
        async for raw in stream:
            key, _, value = raw.decode(errors="replace").strip().partition("=")
            if key != "out_time_us" or not value.lstrip("-").isdigit():
                continue
            out_time = int(value)
            if out_time > self._out_time:
                self._out_time = out_time
                self.advanced_at = time.monotonic()
                if not self._up:
                    self._up = True
                    self._on_up()
