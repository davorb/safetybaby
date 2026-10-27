"""Camera ingest: one supervised ffmpeg process per RTSP session, four outputs.

One RTSP connection feeds everything:

    stream2 --+-- copy video + AAC audio -> 5-min MP4 segments on disk (recorder)
              +-- fps=4, 640x360 gray rawvideo -> fd A -> MotionMeter -> MotionSample facts
              +-- 16 kHz s16le mono          -> fd B -> PcmWindow (1 s) -> Latest
              +-- fps=1, 1280x720 MJPEG      -> fd C -> Still -> Latest

Why one process: one connection to the camera (the Tapo session limit is unknown),
one reconnect point, one place that knows ffmpeg flags. The recorder is not a
separate actor; its retention is the store's job.

Invariant: pipe readers never await a consumer. They drain bytes and publish into
Latest cells or the fact sink, so a slow classifier or vision call cannot back-pressure
ffmpeg and stall the recording.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Never

import numpy as np
import numpy.typing as npt

from babymon.config import CameraConfig
from babymon.domain import Fact, Instant, Still
from babymon.latest import Latest

FRAME_W, FRAME_H = 640, 360
FRAME_BYTES = FRAME_W * FRAME_H
PCM_RATE = 16_000
PCM_WINDOW_BYTES = PCM_RATE * 2


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


def ffmpeg_argv(cfg: CameraConfig, segment_dir: Path, fds: SessionFds) -> list[str]:
    """Pure. The one place ffmpeg flags live.

    Input: -rtsp_transport tcp -timeout <us> -i <stream2 url>
    Recorder output: -map 0:v -map 0:a -c:v copy -c:a aac -b:a 32k -f segment
        -segment_time 300 -segment_atclocktime 1 -reset_timestamps 1 -strftime 1
        -segment_format_options movflags=+frag_keyframe+empty_moov+default_base_moof
        <segment_dir>/%Y%m%d-%H%M%S.mp4
      Fragmented MP4 so the segment being written (and one cut short by a crash) plays.
      ffmpeg runs with TZ=UTC so file names are UTC start times.
    Frames: -map 0:v -vf fps=4,scale=640:360,format=gray -f rawvideo pipe:<fds.frames>
    PCM:    -map 0:a -ac 1 -ar 16000 -f s16le pipe:<fds.pcm>
    Stills: -map 0:v -vf fps=1,scale=1280:720 -q:v 4 -f mjpeg pipe:<fds.stills>
    """
    raise NotImplementedError


class MotionMeter:
    """Frame differencing. Owns the previous frame; nothing else does.

    level = fraction of ROI pixels whose |blurred(cur) - blurred(prev)| exceeds
    noise_threshold. The blur and threshold absorb infrared sensor noise. The ROI
    (crib area, from config) keeps a parent walking past from reading as baby motion.
    """

    def __init__(self, roi: tuple[int, int, int, int] | None, noise_threshold: int) -> None:
        raise NotImplementedError

    def level(self, gray: npt.NDArray[np.uint8]) -> float:
        raise NotImplementedError


def split_jpegs(buffer: bytearray) -> list[bytes]:
    """Pure. Pop every complete SOI..EOI JPEG off the front of buffer, leaving any
    partial tail in place."""
    raise NotImplementedError


class Camera:
    def __init__(self, cfg: CameraConfig, segment_dir: Path) -> None:
        raise NotImplementedError

    async def run(self, out: CameraOutputs) -> Never:
        """Supervise forever. Idempotent by construction: each attempt starts from
        nothing (new pipes, new process, new MotionMeter).

        loop:
            open three os.pipe()s, spawn ffmpeg with pass_fds, close child ends
            reader tasks per pipe; watchdog: no frame bytes for cfg.stall_s -> kill
            on first frame: publish CameraStatus(up=True)
            on exit/stall/kill: publish CameraStatus(up=False, detail=last stderr line)
            sleep backoff (1, 2, 5, 10, 30 s cap; reset after 60 s healthy)
        Cancellation kills the child process group; no orphan ffmpeg survives.
        """
        raise NotImplementedError
