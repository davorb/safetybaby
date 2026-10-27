"""Audio actor: PCM windows -> AudioSample facts.

Owns the noise-floor estimate and the classifier instance. The classifier is the swap
point; nothing outside this module knows which model it is.
"""

from __future__ import annotations

import asyncio
import math
import subprocess
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Never, Protocol

import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from babymon.camera import PcmWindow
from babymon.config import Settings
from babymon.domain import AudioSample, Fact
from babymon.latest import Latest


class AudioClassifier(Protocol):
    def cry_probability(self, samples: npt.NDArray[np.float32]) -> float:
        """One second of 16 kHz mono float32 audio -> P(cry) in 0..1. Synchronous and
        CPU-bound; the actor runs it in a worker thread."""
        ...


class YamnetClassifier:
    """YAMNet ONNX on CPU (onnxruntime). Input `waveform`, output `output_0` of shape
    (patches, 521). P(cry) = max over patches of max(scores[19], scores[20])
    ("Crying, sobbing", "Baby cry, infant cry")."""

    CRY_CLASSES = (19, 20)

    def __init__(self, model_path: Path) -> None:
        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    def cry_probability(self, samples: npt.NDArray[np.float32]) -> float:
        (scores,) = self._session.run(["output_0"], {"waveform": samples})
        return float(scores[:, self.CRY_CLASSES].max())


class NoiseFloor:
    """Rolling 10th percentile of per-second RMS dBFS over the last 10 minutes.
    The white-noise machine sets this floor; loudness is reported above it."""

    def __init__(self, window_s: int = 600) -> None:
        self._levels: deque[float] = deque(maxlen=window_s)

    def above_floor(self, rms_db: float) -> float:
        """Record this second's level and return it relative to the current floor.
        One call per second of audio, so the window length is a count of calls."""
        self._levels.append(rms_db)
        return rms_db - float(np.percentile(self._levels, 10))


SILENCE_DBFS = -100.0


def rms_dbfs(samples: npt.NDArray[np.float32]) -> float:
    """RMS level relative to full scale (1.0), floored at SILENCE_DBFS for digital silence."""
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return max(20.0 * math.log10(rms), SILENCE_DBFS) if rms > 0 else SILENCE_DBFS


async def run_audio(
    pcm: Latest[PcmWindow], classifier: AudioClassifier, publish: Callable[[Fact], None]
) -> Never:
    """for each newer PcmWindow: rms, floor, await to_thread(classifier...),
    publish(AudioSample(at=window.at, ...)). A window missed while classifying is
    dropped; the brain treats audio older than 2 s as absent."""
    floor = NoiseFloor()
    version = 0
    while True:
        version, window = await pcm.newer_than(version)
        loudness = floor.above_floor(rms_dbfs(window.samples))
        cry = await asyncio.to_thread(classifier.cry_probability, window.samples)
        publish(AudioSample(at=window.at, loudness_db=loudness, cry_probability=cry))


def probe_audio(
    samples: npt.NDArray[np.float32], classifier: AudioClassifier
) -> list[tuple[float, float]]:
    """Per whole second of a 16 kHz mono clip: (loudness above floor in dB, P(cry))."""
    floor = NoiseFloor()
    return [
        (
            floor.above_floor(rms_dbfs(second)),
            classifier.cry_probability(second),
        )
        for second in (samples[i : i + 16_000] for i in range(0, len(samples) - 15_999, 16_000))
    ]


async def probe(settings: Settings, path: Path) -> None:
    """`babymon probe audio FILE.wav`: one line per whole second of the clip."""
    if not settings.yamnet_path.is_file():
        raise SystemExit(f"no YAMNet model at {settings.yamnet_path} (run scripts/fetch-yamnet.sh)")
    try:
        samples = load_clip(path)
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"ffmpeg could not decode {path}: {e.stderr.decode().strip()}") from None
    rows = probe_audio(samples, YamnetClassifier(settings.yamnet_path))
    print("second  loudness_db  p_cry")
    for second, (loudness, cry) in enumerate(rows):
        print(f"{second:6d}  {loudness:11.1f}  {cry:5.2f}")


def load_clip(path: Path) -> npt.NDArray[np.float32]:
    """Decode any audio file ffmpeg reads to 16 kHz mono float32, the classifier's input."""
    pcm = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000",
         "-f", "s16le", "-"],
        capture_output=True,
        check=True,
    ).stdout  # fmt: skip
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
