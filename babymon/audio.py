"""Audio actor: PCM windows -> AudioSample facts.

Owns the noise-floor estimate and the classifier instance. The classifier is the swap
point; nothing outside this module knows which model it is.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Never, Protocol

import numpy as np
import numpy.typing as npt

from babymon.camera import PcmWindow
from babymon.domain import Fact
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

    def __init__(self, model_path: Path) -> None:
        raise NotImplementedError

    def cry_probability(self, samples: npt.NDArray[np.float32]) -> float:
        raise NotImplementedError


class NoiseFloor:
    """Rolling 10th percentile of per-second RMS dBFS over the last 10 minutes.
    The white-noise machine sets this floor; loudness is reported above it."""

    def __init__(self, window_s: int = 600) -> None:
        raise NotImplementedError

    def above_floor(self, rms_db: float) -> float:
        raise NotImplementedError


def rms_dbfs(samples: npt.NDArray[np.float32]) -> float:
    raise NotImplementedError


async def run_audio(
    pcm: Latest[PcmWindow], classifier: AudioClassifier, publish: Callable[[Fact], None]
) -> Never:
    """for each newer PcmWindow: rms, floor, await to_thread(classifier...),
    publish(AudioSample(at=window.at, ...)). A window missed while classifying is
    dropped; the brain treats audio older than 2 s as absent."""
    raise NotImplementedError
