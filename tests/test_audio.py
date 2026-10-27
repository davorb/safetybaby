from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from babymon.audio import (
    NoiseFloor,
    YamnetClassifier,
    load_clip,
    probe_audio,
    rms_dbfs,
    run_audio,
)
from babymon.camera import PcmWindow
from babymon.domain import AudioSample
from babymon.latest import Latest

MODEL = Path(__file__).parent.parent / "models" / "yamnet.onnx"
T0 = datetime(2026, 10, 27, 2, 0, tzinfo=UTC)


def second(amplitude: float) -> np.ndarray:
    return np.full(16_000, amplitude, dtype=np.float32)


def test_rms_dbfs_of_known_levels():
    assert rms_dbfs(second(1.0)) == pytest.approx(0.0)
    assert rms_dbfs(second(0.1)) == pytest.approx(-20.0)
    assert rms_dbfs(second(0.0)) == -100.0


def test_noise_floor_reports_loudness_above_the_quiet_baseline():
    floor = NoiseFloor()
    for _ in range(50):
        floor.above_floor(-40.0)
    assert floor.above_floor(-40.0) == pytest.approx(0.0)
    assert floor.above_floor(-10.0) == pytest.approx(30.0)


def test_noise_floor_forgets_levels_older_than_its_window():
    floor = NoiseFloor(window_s=10)
    for _ in range(10):
        floor.above_floor(-60.0)
    for _ in range(10):
        floor.above_floor(-30.0)
    assert floor.above_floor(-30.0) == pytest.approx(0.0)


class ScriptedClassifier:
    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = list(probabilities)

    def cry_probability(self, samples: np.ndarray) -> float:
        return self.probabilities.pop(0)


async def test_run_audio_publishes_one_sample_per_window_with_its_timestamp():
    pcm: Latest[PcmWindow] = Latest()
    published: list[AudioSample] = []
    task = asyncio.create_task(run_audio(pcm, ScriptedClassifier([0.1, 0.9]), published.append))

    for i, amplitude in enumerate([0.01, 0.5]):
        pcm.set(PcmWindow(samples=second(amplitude), at=T0 + timedelta(seconds=i)))
        async with asyncio.timeout(5):
            while len(published) <= i:
                await asyncio.sleep(0.01)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    first, louder = published
    assert (first.at, first.cry_probability) == (T0, 0.1)
    assert (louder.at, louder.cry_probability) == (T0 + timedelta(seconds=1), 0.9)
    assert louder.loudness_db > first.loudness_db


def test_probe_audio_scores_each_whole_second():
    clip = np.concatenate([second(0.01), second(0.2), second(0.01)[:8000]])
    rows = probe_audio(clip, ScriptedClassifier([0.0, 0.7]))
    assert [round(p, 1) for _, p in rows] == [0.0, 0.7]
    assert rows[1][0] > rows[0][0]


@pytest.fixture(scope="module")
def classifier():
    return YamnetClassifier(MODEL)


@pytest.mark.slow
@pytest.mark.skipif(not MODEL.exists(), reason="run scripts/fetch-yamnet.sh")
class TestYamnet:
    def test_a_tone_is_not_a_cry(self, classifier):
        t = np.arange(16_000) / 16_000
        tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        assert classifier.cry_probability(tone) < 0.2

    def test_white_noise_is_not_a_cry(self, classifier):
        noise = (np.random.default_rng(1).standard_normal(16_000) * 0.1).astype(np.float32)
        assert classifier.cry_probability(noise) < 0.2

    def test_silence_is_not_a_cry(self, classifier):
        assert classifier.cry_probability(second(0.0)) < 0.2

    def test_load_clip_round_trips_through_ffmpeg(self, tmp_path):
        import wave

        path = tmp_path / "tone.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16_000)
            w.writeframes((np.full(16_000, 3277, dtype="<i2")).tobytes())
        clip = load_clip(path)
        assert clip.shape == (16_000,) and clip.dtype == np.float32
        assert clip.mean() == pytest.approx(0.1, abs=0.001)
