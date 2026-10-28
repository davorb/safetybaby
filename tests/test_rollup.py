from __future__ import annotations

import pytest

from babymon import rollup
from babymon.domain import (
    AudioSample,
    CameraStatus,
    Fact,
    MinuteRollup,
    MotionSample,
    Tick,
    Tuning,
)
from tests.conftest import T0, at

TUNING = Tuning(cry_probability=0.6)


def fold(facts: list[Fact]) -> list[MinuteRollup]:
    acc, out = None, []
    for fact in facts:
        acc, finished = rollup.step(acc, fact, TUNING)
        if finished is not None:
            out.append(finished)
    return out


def test_minute_is_emitted_when_a_later_minute_starts():
    facts = [MotionSample(at(10), 0.1), MotionSample(at(20), 0.3), Tick(at(59))]
    assert fold(facts) == []
    [m] = fold([*facts, Tick(at(60))])
    assert m.at == T0
    assert (m.motion_samples, m.motion_mean, m.motion_peak) == (2, pytest.approx(0.2), 0.3)


def test_audio_summary_and_cry_seconds_count_samples_at_the_threshold():
    probabilities = [0.1, 0.6, 0.9, 0.59]
    facts: list[Fact] = [
        AudioSample(at(i), loudness_db=i * 2.0, cry_probability=p)
        for i, p in enumerate(probabilities)
    ]
    [m] = fold([*facts, Tick(at(61))])
    assert m.audio_samples == 4
    assert m.cry_seconds == 2
    assert m.cry_peak == 0.9
    assert m.loudness_peak_db == 6.0
    assert m.loudness_mean_db == pytest.approx(3.0)


def test_negative_loudness_peak_is_not_clamped_to_zero():
    [m] = fold([AudioSample(at(1), -5.0, 0.0), AudioSample(at(2), -3.0, 0.0), Tick(at(60))])
    assert m.loudness_peak_db == -3.0


def test_motion_only_minute_has_no_audio_summary_and_vice_versa():
    [motion] = fold([MotionSample(at(1), 0.05), Tick(at(60))])
    assert (motion.audio_samples, motion.loudness_mean_db, motion.cry_peak) == (0, None, None)
    [audio] = fold([AudioSample(at(1), 1.0, 0.0), Tick(at(60))])
    assert (audio.motion_samples, audio.motion_mean, audio.motion_peak) == (0, None, None)


def test_minutes_without_samples_emit_nothing():
    assert fold([Tick(at(s)) for s in range(0, 300)]) == []
    gap = fold([MotionSample(at(1), 0.0), Tick(at(60)), Tick(at(120)), Tick(at(180))])
    assert [m.at for m in gap] == [T0]


def test_a_gap_then_a_sample_emits_the_old_minute_and_starts_the_new_one():
    out = fold([MotionSample(at(5), 0.1), MotionSample(at(305), 0.2), Tick(at(400))])
    assert [(m.at, m.motion_peak) for m in out] == [(T0, 0.1), (at(300), 0.2)]


def test_other_facts_and_late_facts_leave_the_open_minute_alone():
    facts = [
        MotionSample(at(70), 0.1),
        CameraStatus(at(80), up=True, detail=""),
        MotionSample(at(30), 0.5),
        Tick(at(130)),
    ]
    [m] = fold(facts)
    assert (m.at, m.motion_samples, m.motion_peak) == (at(60), 2, 0.5)
