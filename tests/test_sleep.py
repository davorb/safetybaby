"""Signal scripts through the sleep machine. Dwell is measured from the first tick
that carries the signal, so a dwell of N seconds completes on the tick N seconds later."""

from __future__ import annotations

from datetime import timedelta

import pytest

from babymon.domain import Eyes, SleepState, Tuning
from babymon.sleep import Evidence, Signal, SleepTracker, advance, classify
from tests.conftest import T0, TUNING, at, observation

S, G = SleepState, Signal


def drive(tracker: SleepTracker, signals: list[Signal], start_s: int = 1) -> list[SleepState]:
    states = []
    for i, signal in enumerate(signals):
        tracker = advance(tracker, signal, at(start_s + i))
        states.append(tracker.state)
    return states


@pytest.mark.parametrize(
    ("start", "signal", "target", "dwell"),
    [
        (S.ASLEEP, G.MOVING, S.STIRRING, 3),
        (S.STIRRING, G.QUIET, S.ASLEEP, 120),
        (S.ASLEEP, G.CRYING, S.CRYING, 3),
        (S.STIRRING, G.CRYING, S.CRYING, 3),
        (S.AWAKE, G.CRYING, S.CRYING, 3),
        (S.CRYING, G.QUIET, S.AWAKE, 20),
        (S.AWAKE, G.QUIET, S.ASLEEP, 300),
    ],
)
def test_transition_after_dwell(start, signal, target, dwell):
    states = drive(SleepTracker(start, T0), [signal] * (dwell + 1))
    assert states[:dwell] == [start] * dwell
    assert states[dwell] == target


def test_mixed_moving_and_active_both_count_toward_awake():
    alternating = [G.MOVING, G.ACTIVE] * 11
    mixed = drive(SleepTracker(S.STIRRING, T0), alternating)
    only_moving = drive(SleepTracker(S.STIRRING, T0), [G.MOVING] * 22)
    assert S.AWAKE in mixed
    assert S.AWAKE not in only_moving


def test_signal_without_edge_drops_the_pending_transition():
    script = [G.MOVING, G.MOVING, G.QUIET, G.MOVING, G.MOVING, G.MOVING, G.MOVING]
    states = drive(SleepTracker(S.ASLEEP, T0), script)
    assert states == [S.ASLEEP] * 6 + [S.STIRRING]


def test_advance_is_idempotent_per_instant():
    tracker = SleepTracker(S.ASLEEP, T0)
    once = advance(tracker, G.MOVING, at(1))
    twice = advance(once, G.MOVING, at(1))
    assert twice == once


def test_state_since_is_the_transition_instant():
    tracker = SleepTracker(S.ASLEEP, T0)
    for s in range(1, 5):
        tracker = advance(tracker, G.CRYING, at(s))
    assert (tracker.state, tracker.since) == (S.CRYING, at(4))


def test_start_resumes_a_recent_state():
    assert SleepTracker.start(T0, (S.AWAKE, T0 - timedelta(minutes=9))).state is S.AWAKE


@pytest.mark.parametrize("restored", [None, (S.AWAKE, T0 - timedelta(minutes=10))])
def test_start_falls_back_to_stirring(restored):
    assert SleepTracker.start(T0, restored) == SleepTracker(S.STIRRING, T0)


@pytest.mark.parametrize(
    ("evidence", "expected"),
    [
        (Evidence(0.0, 0.0, None), G.QUIET),
        (Evidence(0.02, 0.0, None), G.MOVING),
        (Evidence(0.08, 0.0, None), G.ACTIVE),
        (Evidence(0.0, 0.0, observation(eyes=Eyes.OPEN)), G.EYES_OPEN),
        (Evidence(0.0, 0.0, observation(eyes=Eyes.OPEN, confidence=0.3)), G.QUIET),
        (Evidence(0.05, 0.0, observation(eyes=Eyes.OPEN)), G.EYES_OPEN),
        (Evidence(0.5, 0.0, observation(eyes=Eyes.OPEN)), G.ACTIVE),
        (Evidence(0.5, 0.6, observation(eyes=Eyes.OPEN)), G.CRYING),
    ],
)
def test_classify_precedence(evidence, expected):
    assert classify(evidence, TUNING) is expected


def test_classify_reads_thresholds_from_tuning():
    strict = Tuning(cry_probability=0.95)
    assert classify(Evidence(0.0, 0.9, None), strict) is G.QUIET
