"""Sleep state machine. Pure: (tracker, signal, now) -> tracker.

Two steps, each a table instead of a branch chain:

1. classify(): one second of evidence -> one Signal, by fixed precedence.
2. advance(): (state, signal) looks up an Edge in TRANSITIONS. The edge names a target
   state and how long the evidence must keep pointing at that target before the
   transition happens (dwell). Dwell is tracked per *target*, not per signal, so
   alternating MOVING/ACTIVE seconds both count toward "awake".

Initial state when nothing is known is STIRRING: no alert rule fires on it, and it
resolves to ASLEEP after quiet or AWAKE after activity. That is why a restart with no
recent SleepChanged event cannot produce a false awake-during-nap alert.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import timedelta
from enum import StrEnum

from babymon.domain import Eyes, Instant, Observation, SleepState, Tuning


class Signal(StrEnum):
    QUIET = "quiet"
    MOVING = "moving"
    EYES_OPEN = "eyes_open"
    ACTIVE = "active"
    CRYING = "crying"


@dataclass(frozen=True, slots=True)
class Evidence:
    """One second of fused input, assembled by the brain on each Tick."""

    motion_peak: float
    cry_probability: float
    fresh_observation: Observation | None
    """None when the last observation is older than Tuning.observation_fresh_s."""


@dataclass(frozen=True, slots=True)
class Edge:
    to: SleepState
    dwell_s: float


CRY_CONFIRM_S = 3.0
"""Seconds of continuous cry evidence before the tracker enters CRYING."""

S, G = SleepState, Signal

TRANSITIONS: Mapping[SleepState, Mapping[Signal, Edge]] = {
    S.ASLEEP: {
        G.MOVING: Edge(S.STIRRING, 3),
        G.ACTIVE: Edge(S.STIRRING, 1),
        G.EYES_OPEN: Edge(S.AWAKE, 30),
        G.CRYING: Edge(S.CRYING, CRY_CONFIRM_S),
    },
    S.STIRRING: {
        G.QUIET: Edge(S.ASLEEP, 120),
        G.MOVING: Edge(S.AWAKE, 60),
        G.ACTIVE: Edge(S.AWAKE, 20),
        G.EYES_OPEN: Edge(S.AWAKE, 20),
        G.CRYING: Edge(S.CRYING, CRY_CONFIRM_S),
    },
    S.AWAKE: {
        G.QUIET: Edge(S.ASLEEP, 300),
        G.CRYING: Edge(S.CRYING, CRY_CONFIRM_S),
    },
    S.CRYING: {
        G.QUIET: Edge(S.AWAKE, 20),
        G.MOVING: Edge(S.AWAKE, 20),
        G.EYES_OPEN: Edge(S.AWAKE, 20),
        G.ACTIVE: Edge(S.AWAKE, 20),
    },
}
"""A missing (state, signal) pair means 'stay, and drop any pending transition'."""

RESUME_WITHIN = timedelta(minutes=10)


@dataclass(frozen=True, slots=True)
class Pending:
    to: SleepState
    since: Instant


@dataclass(frozen=True, slots=True)
class SleepTracker:
    state: SleepState
    since: Instant
    pending: Pending | None = None

    @staticmethod
    def start(at: Instant, restored: tuple[SleepState, Instant] | None) -> SleepTracker:
        """Resume the persisted state if recent, else start in STIRRING (see module doc)."""
        if restored is not None and at - restored[1] < RESUME_WITHIN:
            return SleepTracker(restored[0], restored[1])
        return SleepTracker(SleepState.STIRRING, at)


def classify(evidence: Evidence, tuning: Tuning) -> Signal:
    """Precedence: CRYING > ACTIVE > EYES_OPEN > MOVING > QUIET.

    Loudness is deliberately ignored: the white-noise machine makes RMS a poor signal,
    and the classifier already sees the waveform.
    """
    if evidence.cry_probability >= tuning.cry_probability:
        return Signal.CRYING
    if evidence.motion_peak >= tuning.motion_active:
        return Signal.ACTIVE
    seen = evidence.fresh_observation
    if seen is not None and seen.confidence >= tuning.min_confidence and seen.eyes is Eyes.OPEN:
        return Signal.EYES_OPEN
    if evidence.motion_peak >= tuning.motion_moving:
        return Signal.MOVING
    return Signal.QUIET


def advance(tracker: SleepTracker, signal: Signal, at: Instant) -> SleepTracker:
    """Idempotent per instant: advancing twice with the same (signal, at) yields the same
    tracker, so a duplicated Tick cannot double-step the machine.
    """
    edge = TRANSITIONS[tracker.state].get(signal)
    if edge is None:
        return replace(tracker, pending=None)
    pending = tracker.pending
    if pending is None or pending.to != edge.to:
        pending = Pending(edge.to, at)
    if (at - pending.since).total_seconds() >= edge.dwell_s:
        return SleepTracker(edge.to, since=at)
    return replace(tracker, pending=pending)
