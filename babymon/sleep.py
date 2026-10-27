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
from dataclasses import dataclass
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


S, G = SleepState, Signal

TRANSITIONS: Mapping[SleepState, Mapping[Signal, Edge]] = {
    S.ASLEEP: {
        G.MOVING: Edge(S.STIRRING, 3),
        G.ACTIVE: Edge(S.STIRRING, 1),
        G.EYES_OPEN: Edge(S.AWAKE, 30),
        G.CRYING: Edge(S.CRYING, 3),
    },
    S.STIRRING: {
        G.QUIET: Edge(S.ASLEEP, 120),
        G.MOVING: Edge(S.AWAKE, 60),
        G.ACTIVE: Edge(S.AWAKE, 20),
        G.EYES_OPEN: Edge(S.AWAKE, 20),
        G.CRYING: Edge(S.CRYING, 3),
    },
    S.AWAKE: {
        G.QUIET: Edge(S.ASLEEP, 300),
        G.CRYING: Edge(S.CRYING, 3),
    },
    S.CRYING: {
        G.QUIET: Edge(S.AWAKE, 20),
        G.MOVING: Edge(S.AWAKE, 20),
        G.EYES_OPEN: Edge(S.AWAKE, 20),
        G.ACTIVE: Edge(S.AWAKE, 20),
    },
}
"""A missing (state, signal) pair means 'stay, and drop any pending transition'."""


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
        raise NotImplementedError


def classify(evidence: Evidence, tuning: Tuning) -> Signal:
    """Precedence: CRYING > ACTIVE > EYES_OPEN > MOVING > QUIET.

    CRYING: cry_probability >= tuning.cry_probability.
    ACTIVE / MOVING: motion_peak against tuning.motion_active / motion_moving.
    EYES_OPEN: fresh confident observation with eyes == Eyes.OPEN.
    Loudness is deliberately ignored: the white-noise machine makes RMS a poor signal,
    and the classifier already sees the waveform.
    """
    raise NotImplementedError


def advance(tracker: SleepTracker, signal: Signal, at: Instant) -> SleepTracker:
    """Pure transition.

    edge = TRANSITIONS[state].get(signal)
    if edge is None: return tracker with pending=None
    if pending is None or pending.to != edge.to: pending = Pending(edge.to, at)
    if at - pending.since >= edge.dwell_s: return SleepTracker(edge.to, since=at)
    return tracker with pending

    Idempotent per instant: advancing twice with the same (signal, at) yields the same
    tracker, so a duplicated Tick cannot double-step the machine.
    """
    raise NotImplementedError
