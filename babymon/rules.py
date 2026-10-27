"""The decision step: Situation in, Verdicts out. The swap point for Jev.

A Decider sees only the JSON-able Situation and answers which alert conditions hold
right now. Streaks and durations arrive precomputed, already filtered to confident,
in-crib readings by the brain. It knows nothing about priorities, cooldowns, arming, or delivery; those
are policy in brain.ALERT_POLICY and brain.gate(). That keeps the replaceable part as
small as the brief's "text/JSON in, typed answers out".
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from babymon.domain import AlertKind, Situation, SleepState, Tuning, Verdict


class Decider(Protocol):
    async def decide(self, situation: Situation) -> Sequence[Verdict]:
        """Must not raise for a well-formed Situation. A remote decider that fails
        returns the verdicts of its local fallback; see design.md open questions."""
        ...


@dataclass(frozen=True, slots=True)
class Rule:
    kind: AlertKind
    holds: Callable[[Situation, Tuning], bool]
    reason: Callable[[Situation], str]


RULES: tuple[Rule, ...] = (
    Rule(
        AlertKind.FACE_COVERED,
        lambda s, t: s.face_covered_streak >= t.unsafe_streak,
        lambda s: f"Face covered on {s.face_covered_streak} consecutive checks",
    ),
    Rule(
        AlertKind.ON_STOMACH,
        lambda s, t: s.stomach_streak >= t.unsafe_streak,
        lambda s: f"On stomach on {s.stomach_streak} consecutive checks",
    ),
    Rule(
        AlertKind.HEAD_LEFT,
        lambda s, t: s.head_left_for_s > t.head_left_alert_s,
        lambda s: f"Head turned left for {s.head_left_for_s / 60:.0f} min",
    ),
    Rule(
        AlertKind.CRYING,
        lambda s, t: s.sleep is SleepState.CRYING and s.sleep_for_s > t.crying_alert_s,
        lambda s: f"Crying for {s.sleep_for_s:.0f} s",
    ),
    Rule(
        AlertKind.AWAKE_DURING_NAP,
        lambda s, t: s.controls.nap_mode and s.sleep in (SleepState.AWAKE, SleepState.CRYING),
        lambda s: "Awake during nap",
    ),
    Rule(
        AlertKind.CAMERA_OFFLINE,
        lambda s, t: not s.camera_up and s.camera_down_for_s > t.camera_offline_alert_s,
        lambda s: f"No video for {s.camera_down_for_s:.0f} s",
    ),
)
"""One row per condition. Adding a rule is adding a row; brain.ALERT_POLICY must gain
the matching kind or the import-time check in brain.py fails."""

class RulesDecider:
    def __init__(self, tuning: Tuning) -> None:
        self._tuning = tuning

    def evaluate(self, situation: Situation) -> list[Verdict]:
        """Pure, synchronous. Tests call this directly."""
        return [
            Verdict(kind=r.kind, reason=r.reason(situation))
            for r in RULES
            if r.holds(situation, self._tuning)
        ]

    async def decide(self, situation: Situation) -> Sequence[Verdict]:
        return self.evaluate(situation)
