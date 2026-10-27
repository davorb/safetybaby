"""Shared script driver: feed timestamped facts through the core exactly as
app.run_brain does (observe every fact; on a Tick also situation -> decide -> gate)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import pytest

from babymon import brain
from babymon.brain import MonitorState
from babymon.domain import (
    AlertKind,
    AudioSample,
    BodyPosition,
    CameraStatus,
    Capabilities,
    Controls,
    Effect,
    Eyes,
    Fact,
    HeadDirection,
    Instant,
    MotionSample,
    Notify,
    Observation,
    Priority,
    Record,
    RequestVision,
    Restored,
    SleepState,
    SnapshotRef,
    Tick,
    Tuning,
    VisionDone,
    VisionReason,
)
from babymon.rules import RulesDecider

T0 = datetime(2026, 10, 27, 2, 0, tzinfo=UTC)
SEC = timedelta(seconds=1)
TUNING = Tuning()


def at(seconds: float) -> Instant:
    return T0 + seconds * SEC


def observation(**overrides: object) -> Observation:
    base = dict(
        body=BodyPosition.BACK,
        head=HeadDirection.CENTER,
        face_covered=False,
        in_crib=True,
        eyes=Eyes.CLOSED,
        confidence=0.9,
    )
    return Observation.model_validate({**base, **overrides})


def seen(t: float, obs: Observation, reason: VisionReason = VisionReason.PERIODIC) -> VisionDone:
    return VisionDone(at(t), reason, obs, SnapshotRef(f"{t}.jpg"))


def start(
    *,
    controls: Controls | None = None,
    sleep: tuple[SleepState, Instant] | None = None,
    delivered: dict[AlertKind, Instant] | None = None,
    vision: bool = True,
    push: bool = True,
    camera_up: bool = True,
) -> MonitorState:
    state = brain.initial_state(
        Restored(controls or Controls(), sleep, delivered or {}),
        Capabilities(vision=vision, push=push),
        T0,
    )
    if camera_up:
        state, _ = brain.observe(state, CameraStatus(T0, up=True, detail="test"), TUNING)
    return state


@dataclass
class Run:
    state: MonitorState
    tuning: Tuning = TUNING
    effects: list[tuple[Instant, Effect]] = field(default_factory=list)
    notifies: list[Notify] = field(default_factory=list)

    def feed(self, *facts: Fact) -> Run:
        for fact in facts:
            self.state, effects = brain.observe(self.state, fact, self.tuning)
            self.effects.extend((fact.at, e) for e in effects)
            if isinstance(fact, Tick):
                verdicts = RulesDecider(self.tuning).evaluate(self.situation())
                self.state, notifies = brain.gate(self.state, verdicts, self.tuning)
                self.effects.extend((fact.at, n) for n in notifies)
                self.notifies.extend(notifies)
        return self

    def seconds(
        self,
        start_s: int,
        end_s: int,
        *,
        motion: float | None = None,
        cry: float | None = None,
    ) -> Run:
        """One Tick per second over [start_s, end_s], each preceded by the given evidence."""
        for s in range(start_s, end_s + 1):
            facts: list[Fact] = []
            if motion is not None:
                facts.append(MotionSample(at(s), motion))
            if cry is not None:
                facts.append(AudioSample(at(s), loudness_db=0.0, cry_probability=cry))
            facts.append(Tick(at(s)))
            self.feed(*facts)
        return self

    def situation(self):
        return brain.situation(self.state, self.tuning)

    def alerts(self) -> list[tuple[Instant, AlertKind, Priority]]:
        return [(n.alert.at, n.alert.kind, n.alert.priority) for n in self.notifies]

    def records(self) -> list[object]:
        return [e.event for _, e in self.effects if isinstance(e, Record)]

    def requests(self) -> list[tuple[Instant, VisionReason]]:
        return [(e.at, e.reason) for _, e in self.effects if isinstance(e, RequestVision)]


@pytest.fixture
def run() -> Run:
    return Run(start())


def times(items: Iterable[tuple[Instant, *tuple[object, ...]]]) -> list[float]:
    return [(t - T0).total_seconds() for t, *_ in items]
