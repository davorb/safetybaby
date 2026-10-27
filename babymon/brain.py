"""The functional core. Folds facts into MonitorState and decides what to do.

    observe(state, fact)          -> (state, effects)    every fact
    situation(state)              -> Situation           derived, never stored
    gate(state, verdicts)         -> (state, notifies)   after the Decider, on Tick only

No I/O, no clock (time comes from fact.at), no asyncio. The async Decider call sits
between situation() and gate() in app.run_brain, so the core stays pure even when the
decider becomes a network service.

MonitorState is owned by exactly one task (app.run_brain). Every other actor reaches it
only by appending a Fact to the inbox, and reads it only through the published
Situation. There is no shared mutable state to lock.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from babymon.domain import (
    Alert,
    AlertKind,
    Capabilities,
    Controls,
    Effect,
    Fact,
    Instant,
    Notify,
    Observation,
    Priority,
    Restored,
    Situation,
    SleepState,
    Tuning,
    Verdict,
)
from babymon.sleep import Signal, SleepTracker


@dataclass(frozen=True, slots=True)
class AlertPolicy:
    priority: Priority
    title: str


ALERT_POLICY: Mapping[AlertKind, AlertPolicy] = {
    AlertKind.FACE_COVERED: AlertPolicy(Priority.EMERGENCY, "Face covered"),
    AlertKind.ON_STOMACH: AlertPolicy(Priority.EMERGENCY, "Baby on stomach"),
    AlertKind.HEAD_LEFT: AlertPolicy(Priority.HIGH, "Head turned left"),
    AlertKind.CRYING: AlertPolicy(Priority.HIGH, "Crying"),
    AlertKind.AWAKE_DURING_NAP: AlertPolicy(Priority.NORMAL, "Awake"),
    AlertKind.CAMERA_OFFLINE: AlertPolicy(Priority.HIGH, "Camera offline"),
}
if set(ALERT_POLICY) != set(AlertKind):
    raise RuntimeError("ALERT_POLICY must cover every AlertKind")


@dataclass(frozen=True, slots=True)
class VisionTrack:
    """Vision scheduling plus the derived streaks the rules read.

    Streak rules: a confident in-crib observation increments or resets each streak.
    A low-confidence observation, or a failed check, leaves streaks unchanged (it is
    neither evidence of danger nor of safety). in_crib=False resets everything.
    """

    in_flight_since: Instant | None
    last_attempt_at: Instant | None
    last: Observation | None
    last_at: Instant | None
    face_covered_streak: int
    stomach_streak: int
    head_left_since: Instant | None

    @property
    def pending_danger(self) -> bool:
        """Any streak started but not yet alerting: schedule CONFIRM checks sooner."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class MonitorState:
    at: Instant
    controls: Controls
    capabilities: Capabilities
    sleep: SleepTracker
    motion_peak: float
    """Max MotionSample.level since the last Tick; reset on Tick."""
    cry_probability: float
    """From the latest AudioSample; decays to 0 if no sample arrived for 2 s."""
    last_signal: Signal
    vision: VisionTrack
    camera_up: bool
    camera_since: Instant
    ledger: Mapping[AlertKind, Instant]
    """Last time each kind was notified. The only cooldown state."""


def initial_state(restored: Restored, capabilities: Capabilities, at: Instant) -> MonitorState:
    """camera_up starts False with camera_since=at, so a camera that never connects
    after a restart still raises CAMERA_OFFLINE after the threshold."""
    raise NotImplementedError


def observe(state: MonitorState, fact: Fact, tuning: Tuning) -> tuple[MonitorState, tuple[Effect, ...]]:
    """Fold one fact. Dispatches on the Fact union with match + assert_never.

    MotionSample  -> raise motion_peak.
    AudioSample   -> store cry_probability.
    VisionDone    -> clear in-flight, update last/streaks, emit Record(VisionObserved).
    VisionFailed  -> clear in-flight. Streaks untouched.
    CameraStatus  -> on change only: flip camera_up/since, emit Record(CameraChanged).
    SetControls   -> apply patch; if the result differs emit Record(ControlsChanged).
                     Applying the same patch twice emits nothing the second time.
    Tick          -> 1. evidence -> sleep.classify -> sleep.advance;
                        emit Record(SleepChanged) if the state changed.
                     2. vision scheduling (below); emit RequestVision if due.
                     3. reset motion_peak, remember last_signal.

    Vision scheduling, only when capabilities.vision and nothing in flight
    (an in-flight request older than vision_timeout_s counts as lost):
        due_periodic = since_last >= (vision_confirm_s if pending_danger else vision_period_s)
        due_settled  = last_signal in (MOVING, ACTIVE) and signal is QUIET
                       and since_last >= vision_min_spacing_s
    Checking when motion *ends* sees the new resting position instead of motion blur.
    """
    raise NotImplementedError


def situation(state: MonitorState, tuning: Tuning) -> Situation:
    """Project state into the decider/dashboard contract. Pure, cheap, called each Tick."""
    raise NotImplementedError


def gate(
    state: MonitorState, verdicts: Sequence[Verdict], tuning: Tuning
) -> tuple[MonitorState, tuple[Notify, ...]]:
    """Turn verdicts into notifications under policy.

    - Not armed: no notifications, ledger untouched.
    - A verdict fires when its kind has no ledger entry or the entry is older than
      tuning.alert_cooldown_s. Firing writes ledger[kind] = state.at.
    - A condition that persists past the cooldown fires again: that is the re-alert.
    - Priority and title come from ALERT_POLICY; message from verdict.reason.
    """
    raise NotImplementedError
