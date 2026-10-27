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
from dataclasses import dataclass, replace
from datetime import timedelta
from typing import assert_never

from babymon.domain import (
    Alert,
    AlertKind,
    AudioSample,
    BodyPosition,
    CameraChanged,
    CameraStatus,
    Capabilities,
    Controls,
    ControlsChanged,
    DeliveryResult,
    Effect,
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
    SetControls,
    Situation,
    SleepChanged,
    Tick,
    Tuning,
    Verdict,
    VisionDone,
    VisionFailed,
    VisionObserved,
    VisionReason,
)
from babymon.sleep import Evidence, Signal, SleepTracker, advance, classify

AUDIO_STALE = timedelta(seconds=2)
"""An AudioSample older than this no longer counts as evidence of crying."""


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


def _seconds(since: Instant, until: Instant) -> float:
    return (until - since).total_seconds()


@dataclass(frozen=True, slots=True)
class VisionTrack:
    """Vision scheduling plus the derived streaks the rules read.

    Streak rules: a confident in-crib observation increments or resets each streak.
    A low-confidence observation, or a failed check, leaves streaks unchanged (it is
    neither evidence of danger nor of safety). in_crib=False resets everything.
    """

    in_flight_since: Instant | None = None
    last_attempt_at: Instant | None = None
    last: Observation | None = None
    last_at: Instant | None = None
    face_covered_streak: int = 0
    stomach_streak: int = 0
    head_left_since: Instant | None = None
    motion_since_attempt: bool = False
    """A MotionSample at or above motion_moving arrived since the last attempt."""

    def pending_danger(self, at: Instant, tuning: Tuning) -> bool:
        """Any streak started but not yet alerting: schedule CONFIRM checks sooner."""
        streak_pending = any(
            0 < streak < tuning.unsafe_streak
            for streak in (self.face_covered_streak, self.stomach_streak)
        )
        head_pending = (
            self.head_left_since is not None
            and _seconds(self.head_left_since, at) <= tuning.head_left_alert_s
        )
        return streak_pending or head_pending

    def observed(self, fact: VisionDone, tuning: Tuning) -> VisionTrack:
        obs = fact.observation
        track = replace(self, in_flight_since=None, last=obs, last_at=fact.at)
        if obs.confidence < tuning.min_confidence:
            return track
        if not obs.in_crib:
            return replace(track, face_covered_streak=0, stomach_streak=0, head_left_since=None)
        head_left_since = None
        if obs.head is HeadDirection.LEFT:
            head_left_since = self.head_left_since or fact.at
        return replace(
            track,
            face_covered_streak=self.face_covered_streak + 1 if obs.face_covered else 0,
            stomach_streak=self.stomach_streak + 1 if obs.body is BodyPosition.STOMACH else 0,
            head_left_since=head_left_since,
        )


@dataclass(frozen=True, slots=True)
class MonitorState:
    at: Instant
    """Instant of the last Tick (or of startup). Facts between Ticks do not move it."""
    controls: Controls
    capabilities: Capabilities
    sleep: SleepTracker
    motion_peak: float
    """Max MotionSample.level since the last Tick; reset on Tick."""
    audio: AudioSample | None
    """The latest AudioSample; ignored once older than AUDIO_STALE."""
    last_signal: Signal
    vision: VisionTrack
    camera_up: bool
    camera_since: Instant
    ledger: Mapping[AlertKind, Instant]
    """Last time each kind was notified. The only cooldown state."""


def initial_state(restored: Restored, capabilities: Capabilities, at: Instant) -> MonitorState:
    """camera_up starts False with camera_since=at, so a camera that never connects
    after a restart still raises CAMERA_OFFLINE after the threshold.

    `at` is the startup instant. The first Tick must be later than it."""
    return MonitorState(
        at=at,
        controls=restored.controls,
        capabilities=capabilities,
        sleep=SleepTracker.start(at, restored.sleep),
        motion_peak=0.0,
        audio=None,
        last_signal=Signal.QUIET,
        vision=VisionTrack(),
        camera_up=False,
        camera_since=at,
        ledger=dict(restored.delivered),
    )


def observe(
    state: MonitorState, fact: Fact, tuning: Tuning
) -> tuple[MonitorState, tuple[Effect, ...]]:
    """Fold one fact. Dispatches on the Fact union with match + assert_never.

    MotionSample   -> raise motion_peak; at or above motion_moving also mark motion
                      since the last vision attempt.
    AudioSample    -> store it.
    VisionDone     -> clear in-flight, update last/streaks, emit Record(VisionObserved).
    VisionFailed   -> clear in-flight. Streaks untouched.
    CameraStatus   -> on change only: flip camera_up/since, emit Record(CameraChanged).
    SetControls    -> apply patch; if the result differs emit Record(ControlsChanged).
                      Applying the same patch twice emits nothing the second time.
    DeliveryResult -> delivered=False drops the kind's ledger entry so the next Tick
                      re-fires. With push disabled nothing can ever deliver, so the
                      cooldown stands and the log gets one undelivered row per cooldown
                      instead of one per second.
    Tick           -> ignored unless later than state.at (a duplicate changes nothing).
                      1. evidence -> sleep.classify -> sleep.advance;
                         emit Record(SleepChanged) if the state changed.
                      2. vision scheduling (below); emit RequestVision if due.
                      3. reset motion_peak, remember last_signal.

    Vision scheduling, only when capabilities.vision and nothing in flight
    (an in-flight request older than vision_timeout_s counts as lost):
        due_periodic = since_last >= (vision_confirm_s if pending_danger else
                                      vision_period_s if motion since last attempt else
                                      vision_period_quiet_s)
        due_settled  = last_signal in (MOVING, ACTIVE) and signal is QUIET
                       and since_last >= vision_min_spacing_s
    Checking when motion *ends* sees the new resting position instead of motion blur.
    """
    match fact:
        case MotionSample():
            vision = state.vision
            if fact.level >= tuning.motion_moving:
                vision = replace(vision, motion_since_attempt=True)
            return replace(state, motion_peak=max(state.motion_peak, fact.level), vision=vision), ()
        case AudioSample():
            return replace(state, audio=fact), ()
        case VisionDone():
            event = VisionObserved(
                at=fact.at,
                reason=fact.reason,
                observation=fact.observation,
                snapshot=fact.snapshot,
            )
            return replace(state, vision=state.vision.observed(fact, tuning)), (Record(event),)
        case VisionFailed():
            return replace(state, vision=replace(state.vision, in_flight_since=None)), ()
        case CameraStatus():
            if fact.up == state.camera_up:
                return state, ()
            event = CameraChanged(at=fact.at, up=fact.up, detail=fact.detail)
            return replace(state, camera_up=fact.up, camera_since=fact.at), (Record(event),)
        case SetControls():
            patch = {
                k: v
                for k, v in (("armed", fact.armed), ("nap_mode", fact.nap_mode))
                if v is not None
            }
            controls = state.controls.model_copy(update=patch)
            if controls == state.controls:
                return state, ()
            event = ControlsChanged(at=fact.at, controls=controls)
            return replace(state, controls=controls), (Record(event),)
        case DeliveryResult():
            if fact.delivered or not state.capabilities.push or fact.kind not in state.ledger:
                return state, ()
            ledger = {k: v for k, v in state.ledger.items() if k != fact.kind}
            return replace(state, ledger=ledger), ()
        case Tick():
            if fact.at <= state.at:
                return state, ()
            return _tick(state, fact.at, tuning)
        case _:
            assert_never(fact)


def _tick(
    state: MonitorState, at: Instant, tuning: Tuning
) -> tuple[MonitorState, tuple[Effect, ...]]:
    effects: list[Effect] = []
    signal = classify(_evidence(state, at, tuning), tuning)
    sleep = advance(state.sleep, signal, at)
    if sleep.state is not state.sleep.state:
        effects.append(Record(SleepChanged(at=at, previous=state.sleep.state, current=sleep.state)))
    vision, request = _schedule_vision(state, signal, at, tuning)
    if request is not None:
        effects.append(request)
    next_state = replace(
        state, at=at, sleep=sleep, vision=vision, motion_peak=0.0, last_signal=signal
    )
    return next_state, tuple(effects)


def _evidence(state: MonitorState, at: Instant, tuning: Tuning) -> Evidence:
    audio = state.audio
    cry = audio.cry_probability if audio is not None and at - audio.at <= AUDIO_STALE else 0.0
    seen = state.vision.last
    fresh = (
        seen is not None
        and state.vision.last_at is not None
        and _seconds(state.vision.last_at, at) <= tuning.observation_fresh_s
    )
    return Evidence(state.motion_peak, cry, seen if fresh else None)


def _schedule_vision(
    state: MonitorState, signal: Signal, at: Instant, tuning: Tuning
) -> tuple[VisionTrack, RequestVision | None]:
    track = state.vision
    if not state.capabilities.vision:
        return track, None
    if track.in_flight_since is not None:
        if _seconds(track.in_flight_since, at) <= tuning.vision_timeout_s:
            return track, None
        track = replace(track, in_flight_since=None)
    since_last = (
        float("inf") if track.last_attempt_at is None else _seconds(track.last_attempt_at, at)
    )
    confirming = track.pending_danger(at, tuning)
    if confirming:
        period = tuning.vision_confirm_s
    elif track.motion_since_attempt:
        period = tuning.vision_period_s
    else:
        period = tuning.vision_period_quiet_s
    if since_last >= period:
        reason = VisionReason.CONFIRM if confirming else VisionReason.PERIODIC
    elif (
        state.last_signal in (Signal.MOVING, Signal.ACTIVE)
        and signal is Signal.QUIET
        and since_last >= tuning.vision_min_spacing_s
    ):
        reason = VisionReason.SETTLED_AFTER_MOTION
    else:
        return track, None
    track = replace(track, in_flight_since=at, last_attempt_at=at, motion_since_attempt=False)
    return track, RequestVision(at=at, reason=reason)


def situation(state: MonitorState, tuning: Tuning) -> Situation:
    """Project state into the decider/dashboard contract. Pure, cheap, called each Tick."""
    at = state.at
    vision = state.vision
    return Situation(
        at=at,
        controls=state.controls,
        capabilities=state.capabilities,
        camera_up=state.camera_up,
        camera_down_for_s=0.0 if state.camera_up else _seconds(state.camera_since, at),
        sleep=state.sleep.state,
        sleep_for_s=_seconds(state.sleep.since, at),
        observation=vision.last,
        observation_age_s=None if vision.last_at is None else _seconds(vision.last_at, at),
        face_covered_streak=vision.face_covered_streak,
        stomach_streak=vision.stomach_streak,
        head_left_for_s=(
            0.0 if vision.head_left_since is None else _seconds(vision.head_left_since, at)
        ),
    )


def gate(
    state: MonitorState, verdicts: Sequence[Verdict], tuning: Tuning
) -> tuple[MonitorState, tuple[Notify, ...]]:
    """Turn verdicts into notifications under policy.

    - Not armed: no notifications, ledger untouched.
    - A verdict fires when its kind has no ledger entry or the entry is at least
      tuning.alert_cooldown_s old. Firing writes ledger[kind] = state.at.
    - A condition that persists past the cooldown fires again: that is the re-alert.
    - Priority and title come from ALERT_POLICY; message from verdict.reason.
    """
    if not state.controls.armed:
        return state, ()
    at = state.at
    ledger = dict(state.ledger)
    notifies: list[Notify] = []
    for verdict in verdicts:
        last = ledger.get(verdict.kind)
        if last is not None and _seconds(last, at) < tuning.alert_cooldown_s:
            continue
        policy = ALERT_POLICY[verdict.kind]
        ledger[verdict.kind] = at
        alert = Alert(
            kind=verdict.kind,
            priority=policy.priority,
            at=at,
            title=policy.title,
            message=verdict.reason,
        )
        notifies.append(Notify(alert))
    if not notifies:
        return state, ()
    return replace(state, ledger=ledger), tuple(notifies)
