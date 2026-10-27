"""The vocabulary every module shares. Pure data: no I/O, no clock, no SDKs.

Three families of types cross actor boundaries, and all of them live here:

- Fact: something an actor observed. Actors only ever *append* facts to the brain's
  inbox. Facts are immutable, so the inbox is the only place they meet.
- Effect: something the brain wants done. The shell executes effects; the brain never
  performs I/O.
- Event: what gets persisted. Events are the single source of truth for the dashboard
  timeline, the event list, and restart recovery (sleep state, alert cooldowns).

Situation is the JSON contract between the brain and any Decider (rules today, Jev
later) and also what the dashboard renders. One shape, two readers.

Timestamps are tz-aware UTC datetimes. They are created at the boundary (the actor that
read the bytes) and trusted everywhere inside.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, NewType

from pydantic import BaseModel, ConfigDict, Field

type Instant = datetime
"""tz-aware UTC. Naive datetimes never enter the core."""

SnapshotRef = NewType("SnapshotRef", str)
"""File name of a JPEG inside the store's snapshot directory. Never a full path."""


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# ---------------------------------------------------------------- vision vocabulary


class BodyPosition(StrEnum):
    BACK = "back"
    SIDE_LEFT = "side_left"
    SIDE_RIGHT = "side_right"
    STOMACH = "stomach"
    UNKNOWN = "unknown"


class HeadDirection(StrEnum):
    """Direction the baby's face points, from the baby's own perspective."""

    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    UNKNOWN = "unknown"


class Eyes(StrEnum):
    OPEN = "open"
    CLOSED = "closed"
    UNKNOWN = "unknown"


class Observation(Frozen):
    """One typed reading of one frame. Produced only by a VisionBackend."""

    body: BodyPosition
    head: HeadDirection
    face_covered: bool
    in_crib: bool
    eyes: Eyes
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]


class VisionReason(StrEnum):
    PERIODIC = "periodic"
    SETTLED_AFTER_MOTION = "settled_after_motion"
    CONFIRM = "confirm"
    """A follow-up check scheduled early because a dangerous reading is pending."""


# ---------------------------------------------------------------- sleep, controls, alerts


class SleepState(StrEnum):
    ASLEEP = "asleep"
    STIRRING = "stirring"
    AWAKE = "awake"
    CRYING = "crying"


class Controls(Frozen):
    """Operator switches. Persisted; survive restart.

    armed=False silences every notification but keeps sensing and logging.
    nap_mode=True additionally enables the awake-during-nap alert.
    """

    armed: bool = True
    nap_mode: bool = False


class Capabilities(Frozen):
    """What the running process can do, derived once from Settings at startup."""

    vision: bool
    push: bool


class AlertKind(StrEnum):
    FACE_COVERED = "face_covered"
    ON_STOMACH = "on_stomach"
    HEAD_LEFT = "head_left"
    CRYING = "crying"
    AWAKE_DURING_NAP = "awake_during_nap"
    CAMERA_OFFLINE = "camera_offline"


class Priority(StrEnum):
    """How loud an alert is. notify.py maps this to the transport's own scale and adds
    retry/expire for EMERGENCY; the domain never sees transport parameters."""

    NORMAL = "normal"
    HIGH = "high"
    EMERGENCY = "emergency"


class Tuning(Frozen):
    """Every threshold the pure core reads. Built from Settings at the boundary."""

    # alert rules
    unsafe_streak: int = 2
    head_left_alert_s: float = 120.0
    crying_alert_s: float = 60.0
    camera_offline_alert_s: float = 60.0
    min_confidence: float = 0.5
    # vision scheduling
    vision_period_s: float = 60.0
    vision_period_quiet_s: float = 900.0
    """Periodic interval when no motion was seen since the last attempt: a floor so an
    infrared switch or a slow drift is still re-checked."""
    vision_confirm_s: float = 15.0
    vision_min_spacing_s: float = 10.0
    vision_timeout_s: float = 45.0
    observation_fresh_s: float = 90.0
    # sleep evidence
    motion_moving: float = 0.02
    motion_active: float = 0.08
    cry_probability: float = 0.6
    # cooldown
    alert_cooldown_s: float = 600.0


class Situation(Frozen):
    """Everything a Decider may look at, as plain JSON-able values.

    Durations are precomputed by the brain so a Decider needs no memory and no clock:
    decide(situation) is a pure function of this one value. That is what lets an
    external text/JSON service (Jev) replace the rules without seeing history.
    """

    at: Instant
    controls: Controls
    capabilities: Capabilities
    camera_up: bool
    camera_down_for_s: float
    sleep: SleepState
    sleep_for_s: float
    observation: Observation | None
    observation_age_s: float | None
    face_covered_streak: int
    stomach_streak: int
    head_left_for_s: float
    """0.0 unless every confident check since the first left reading said LEFT."""


class Verdict(Frozen):
    """A Decider's claim that an alert condition holds right now. No priority, no
    cooldown: those are policy owned by brain.ALERT_POLICY, not by the decider."""

    kind: AlertKind
    reason: str


class Alert(Frozen):
    kind: AlertKind
    priority: Priority
    at: Instant
    title: str
    message: str


@dataclass(frozen=True, slots=True)
class Restored:
    """What the store hands back on startup. Everything else starts fresh."""

    controls: Controls
    sleep: tuple[SleepState, Instant] | None
    delivered: Mapping[AlertKind, Instant]
    """Last *delivered* alert per kind, within the cooldown horizon."""


# ---------------------------------------------------------------- media payloads


@dataclass(frozen=True, slots=True)
class Still:
    """One JPEG frame (1280x720) as ffmpeg encoded it. Shared by reference, never mutated."""

    jpeg: bytes
    at: Instant


# ---------------------------------------------------------------- facts (actors -> brain)


@dataclass(frozen=True, slots=True)
class MotionSample:
    at: Instant
    level: float
    """Fraction of ROI pixels that changed beyond the noise threshold, 0..1."""


@dataclass(frozen=True, slots=True)
class AudioSample:
    at: Instant
    loudness_db: float
    """RMS above the rolling noise floor (the white-noise machine), in dB."""
    cry_probability: float


@dataclass(frozen=True, slots=True)
class VisionDone:
    at: Instant
    reason: VisionReason
    observation: Observation
    snapshot: SnapshotRef


@dataclass(frozen=True, slots=True)
class VisionFailed:
    at: Instant
    reason: VisionReason
    error: str


@dataclass(frozen=True, slots=True)
class CameraStatus:
    at: Instant
    up: bool
    detail: str


@dataclass(frozen=True, slots=True)
class SetControls:
    """A patch, not a replacement: two phones toggling different switches never clobber."""

    at: Instant
    armed: bool | None = None
    nap_mode: bool | None = None


@dataclass(frozen=True, slots=True)
class DeliveryResult:
    """The alerter's report on one Notify. delivered=False clears the kind's cooldown so
    the next Tick re-fires instead of waiting it out; the store's restore does the same
    for a crash between deciding and delivering."""

    at: Instant
    kind: AlertKind
    delivered: bool


@dataclass(frozen=True, slots=True)
class Tick:
    """1 Hz heartbeat. The only fact that advances time-based rules, so a silent camera
    still produces CAMERA_OFFLINE instead of freezing the brain."""

    at: Instant


type Fact = (
    MotionSample
    | AudioSample
    | VisionDone
    | VisionFailed
    | CameraStatus
    | SetControls
    | DeliveryResult
    | Tick
)


# ---------------------------------------------------------------- events (persisted)


class SleepChanged(Frozen):
    type: Literal["sleep_changed"] = "sleep_changed"
    at: Instant
    previous: SleepState
    current: SleepState


class VisionObserved(Frozen):
    type: Literal["vision"] = "vision"
    at: Instant
    reason: VisionReason
    observation: Observation
    snapshot: SnapshotRef


class AlertSent(Frozen):
    """Written by the alerter after the delivery attempt. Only delivered=True rows
    count toward cooldown on restart, so a crash before delivery re-alerts."""

    type: Literal["alert"] = "alert"
    at: Instant
    alert: Alert
    snapshot: SnapshotRef | None
    delivered: bool
    detail: str = ""


class ControlsChanged(Frozen):
    type: Literal["controls"] = "controls"
    at: Instant
    controls: Controls


class CameraChanged(Frozen):
    type: Literal["camera"] = "camera"
    at: Instant
    up: bool
    detail: str


type Event = Annotated[
    SleepChanged | VisionObserved | AlertSent | ControlsChanged | CameraChanged,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------- effects (brain -> shell)


@dataclass(frozen=True, slots=True)
class RequestVision:
    at: Instant
    reason: VisionReason


@dataclass(frozen=True, slots=True)
class Notify:
    alert: Alert


@dataclass(frozen=True, slots=True)
class Record:
    event: SleepChanged | ControlsChanged | CameraChanged | VisionObserved


type Effect = RequestVision | Notify | Record
