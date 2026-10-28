"""What every block reads: the window, the calendar over it and the observed sleep spans."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import tzinfo

from babymon.domain import (
    AlertSent,
    CameraChanged,
    Instant,
    MinuteRollup,
    SleepSpan,
    Tuning,
    VisionObserved,
)
from babymon.stats.clock import Calendar
from babymon.stats.timeline import Interval


@dataclass(frozen=True, slots=True)
class StatsInput:
    """Plain domain values, as the store returns them. `spans` must be contiguous over
    [start, now]; `camera_before` is the last CameraChanged ahead of the window."""

    start: Instant
    now: Instant
    tz: tzinfo
    tuning: Tuning
    spans: Sequence[SleepSpan]
    visions: Sequence[VisionObserved]
    alerts: Sequence[AlertSent]
    camera: Sequence[CameraChanged]
    camera_before: CameraChanged | None
    minutes: Sequence[MinuteRollup]


@dataclass(frozen=True, slots=True)
class Context:
    inp: StatsInput
    calendar: Calendar
    spans: Sequence[SleepSpan]
    """Observed: camera-down and process-gap time is blanked to state None."""
    blanked: Sequence[Interval]
    gaps: Sequence[Interval]
    """Stretches the process was not running; a subset of `blanked`."""
