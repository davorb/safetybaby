"""The /api/stats JSON contract. Every block carries its own denominator (`n`, seconds
covered) so a page never shows a number without the evidence behind it.

Units: durations in seconds, instants as ISO UTC, `*_min` fields as minutes since local
midnight, `date` as the local sleep-day label (the date its 07:00 start falls on).
"""

from __future__ import annotations

from datetime import date

from babymon.domain import (
    AlertKind,
    BabyLocation,
    BodyPosition,
    Eyes,
    Frozen,
    HeadDirection,
    Instant,
    SleepState,
    VisionReason,
)


class Coverage(Frozen):
    start: Instant
    end: Instant
    tz: str
    window_s: float
    sleep_known_s: float
    """Seconds with a known sleep state: a state exists and the camera was not known down."""
    camera_up_s: float
    camera_down_s: float
    camera_unknown_s: float
    process_gap_s: float
    """Time the process was not running (a hole in the minute rollups, or a restart). Counted
    as unknown, not as a continuation of the last sleep state."""
    fraction_covered: float
    """sleep_known_s / window_s."""
    days_with_data: int
    """Sleep days with at least one second of known sleep state."""
    minutes_with_data: int
    """Rows in the minutes table inside the window; a full hour is 60."""


class StateTotal(Frozen):
    state: SleepState
    seconds: float
    fraction: float | None
    """Of known time; None when nothing is known."""


class Dwell(Frozen):
    state: SleepState
    n: int
    n_censored: int
    """Spans excluded because the window edge or unknown time hides one of their ends."""
    mean_s: float | None
    median_s: float | None


class Transition(Frozen):
    previous: SleepState
    current: SleepState
    count: int


class Transitions(Frozen):
    n: int
    counts: list[Transition]
    """Every previous != current pair, zero counts included."""
    dwell: list[Dwell]


class DayRow(Frozen):
    date: date
    span_s: float
    """Seconds of this sleep day that fall inside the window."""
    covered_s: float
    asleep_s: float
    night_sleep_s: float
    day_sleep_s: float
    state_s: dict[SleepState, float]
    nap_count: int
    longest_bout_s: float | None
    longest_bout_censored: bool
    night_wakings: int
    crying_s: float
    cry_episodes: int
    bedtime_min: int | None
    final_wake_min: int | None


class HistBin(Frozen):
    from_s: float
    to_s: float | None
    """None for the last bin, which holds everything longer."""
    count: int
    censored: int


class SurvivalStep(Frozen):
    t_s: float
    at_risk: int
    events: int
    survival: float
    lo: float
    hi: float


class Bouts(Frozen):
    n: int
    n_censored: int
    """Bouts that touch unknown time, the window start or now. Their length is a lower bound."""
    brief_arousals: int
    """Interruptions under 3 minutes absorbed into a bout."""
    n_uncensored: int
    median_s: float | None
    p25_s: float | None
    p75_s: float | None
    """Quantiles are over uncensored bouts only; censored ones would drag them down."""
    longest_s: float | None
    longest_censored: bool
    histogram: list[HistBin]
    survival: list[SurvivalStep]
    km_median_s: float | None
    """Where the Kaplan-Meier curve first reaches 0.5 or below; None if it never does."""


class HourStates(Frozen):
    hour: int
    seconds: dict[SleepState, float]
    known_s: float


class ActogramSlot(Frozen):
    state: SleepState | None
    """Majority state by seconds inside the slot; None with nothing known."""
    known_s: float
    motion_mean: float | None
    motion_minutes: int


class ActogramDay(Frozen):
    date: date
    slots: list[ActogramSlot]


class Actogram(Frozen):
    origin_min: int
    """Local minute of day where slot 0 starts (07:00)."""
    slot_min: int
    days: list[ActogramDay]


class Circadian(Frozen):
    hours: list[HourStates]
    actogram: Actogram


class Clock(Frozen):
    n: int
    mean_min: float | None
    sd_min: float | None
    """Sample SD in minutes; None below two days."""


class Regularity(Frozen):
    bedtime: Clock
    final_wake: Clock


class DateCount(Frozen):
    date: date
    n: int


class Crying(Frozen):
    n: int
    n_censored: int
    total_s: float
    median_s: float | None
    longest_s: float | None
    by_hour: list[int]
    """Episodes by the local hour they started in, 24 entries."""
    by_day: list[DateCount]


class ReasonCount(Frozen):
    reason: VisionReason
    n: int


class ConfidenceBin(Frozen):
    lo: float
    hi: float
    count: int


class ValueShare[T](Frozen):
    value: T
    seconds: float
    fraction: float | None


class Held[T](Frozen):
    """Time-weighted share of each value over the confident observations."""

    seconds: float
    shares: list[ValueShare[T]]


class DayVisits(Frozen):
    date: date
    total: int
    night: int


class AdultVisits(Frozen):
    n: int
    by_day: list[DayVisits]
    by_hour: list[int]
    """Visits by the local hour they began in, 24 entries."""


class Vision(Frozen):
    n: int
    by_reason: list[ReasonCount]
    by_day: list[DateCount]
    confidence: list[ConfidenceBin]
    n_low_confidence: int
    low_confidence_share: float | None
    n_confident: int
    face_covered_readings: int
    """Confident checks that read face_covered."""
    body: Held[BodyPosition]
    head: Held[HeadDirection]
    eyes: Held[Eyes]
    location: Held[BabyLocation]
    adult_visits: AdultVisits


class RateRow[T](Frozen):
    value: T
    n: int
    wakes: int
    rate: float | None
    ci_lo: float | None
    ci_hi: float | None


class PositionWaking(Frozen):
    """Association, not causation: babies are laid in a position, and a restless baby
    changes position. Checks inside one bout are not independent either."""

    horizon_s: float
    n: int
    wakes: int
    rate: float | None
    ci_lo: float | None
    ci_hi: float | None
    n_unresolved: int
    """Asleep checks left out because the follow-up window was cut by now or unknown time."""
    body: list[RateRow[BodyPosition]]
    head: list[RateRow[HeadDirection]]


class KindAlerts(Frozen):
    kind: AlertKind
    count: int
    delivered: int
    by_hour: list[int]


class Alerts(Frozen):
    n: int
    delivered: int
    by_kind: list[KindAlerts]
    by_day: list[DateCount]


class Outage(Frozen):
    start: Instant
    end: Instant
    duration_s: float
    ongoing: bool


class Camera(Frozen):
    known_s: float
    n_outages: int
    down_s: float
    longest_outage_s: float | None
    uptime_fraction: float | None
    outages: list[Outage]


class Quantiles(Frozen):
    n: int
    p10: float | None
    p25: float | None
    median: float | None
    p75: float | None
    p90: float | None


class StateSignal(Frozen):
    state: SleepState
    n_minutes: int
    motion_mean: Quantiles
    """Distribution of each minute's mean motion level."""
    motion_peak: Quantiles
    """Distribution of each minute's peak motion level; closest to what the state machine
    thresholds see, since it classifies on the per-second peak."""
    cry_minutes: int
    """Minutes with at least one second at or above the cry threshold."""


class HourLoudness(Frozen):
    hour: int
    n_samples: int
    mean_db: float | None


class Signals(Frozen):
    n_minutes: int
    n_unattributed: int
    """Minutes with no known sleep state, left out of by_state."""
    by_state: list[StateSignal]
    loudness_by_hour: list[HourLoudness]


class TuningEcho(Frozen):
    motion_moving: float
    motion_active: float
    cry_probability: float
    min_confidence: float
    crying_alert_s: float
    vision_period_quiet_s: float
    night_start_min: int
    night_end_min: int
    sleep_day_start_min: int
    brief_arousal_s: float
    bedtime_min_bout_s: float
    nap_min_s: float
    position_horizon_s: float
    adult_gap_s: float


class Stats(Frozen):
    generated_at: Instant
    window_days: int
    coverage: Coverage
    state_totals: list[StateTotal]
    days: list[DayRow]
    bouts: Bouts
    transitions: Transitions
    circadian: Circadian
    regularity: Regularity
    crying: Crying
    vision: Vision
    position_waking: PositionWaking
    alerts: Alerts
    camera: Camera
    signals: Signals
    tuning: TuningEcho
