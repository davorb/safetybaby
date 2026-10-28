"""The blocks built from observations, alerts, the camera and the minute rollups."""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from datetime import date, timedelta
from enum import StrEnum

from babymon.domain import (
    AlertKind,
    BabyLocation,
    BodyPosition,
    Eyes,
    HeadDirection,
    Instant,
    SleepSpan,
    SleepState,
    VisionObserved,
    VisionReason,
)
from babymon.stats import numerics
from babymon.stats.context import Context
from babymon.stats.model import (
    AdultVisits,
    Alerts,
    Camera,
    ConfidenceBin,
    DateCount,
    DayVisits,
    Held,
    HourLoudness,
    KindAlerts,
    Outage,
    PositionWaking,
    Quantiles,
    RateRow,
    ReasonCount,
    Signals,
    StateSignal,
    ValueShare,
    Vision,
)
from babymon.stats.sleep_blocks import STATES
from babymon.stats.timeline import CameraTimeline, subtract

POSITION_HORIZON = timedelta(minutes=30)
ADULT_GAP = timedelta(minutes=5)


def _seconds(intervals: Iterable[tuple[Instant, Instant]]) -> float:
    return sum((hi - lo).total_seconds() for lo, hi in intervals)


def _by_day(ctx: Context, times: Iterable[Instant]) -> list[DateCount]:
    counts = Counter(ctx.calendar.cell_at(t).day for t in times)
    return [DateCount(date=d, n=counts[d]) for d in ctx.calendar.days]


def _by_hour(ctx: Context, times: Iterable[Instant]) -> list[int]:
    counts = Counter(ctx.calendar.cell_at(t).hour for t in times)
    return [counts[h] for h in range(24)]


def camera_block(camera: CameraTimeline, now: Instant) -> Camera:
    known = _seconds(camera.up) + _seconds(camera.down)
    outages = [
        Outage(start=lo, end=hi, duration_s=(hi - lo).total_seconds(), ongoing=hi == now)
        for lo, hi in camera.down
    ]
    return Camera(
        known_s=known,
        n_outages=len(outages),
        down_s=_seconds(camera.down),
        longest_outage_s=max((o.duration_s for o in outages), default=None),
        uptime_fraction=_seconds(camera.up) / known if known else None,
        outages=outages,
    )


def alerts_block(ctx: Context) -> Alerts:
    sent = ctx.inp.alerts
    kinds = []
    for kind in AlertKind:
        mine = [a for a in sent if a.alert.kind is kind]
        kinds.append(
            KindAlerts(
                kind=kind,
                count=len(mine),
                delivered=sum(a.delivered for a in mine),
                by_hour=_by_hour(ctx, (a.at for a in mine)),
            )
        )
    return Alerts(
        n=len(sent),
        delivered=sum(a.delivered for a in sent),
        by_kind=kinds,
        by_day=_by_day(ctx, (a.at for a in sent)),
    )


def _held[T: StrEnum](
    held: Sequence[tuple[VisionObserved, float]],
    pick: Callable[[VisionObserved], T],
    values: type[T],
) -> Held[T]:
    seconds: dict[T, float] = dict.fromkeys(values, 0.0)
    for obs, secs in held:
        seconds[pick(obs)] += secs
    total = sum(seconds.values())
    return Held(
        seconds=total,
        shares=[
            ValueShare(value=v, seconds=s, fraction=s / total if total else None)
            for v, s in seconds.items()
        ],
    )


def _held_seconds(
    ctx: Context, confident: Sequence[VisionObserved]
) -> list[tuple[VisionObserved, float]]:
    """A confident reading holds until the next reading of any confidence, at most
    vision_period_quiet_s, and never across a stretch the camera was down."""
    cap = timedelta(seconds=ctx.inp.tuning.vision_period_quiet_s)
    every = sorted(ctx.inp.visions, key=lambda v: v.at)
    following = {id(v): n.at for v, n in zip(every, every[1:], strict=False)}
    out = []
    for obs in confident:
        end = min(following.get(id(obs), ctx.inp.now), obs.at + cap, ctx.inp.now)
        out.append((obs, _seconds(subtract(obs.at, end, ctx.down))))
    return out


def vision_block(ctx: Context) -> Vision:
    visions = ctx.inp.visions
    confident = [v for v in visions if v.observation.confidence >= ctx.inp.tuning.min_confidence]
    held = _held_seconds(ctx, confident)
    reasons = Counter(v.reason for v in visions)
    bins = Counter(min(int(v.observation.confidence * 10), 9) for v in visions)
    low = len(visions) - len(confident)
    return Vision(
        n=len(visions),
        by_reason=[ReasonCount(reason=r, n=reasons[r]) for r in VisionReason],
        by_day=_by_day(ctx, (v.at for v in visions)),
        confidence=[ConfidenceBin(lo=i / 10, hi=(i + 1) / 10, count=bins[i]) for i in range(10)],
        n_low_confidence=low,
        low_confidence_share=low / len(visions) if visions else None,
        n_confident=len(confident),
        face_covered_readings=sum(v.observation.face_covered for v in confident),
        body=_held(held, lambda v: v.observation.body, BodyPosition),
        head=_held(held, lambda v: v.observation.head, HeadDirection),
        eyes=_held(held, lambda v: v.observation.eyes, Eyes),
        location=_held(held, lambda v: v.observation.location, BabyLocation),
        adult_visits=_adult_visits(ctx, confident),
    )


def _adult_visits(ctx: Context, confident: Sequence[VisionObserved]) -> AdultVisits:
    """A visit is a run of adult_present readings with no gap over five minutes; it starts
    at its first reading."""
    starts: list[Instant] = []
    previous: Instant | None = None
    for obs in sorted((v for v in confident if v.observation.adult_present), key=lambda v: v.at):
        if previous is None or obs.at - previous > ADULT_GAP:
            starts.append(obs.at)
        previous = obs.at
    night = Counter(ctx.calendar.cell_at(t).day for t in starts if ctx.calendar.cell_at(t).night)
    total = Counter(ctx.calendar.cell_at(t).day for t in starts)
    return AdultVisits(
        n=len(starts),
        by_day=[DayVisits(date=d, total=total[d], night=night[d]) for d in ctx.calendar.days],
        by_hour=_by_hour(ctx, starts),
    )


def _rate_rows[T: StrEnum](outcomes: Sequence[tuple[T, bool]], values: type[T]) -> list[RateRow[T]]:
    rows = []
    for value in values:
        mine = [woke for v, woke in outcomes if v == value]
        wakes = sum(mine)
        interval = numerics.wilson(wakes, len(mine))
        rows.append(
            RateRow(
                value=value,
                n=len(mine),
                wakes=wakes,
                rate=wakes / len(mine) if mine else None,
                ci_lo=interval[0] if interval else None,
                ci_hi=interval[1] if interval else None,
            )
        )
    return rows


def position_waking(ctx: Context) -> PositionWaking:
    """Each confident check taken while ASLEEP asks: did AWAKE or CRYING start within 30
    minutes? A check whose 30 minutes are not fully observed is left out whatever happened,
    because keeping only the ones that ended in a wake would inflate the rate."""
    spans, now = ctx.spans, ctx.inp.now
    starts = [s.start for s in spans]
    asleep: list[tuple[VisionObserved, bool]] = []
    unresolved = 0
    for obs in ctx.inp.visions:
        if obs.observation.confidence < ctx.inp.tuning.min_confidence:
            continue
        i = bisect_right(starts, obs.at) - 1
        if i < 0 or spans[i].state is not SleepState.ASLEEP:
            continue
        horizon = obs.at + POSITION_HORIZON
        ahead = [s for s in spans[i + 1 :] if s.start < horizon]
        if horizon > now or any(s.state is None for s in ahead):
            unresolved += 1
            continue
        asleep.append((obs, any(s.state in (SleepState.AWAKE, SleepState.CRYING) for s in ahead)))
    wakes = sum(woke for _, woke in asleep)
    overall = numerics.wilson(wakes, len(asleep))
    return PositionWaking(
        horizon_s=POSITION_HORIZON.total_seconds(),
        n=len(asleep),
        wakes=wakes,
        rate=wakes / len(asleep) if asleep else None,
        ci_lo=overall[0] if overall else None,
        ci_hi=overall[1] if overall else None,
        n_unresolved=unresolved,
        body=_rate_rows([(o.observation.body, w) for o, w in asleep], BodyPosition),
        head=_rate_rows([(o.observation.head, w) for o, w in asleep], HeadDirection),
    )


def _majority(
    spans: Sequence[SleepSpan], starts: Sequence[Instant], lo: Instant
) -> SleepState | None:
    """State holding the most known seconds of the minute starting at `lo`."""
    hi = lo + timedelta(minutes=1)
    seconds: dict[SleepState, float] = defaultdict(float)
    for span in spans[max(0, bisect_right(starts, lo) - 1) :]:
        if span.start >= hi:
            break
        if span.state is not None:
            seconds[span.state] += (min(span.end, hi) - max(span.start, lo)).total_seconds()
    return max(seconds, key=lambda s: seconds[s]) if seconds else None


def _quantiles(values: list[float]) -> Quantiles:
    values.sort()
    q = [numerics.quantile(values, p) for p in (0.1, 0.25, 0.5, 0.75, 0.9)]
    return Quantiles(n=len(values), p10=q[0], p25=q[1], median=q[2], p75=q[3], p90=q[4])


def signals_block(ctx: Context) -> Signals:
    spans, minutes = ctx.spans, ctx.inp.minutes
    starts = [s.start for s in spans]
    mean: dict[SleepState, list[float]] = defaultdict(list)
    peak: dict[SleepState, list[float]] = defaultdict(list)
    cry: Counter[SleepState] = Counter()
    minutes_in: Counter[SleepState] = Counter()
    unattributed = 0
    for m in minutes:
        state = _majority(spans, starts, m.at)
        if state is None:
            unattributed += 1
            continue
        minutes_in[state] += 1
        cry[state] += m.cry_seconds > 0
        if m.motion_mean is not None and m.motion_peak is not None:
            mean[state].append(m.motion_mean)
            peak[state].append(m.motion_peak)
    loud_sum, loud_n = [0.0] * 24, [0] * 24
    for m in minutes:
        if m.loudness_mean_db is not None:
            hour = ctx.calendar.cell_at(m.at).hour
            loud_sum[hour] += m.loudness_mean_db * m.audio_samples
            loud_n[hour] += m.audio_samples
    return Signals(
        n_minutes=len(minutes),
        n_unattributed=unattributed,
        by_state=[
            StateSignal(
                state=s,
                n_minutes=minutes_in[s],
                motion_mean=_quantiles(mean[s]),
                motion_peak=_quantiles(peak[s]),
                cry_minutes=cry[s],
            )
            for s in STATES
        ],
        loudness_by_hour=[
            HourLoudness(
                hour=h, n_samples=loud_n[h], mean_db=loud_sum[h] / loud_n[h] if loud_n[h] else None
            )
            for h in range(24)
        ],
    )


def slot_motion(ctx: Context) -> dict[tuple[date, int], tuple[float, int]]:
    """(sum of minute means, minutes) per actogram slot."""
    out: dict[tuple[date, int], tuple[float, int]] = {}
    for m in ctx.inp.minutes:
        if m.motion_mean is None:
            continue
        cell = ctx.calendar.cell_at(m.at)
        total, n = out.get((cell.day, cell.slot), (0.0, 0))
        out[(cell.day, cell.slot)] = (total + m.motion_mean, n + 1)
    return out
