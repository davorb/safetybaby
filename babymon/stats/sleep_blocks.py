"""The blocks built from sleep spans alone."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from datetime import date

from babymon.domain import SleepState
from babymon.stats import numerics
from babymon.stats.clock import SLEEP_DAY_START_MIN, SLOT_MIN, SLOTS_PER_DAY, local_minutes
from babymon.stats.context import Context
from babymon.stats.model import (
    Actogram,
    ActogramDay,
    ActogramSlot,
    Bouts,
    Circadian,
    Clock,
    Crying,
    DateCount,
    DayRow,
    Dwell,
    HistBin,
    HourStates,
    Regularity,
    StateTotal,
    SurvivalStep,
    Transition,
    Transitions,
)
from babymon.stats.timeline import Bout, Episode

STATES = tuple(SleepState)
NAP_MIN_S = 10 * 60
BEDTIME_MIN_BOUT_S = 20 * 60
HIST_BIN_S = 600
HIST_BINS = 72


def known_seconds(ctx: Context) -> dict[SleepState, float]:
    totals: dict[SleepState, float] = dict.fromkeys(STATES, 0.0)
    for span in ctx.spans:
        if span.state is not None:
            totals[span.state] += (span.end - span.start).total_seconds()
    return totals


def state_totals(ctx: Context) -> list[StateTotal]:
    totals = known_seconds(ctx)
    known = sum(totals.values())
    return [
        StateTotal(state=s, seconds=totals[s], fraction=totals[s] / known if known else None)
        for s in STATES
    ]


def transitions(ctx: Context) -> Transitions:
    spans = ctx.spans
    counts = {(a, b): 0 for a in STATES for b in STATES if a != b}
    for before, after in zip(spans, spans[1:], strict=False):
        if before.state is not None and after.state is not None:
            counts[(before.state, after.state)] += 1
    dwells: dict[SleepState, list[float]] = defaultdict(list)
    censored: dict[SleepState, int] = defaultdict(int)
    for i, span in enumerate(spans):
        if span.state is None:
            continue
        hidden = (
            i == 0
            or i == len(spans) - 1
            or spans[i - 1].state is None
            or spans[i + 1].state is None
        )
        if hidden:
            censored[span.state] += 1
        else:
            dwells[span.state].append((span.end - span.start).total_seconds())
    rows = []
    for state in STATES:
        values = sorted(dwells[state])
        rows.append(
            Dwell(
                state=state,
                n=len(values),
                n_censored=censored[state],
                mean_s=sum(values) / len(values) if values else None,
                median_s=numerics.quantile(values, 0.5),
            )
        )
    return Transitions(
        n=sum(counts.values()),
        counts=[Transition(previous=a, current=b, count=n) for (a, b), n in counts.items()],
        dwell=rows,
    )


def bouts_block(found: Sequence[Bout]) -> Bouts:
    complete = sorted(b.duration_s for b in found if not b.censored)
    censored = [b.duration_s for b in found if b.censored]
    curve = numerics.kaplan_meier(complete, censored)
    longest = max(found, key=lambda b: b.duration_s, default=None)
    return Bouts(
        n=len(found),
        n_censored=len(censored),
        brief_arousals=sum(b.brief_arousals for b in found),
        n_uncensored=len(complete),
        median_s=numerics.quantile(complete, 0.5),
        p25_s=numerics.quantile(complete, 0.25),
        p75_s=numerics.quantile(complete, 0.75),
        longest_s=longest.duration_s if longest else None,
        longest_censored=longest.censored if longest else False,
        histogram=_histogram(found),
        survival=[SurvivalStep(**asdict(p)) for p in curve],
        km_median_s=next((p.t_s for p in curve if p.survival <= 0.5), None),
    )


def _histogram(found: Sequence[Bout]) -> list[HistBin]:
    if not found:
        return []
    top = min(int(max(b.duration_s for b in found) // HIST_BIN_S), HIST_BINS - 1)
    counts = [[0, 0] for _ in range(top + 1)]
    for b in found:
        counts[min(int(b.duration_s // HIST_BIN_S), top)][b.censored] += 1
    return [
        HistBin(
            from_s=i * HIST_BIN_S,
            to_s=None if i == HIST_BINS - 1 else (i + 1) * HIST_BIN_S,
            count=done + cut,
            censored=cut,
        )
        for i, (done, cut) in enumerate(counts)
    ]


def circadian(ctx: Context, slot_motion: dict[tuple[date, int], tuple[float, int]]) -> Circadian:
    by_hour = [dict.fromkeys(STATES, 0.0) for _ in range(24)]
    by_slot: dict[tuple[date, int], dict[SleepState, float]] = defaultdict(
        lambda: dict.fromkeys(STATES, 0.0)
    )
    for span in ctx.spans:
        if span.state is None:
            continue
        for cell, secs in ctx.calendar.pieces(span.start, span.end):
            by_hour[cell.hour][span.state] += secs
            by_slot[(cell.day, cell.slot)][span.state] += secs
    hours = [HourStates(hour=h, seconds=s, known_s=sum(s.values())) for h, s in enumerate(by_hour)]
    days = []
    for day in ctx.calendar.days:
        slots = []
        for slot in range(SLOTS_PER_DAY):
            seconds = by_slot.get((day, slot))
            known = sum(seconds.values()) if seconds else 0.0
            motion = slot_motion.get((day, slot))
            slots.append(
                ActogramSlot(
                    state=max(STATES, key=lambda s: seconds[s]) if known else None,
                    known_s=known,
                    motion_mean=motion[0] / motion[1] if motion else None,
                    motion_minutes=motion[1] if motion else 0,
                )
            )
        days.append(ActogramDay(date=day, slots=slots))
    return Circadian(
        hours=hours, actogram=Actogram(origin_min=SLEEP_DAY_START_MIN, slot_min=SLOT_MIN, days=days)
    )


def day_rows(ctx: Context, found: Sequence[Bout], episodes: Sequence[Episode]) -> list[DayRow]:
    cal = ctx.calendar
    span_s: dict[date, float] = defaultdict(float)
    for cell, secs in cal.pieces(ctx.inp.start, ctx.inp.now):
        span_s[cell.day] += secs
    state_s = {d: dict.fromkeys(STATES, 0.0) for d in cal.days}
    asleep_night: dict[date, float] = defaultdict(float)
    for span in ctx.spans:
        if span.state is None:
            continue
        for cell, secs in cal.pieces(span.start, span.end):
            state_s[cell.day][span.state] += secs
            if span.state is SleepState.ASLEEP and cell.night:
                asleep_night[cell.day] += secs

    naps: dict[date, int] = defaultdict(int)
    wakings: dict[date, int] = defaultdict(int)
    longest: dict[date, Bout] = {}
    night_bouts: dict[date, list[Bout]] = defaultdict(list)
    for bout in found:
        day = cal.cell_at(bout.start).day
        if day not in longest or bout.duration_s > longest[day].duration_s:
            longest[day] = bout
        overlaps_night = {c.day for c, _ in cal.pieces(bout.start, bout.end) if c.night}
        for night_day in overlaps_night:
            if bout.duration_s >= BEDTIME_MIN_BOUT_S and night_day == day:
                night_bouts[night_day].append(bout)
        starts_in_day_window = not cal.cell_at(bout.start).night
        if starts_in_day_window and bout.duration_s >= NAP_MIN_S and not overlaps_night:
            naps[day] += 1
        end_cell = cal.cell_at(bout.end)
        if bout.woke and not bout.censored and end_cell.night:
            wakings[end_cell.day] += 1

    cry_s: dict[date, float] = defaultdict(float)
    for span in ctx.spans:
        if span.state is SleepState.CRYING:
            for cell, secs in cal.pieces(span.start, span.end):
                cry_s[cell.day] += secs
    cry_n: dict[date, int] = defaultdict(int)
    for episode in episodes:
        cry_n[cal.cell_at(episode.start).day] += 1

    rows = []
    for day in cal.days:
        states = state_s[day]
        night = night_bouts.get(day, [])
        bedtime = night[0] if night and not _left_cut(night[0], ctx) else None
        wake = night[-1] if night and not night[-1].censored else None
        rows.append(
            DayRow(
                date=day,
                span_s=span_s[day],
                covered_s=sum(states.values()),
                asleep_s=states[SleepState.ASLEEP],
                night_sleep_s=asleep_night[day],
                day_sleep_s=states[SleepState.ASLEEP] - asleep_night[day],
                state_s=states,
                nap_count=naps[day],
                longest_bout_s=longest[day].duration_s if day in longest else None,
                longest_bout_censored=longest[day].censored if day in longest else False,
                night_wakings=wakings[day],
                crying_s=cry_s[day],
                cry_episodes=cry_n[day],
                bedtime_min=local_minutes(bedtime.start, ctx.inp.tz) if bedtime else None,
                final_wake_min=local_minutes(wake.end, ctx.inp.tz) if wake else None,
            )
        )
    return rows


def _left_cut(bout: Bout, ctx: Context) -> bool:
    """A bout that begins at the window start or after unknown time has an unknown bedtime."""
    return bout.start <= ctx.inp.start or any(
        s.state is None and s.end == bout.start for s in ctx.spans
    )


def regularity(rows: Sequence[DayRow]) -> Regularity:
    return Regularity(
        bedtime=_clock([r.bedtime_min for r in rows]),
        final_wake=_clock([r.final_wake_min for r in rows]),
    )


def _clock(minutes: Sequence[int | None]) -> Clock:
    """Minutes are anchored at noon so 23:50 and 00:10 sit 20 apart instead of 1420."""
    anchored = [(m - 720) % 1440 for m in minutes if m is not None]
    mean, sd = numerics.mean_sd(anchored)
    return Clock(
        n=len(anchored), mean_min=(mean + 720) % 1440 if mean is not None else None, sd_min=sd
    )


def crying_block(ctx: Context, episodes: Sequence[Episode], rows: Sequence[DayRow]) -> Crying:
    durations = sorted(e.duration_s for e in episodes)
    by_hour = [0] * 24
    for e in episodes:
        by_hour[ctx.calendar.cell_at(e.start).hour] += 1
    return Crying(
        n=len(episodes),
        n_censored=sum(e.censored for e in episodes),
        total_s=sum(durations),
        median_s=numerics.quantile(durations, 0.5),
        longest_s=durations[-1] if durations else None,
        by_hour=by_hour,
        by_day=[DateCount(date=r.date, n=r.cry_episodes) for r in rows],
    )
