"""Statistics over the stored history. Pure: plain domain values in, a `Stats` model out.

`compute` is the only entry point. It builds the calendar and the observed spans once and
hands them to the block functions, so every block agrees on what "known time" means.
"""

from __future__ import annotations

from babymon.domain import Tuning
from babymon.stats import clock, sleep_blocks, timeline
from babymon.stats import signal_blocks as sig
from babymon.stats.clock import Calendar
from babymon.stats.context import Context, StatsInput
from babymon.stats.model import Coverage, DayRow, Stats, TuningEcho

__all__ = ["Stats", "StatsInput", "compute"]


def compute(inp: StatsInput) -> Stats:
    camera = timeline.camera_timeline(inp.camera, inp.camera_before, inp.start, inp.now)
    spans = timeline.observed_spans(inp.spans, camera.down, inp.start, inp.now)
    calendar = Calendar(inp.start, inp.now, inp.tz)
    ctx = Context(inp, calendar, spans, camera.down)

    bouts = timeline.bouts(spans)
    episodes = timeline.crying_episodes(spans)
    days = sleep_blocks.day_rows(ctx, bouts, episodes)
    return Stats(
        generated_at=inp.now,
        window_days=round((inp.now - inp.start).total_seconds() / 86400),
        coverage=_coverage(ctx, camera, days),
        state_totals=sleep_blocks.state_totals(ctx),
        days=days,
        bouts=sleep_blocks.bouts_block(bouts),
        transitions=sleep_blocks.transitions(ctx),
        circadian=sleep_blocks.circadian(ctx, sig.slot_motion(ctx)),
        regularity=sleep_blocks.regularity(days),
        crying=sleep_blocks.crying_block(ctx, episodes, days),
        vision=sig.vision_block(ctx),
        position_waking=sig.position_waking(ctx),
        alerts=sig.alerts_block(ctx),
        camera=sig.camera_block(camera, inp.now),
        signals=sig.signals_block(ctx),
        tuning=_tuning_echo(inp.tuning),
    )


def _coverage(ctx: Context, camera: timeline.CameraTimeline, days: list[DayRow]) -> Coverage:
    inp = ctx.inp
    window = (inp.now - inp.start).total_seconds()
    known = sum(r.covered_s for r in days)
    return Coverage(
        start=inp.start,
        end=inp.now,
        tz=str(inp.tz),
        window_s=window,
        sleep_known_s=known,
        camera_up_s=sum((hi - lo).total_seconds() for lo, hi in camera.up),
        camera_down_s=sum((hi - lo).total_seconds() for lo, hi in camera.down),
        camera_unknown_s=sum((hi - lo).total_seconds() for lo, hi in camera.unknown),
        fraction_covered=known / window if window else 0.0,
        days_with_data=sum(r.covered_s > 0 for r in days),
        minutes_with_data=len(inp.minutes),
    )


def _tuning_echo(tuning: Tuning) -> TuningEcho:
    return TuningEcho(
        motion_moving=tuning.motion_moving,
        motion_active=tuning.motion_active,
        cry_probability=tuning.cry_probability,
        min_confidence=tuning.min_confidence,
        crying_alert_s=tuning.crying_alert_s,
        vision_period_quiet_s=tuning.vision_period_quiet_s,
        night_start_min=clock.NIGHT_START_MIN,
        night_end_min=clock.NIGHT_END_MIN,
        sleep_day_start_min=clock.SLEEP_DAY_START_MIN,
        brief_arousal_s=timeline.BRIEF_AROUSAL.total_seconds(),
        bedtime_min_bout_s=sleep_blocks.BEDTIME_MIN_BOUT_S,
        nap_min_s=sleep_blocks.NAP_MIN_S,
        position_horizon_s=sig.POSITION_HORIZON.total_seconds(),
        adult_gap_s=sig.ADULT_GAP.total_seconds(),
    )
