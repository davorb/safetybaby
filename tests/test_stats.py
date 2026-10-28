from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from babymon.domain import (
    Alert,
    AlertKind,
    AlertSent,
    CameraChanged,
    MinuteRollup,
    Priority,
    SleepSpan,
    SleepState,
    SnapshotRef,
    Tuning,
    VisionObserved,
    VisionReason,
)
from babymon.stats import Stats, StatsInput, compute
from babymon.stats.numerics import kaplan_meier, wilson
from tests.conftest import observation

A, S, W, C = SleepState.ASLEEP, SleepState.STIRRING, SleepState.AWAKE, SleepState.CRYING
UTC_ZONE = ZoneInfo("UTC")
STOCKHOLM = ZoneInfo("Europe/Stockholm")
MIN = timedelta(minutes=1)
H = timedelta(hours=1)


def utc(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


def timeline(start: datetime, *runs: tuple[SleepState | None, timedelta]) -> list[SleepSpan]:
    spans, cursor = [], start
    for state, length in runs:
        spans.append(SleepSpan(state, cursor, cursor + length))
        cursor += length
    return spans


def stats(spans, *, now=None, tz=UTC_ZONE, tuning=None, **inputs) -> Stats:
    start = spans[0].start if spans else utc("2026-10-27T00:00")
    end = now or (spans[-1].end if spans else start + H)
    return compute(
        StatsInput(
            start=start,
            now=end,
            tz=tz,
            tuning=tuning or Tuning(),
            spans=spans,
            visions=inputs.get("visions", []),
            alerts=inputs.get("alerts", []),
            camera=inputs.get("camera", []),
            camera_before=inputs.get("camera_before"),
            minutes=inputs.get("minutes", []),
        )
    )


def check(at: datetime, **overrides: object) -> VisionObserved:
    return VisionObserved(
        at=at,
        reason=VisionReason.PERIODIC,
        observation=observation(**overrides),
        snapshot=SnapshotRef("s.jpg"),
    )


def by_state(items, key="seconds"):
    return {i.state: getattr(i, key) for i in items}


class TestEmptyWindow:
    def test_no_data_gives_zeros_and_no_crash(self):
        result = stats([])

        assert result.coverage.sleep_known_s == 0
        assert result.coverage.fraction_covered == 0
        assert all(t.seconds == 0 and t.fraction is None for t in result.state_totals)
        assert result.bouts.n == 0 and result.bouts.survival == [] and result.bouts.median_s is None
        assert result.crying.n == 0 and result.alerts.n == 0 and result.vision.n == 0
        assert result.regularity.bedtime.n == 0 and result.regularity.bedtime.mean_min is None
        assert result.position_waking.rate is None
        assert result.camera.uptime_fraction is None
        assert result.vision.low_confidence_share is None

    def test_a_window_of_unknown_state_is_all_unknown(self):
        result = stats(timeline(utc("2026-10-27T00:00"), (None, 6 * H)))

        assert result.coverage.sleep_known_s == 0
        assert result.coverage.days_with_data == 0
        assert result.bouts.n == 0


class TestCalendar:
    def test_a_night_across_the_dst_end_lands_in_the_right_hours(self):
        start = utc("2026-10-24T15:00")
        result = stats(
            timeline(start, (W, 2 * H), (A, 12 * H), (W, 3 * H)),
            tz=STOCKHOLM,
        )

        [day] = [d for d in result.days if d.asleep_s > 0]
        assert day.date.isoformat() == "2026-10-24"
        assert day.night_sleep_s == 12 * 3600 and day.day_sleep_s == 0
        assert day.bedtime_min == 19 * 60
        assert day.final_wake_min == 6 * 60
        hours = {h.hour: h for h in result.circadian.hours}
        assert hours[2].seconds[A] == 2 * 3600
        assert hours[23].seconds[A] == 3600
        assert sum(h.known_s for h in result.circadian.hours) == 17 * 3600

    def test_the_sleep_day_turns_over_at_seven_local(self):
        result = stats(timeline(utc("2026-10-27T06:00"), (A, 2 * H)), tz=UTC_ZONE)

        assert {d.date.isoformat(): d.asleep_s for d in result.days} == {
            "2026-10-26": 3600,
            "2026-10-27": 3600,
        }

    def test_actogram_slots_start_at_seven_and_hold_the_majority_state(self):
        start = utc("2026-10-27T07:00")
        result = stats(timeline(start, (A, 20 * MIN), (W, 10 * MIN), (W, 30 * MIN), (A, 30 * MIN)))

        slots = result.circadian.actogram.days[-1].slots
        assert (result.circadian.actogram.origin_min, len(slots)) == (420, 48)
        assert [s.state for s in slots[:4]] == [A, W, A, None]
        assert slots[3].known_s == 0


class TestBouts:
    def test_brief_interruptions_merge_into_one_bout_and_long_ones_split(self):
        spans = timeline(
            utc("2026-10-27T19:00"),
            (W, 10 * MIN),
            (A, H),
            (S, 2 * MIN),
            (A, 58 * MIN),
            (W, 5 * MIN),
            (A, 55 * MIN),
        )

        result = stats(spans).bouts

        assert (result.n, result.brief_arousals, result.n_censored) == (2, 1, 1)
        assert (result.median_s, result.longest_s, result.longest_censored) == (7200, 7200, False)

    def test_crying_is_never_absorbed(self):
        spans = timeline(
            utc("2026-10-27T19:00"),
            (W, 5 * MIN),
            (A, 30 * MIN),
            (C, MIN),
            (A, 30 * MIN),
            (W, 5 * MIN),
        )

        assert stats(spans).bouts.n == 2

    def test_a_bout_touching_unknown_time_or_now_is_censored(self):
        spans = timeline(
            utc("2026-10-27T19:00"), (W, 5 * MIN), (A, 30 * MIN), (None, 10 * MIN), (A, 30 * MIN)
        )

        bouts = stats(spans).bouts

        assert (bouts.n, bouts.n_censored, bouts.n_uncensored) == (2, 2, 0)
        assert bouts.median_s is None and bouts.longest_censored

    def test_a_camera_outage_blanks_the_sleep_state_it_covers(self):
        start = utc("2026-10-27T19:00")
        spans = timeline(start, (W, 5 * MIN), (A, H), (W, 5 * MIN))
        down = CameraChanged(at=start + 25 * MIN, up=False, detail="")
        up = CameraChanged(at=start + 45 * MIN, up=True, detail="")

        result = stats(
            spans, camera=[down, up], camera_before=CameraChanged(at=start - H, up=True, detail="")
        )

        assert result.camera.n_outages == 1 and result.camera.down_s == 20 * 60
        assert result.camera.longest_outage_s == 1200
        assert result.camera.uptime_fraction == pytest.approx(50 / 70)
        assert result.coverage.sleep_known_s == 50 * 60
        assert result.bouts.n == 2 and result.bouts.n_censored == 2
        assert by_state(result.state_totals)[A] == 40 * 60

    def test_night_wakings_count_real_wakings_not_stirs(self):
        spans = timeline(
            utc("2026-10-27T19:00"),
            (W, 10 * MIN),
            (A, 3 * H),
            (S, 5 * MIN),
            (W, 20 * MIN),
            (A, 3 * H),
            (S, 6 * MIN),
            (A, 2 * H),
            (W, 30 * MIN),
        )

        day = [d for d in stats(spans).days if d.night_wakings or d.asleep_s][0]

        assert day.night_wakings == 2
        assert day.bedtime_min == 19 * 60 + 10
        assert day.final_wake_min == 3 * 60 + 41

    def test_naps_exclude_the_night_bout(self):
        spans = timeline(
            utc("2026-10-27T07:00"),
            (A, 90 * MIN),
            (W, 3 * H),
            (A, 5 * MIN),
            (W, H),
            (A, 45 * MIN),
            (W, 3 * H),
            (A, 8 * H),
            (W, H),
        )

        day = stats(spans).days[0]

        assert day.nap_count == 2
        assert day.bedtime_min is not None

    def test_longest_bout_is_the_longest_that_started_that_day(self):
        spans = timeline(
            utc("2026-10-27T07:00"), (A, 40 * MIN), (W, 10 * MIN), (A, 80 * MIN), (W, 10 * MIN)
        )

        assert stats(spans).days[0].longest_bout_s == 80 * 60


class TestSurvival:
    def test_kaplan_meier_matches_the_hand_computation(self):
        curve = kaplan_meier([3.0, 5.0, 5.0, 8.0], [4.0, 6.0])

        assert [(p.t_s, p.at_risk, p.events) for p in curve] == [
            (0.0, 6, 0),
            (3.0, 6, 1),
            (5.0, 4, 2),
            (8.0, 1, 1),
        ]
        assert [p.survival for p in curve] == pytest.approx([1.0, 5 / 6, 5 / 12, 0.0])

    def test_without_events_the_curve_stays_at_one(self):
        curve = kaplan_meier([], [10.0, 20.0])

        assert [(p.t_s, p.survival) for p in curve] == [(0.0, 1.0)]

    def test_the_median_is_where_the_curve_first_reaches_one_half(self):
        spans = timeline(
            utc("2026-10-27T19:00"),
            (W, MIN),
            (A, 30 * MIN),
            (W, 5 * MIN),
            (A, 60 * MIN),
            (W, 5 * MIN),
        )

        assert stats(spans).bouts.km_median_s == 30 * 60


class TestWilson:
    def test_known_values(self):
        low, high = wilson(0, 10)
        assert (low, high) == (0.0, pytest.approx(0.2775, abs=1e-3))
        low, high = wilson(5, 10)
        assert (low, high) == (pytest.approx(0.2366, abs=1e-3), pytest.approx(0.7634, abs=1e-3))
        assert wilson(0, 0) is None


class TestRegularity:
    def test_bedtimes_either_side_of_midnight_average_to_midnight(self):
        spans = timeline(
            utc("2026-10-27T10:00"),
            (W, 13 * H + 50 * MIN),
            (A, 6 * H + 10 * MIN),
            (W, 18 * H + 10 * MIN),
            (A, 6 * H),
            (W, H),
        )

        result = stats(spans).regularity

        assert result.bedtime.n == 2
        assert result.bedtime.mean_min == pytest.approx(0.0, abs=1e-6)
        assert result.bedtime.sd_min == pytest.approx(14.142, abs=0.01)
        assert result.final_wake.mean_min == pytest.approx(365)


class TestCrying:
    def test_episodes_split_by_state_change(self):
        spans = timeline(
            utc("2026-10-27T01:00"),
            (A, 10 * MIN),
            (C, 2 * MIN),
            (W, MIN),
            (C, 4 * MIN),
            (A, 10 * MIN),
        )

        crying = stats(spans).crying

        assert (crying.n, crying.total_s, crying.median_s, crying.longest_s) == (2, 360, 180, 240)
        assert crying.by_hour[1] == 2


class TestVision:
    def test_confident_readings_hold_until_the_next_reading_or_the_quiet_cap(self):
        t = utc("2026-10-27T01:00")
        visions = [
            check(t, body="back"),
            check(t + 2 * MIN, body="stomach"),
            check(t + 3 * MIN, body="side_left", confidence=0.2),
            check(t + 60 * MIN, body="back"),
        ]
        spans = timeline(t, (A, 2 * H))

        vision = stats(spans, visions=visions).vision

        body = {s.value: s.seconds for s in vision.body.shares}
        assert body["back"] == 120 + 900
        assert body["stomach"] == 60
        assert body["side_left"] == 0
        assert (vision.n, vision.n_low_confidence, vision.n_confident) == (4, 1, 3)
        assert vision.low_confidence_share == 0.25
        assert vision.confidence[9].count == 3 and vision.confidence[2].count == 1

    def test_adult_visits_join_readings_five_minutes_apart(self):
        t = utc("2026-10-27T02:00")
        readings = [0, 4, 8, 30, 33, 120]
        visions = [check(t + m * MIN, adult_present=True) for m in readings]

        visits = stats(timeline(t, (W, 3 * H)), visions=visions).vision.adult_visits

        assert visits.n == 3
        assert visits.by_hour[2] == 2 and visits.by_hour[4] == 1
        assert visits.by_day[0].night == 3

    def test_face_covered_counts_confident_readings(self):
        t = utc("2026-10-27T02:00")
        visions = [check(t, face_covered=True), check(t + MIN, face_covered=True, confidence=0.1)]

        assert stats(timeline(t, (A, H)), visions=visions).vision.face_covered_readings == 1


class TestPositionAndWaking:
    def test_rate_counts_wakes_within_thirty_minutes_and_skips_unresolved_checks(self):
        t = utc("2026-10-27T01:00")
        spans = timeline(t, (A, 20 * MIN), (S, MIN), (W, 10 * MIN), (A, 2 * H))
        visions = [
            check(t + MIN, body="back"),
            check(t + 40 * MIN, body="back"),
            check(t + 50 * MIN, body="stomach", head="left"),
            check(t + 15 * MIN, body="side_left"),
            check(t + 2 * H + 20 * MIN, body="back"),
        ]

        result = stats(spans, visions=visions).position_waking

        rows = {r.value: r for r in result.body}
        assert (rows["back"].n, rows["back"].wakes) == (2, 1)
        assert rows["back"].rate == 0.5
        assert rows["stomach"].n == 1 and rows["stomach"].wakes == 0
        assert rows["side_left"].wakes == 1
        assert result.n_unresolved == 1
        assert result.n == 4 and result.wakes == 2
        head = {r.value: r for r in result.head}
        assert head["left"].n == 1 and head["center"].n == 3
        assert rows["back"].ci_lo == pytest.approx(wilson(1, 2)[0])


class TestAlertsAndSignals:
    def test_alerts_count_by_kind_hour_and_delivery(self):
        t = utc("2026-10-27T02:10")

        def sent(at, kind, delivered):
            alert = Alert(kind=kind, priority=Priority.HIGH, at=at, title="t", message="m")
            return AlertSent(at=at, alert=alert, snapshot=None, delivered=delivered)

        alerts = [sent(t, AlertKind.CRYING, True), sent(t + 5 * MIN, AlertKind.CRYING, False)]
        result = stats(timeline(t, (A, H)), alerts=alerts).alerts

        crying = next(k for k in result.by_kind if k.kind is AlertKind.CRYING)
        assert (result.n, result.delivered, crying.count, crying.delivered) == (2, 1, 2, 1)
        assert crying.by_hour[2] == 2

    def test_minutes_are_attributed_to_their_majority_state(self):
        t = utc("2026-10-27T02:00")
        spans = timeline(t, (A, 10 * MIN), (W, 10 * MIN))

        def minute(offset, mean, peak, cry=0):
            return MinuteRollup(t + offset * MIN, 240, mean, peak, 60, 4.0, 9.0, 0.1, cry)

        minutes = [minute(0, 0.001, 0.01), minute(5, 0.003, 0.02, cry=2), minute(12, 0.05, 0.3)]
        result = stats(spans, minutes=minutes).signals

        asleep = next(s for s in result.by_state if s.state is A)
        awake = next(s for s in result.by_state if s.state is W)
        assert (asleep.n_minutes, asleep.cry_minutes) == (2, 1)
        assert asleep.motion_mean.median == pytest.approx(0.002)
        assert awake.motion_peak.median == 0.3
        assert result.n_unattributed == 0
        assert (
            result.loudness_by_hour[2].mean_db == 4.0
            and result.loudness_by_hour[2].n_samples == 180
        )
        slots = stats(spans, minutes=minutes).circadian.actogram.days[0].slots
        assert [s.motion_mean for s in slots if s.motion_mean is not None] == pytest.approx(
            [(0.001 + 0.003 + 0.05) / 3]
        )

    def test_transitions_and_dwell_skip_censored_spans(self):
        spans = timeline(
            utc("2026-10-27T02:00"),
            (W, 5 * MIN),
            (A, 10 * MIN),
            (S, 2 * MIN),
            (A, 10 * MIN),
            (W, 4 * MIN),
        )

        result = stats(spans).transitions

        counts = {(t.previous, t.current): t.count for t in result.counts}
        assert counts[(A, S)] == 1 and counts[(S, A)] == 1 and counts[(A, W)] == 1
        assert result.n == 4
        dwell = {d.state: d for d in result.dwell}
        assert dwell[A].n == 2 and dwell[A].mean_s == 600
        assert dwell[W].n == 0 and dwell[W].n_censored == 2
