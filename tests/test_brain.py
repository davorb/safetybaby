"""Scripted fact sequences through observe -> situation -> rules -> gate."""

from __future__ import annotations

from datetime import timedelta

from babymon.domain import (
    AlertKind,
    BodyPosition,
    CameraChanged,
    CameraStatus,
    Controls,
    ControlsChanged,
    DeliveryResult,
    Eyes,
    HeadDirection,
    Priority,
    SetControls,
    SleepChanged,
    SleepState,
    Tick,
    VisionFailed,
    VisionObserved,
    VisionReason,
)
from tests.conftest import T0, Run, at, observation, seen, start, times

STOMACH = observation(body=BodyPosition.STOMACH)
COVERED = observation(face_covered=True)
LEFT = observation(head=HeadDirection.LEFT)
AWAKE_EARLIER = (SleepState.AWAKE, T0 - timedelta(minutes=1))
ASLEEP_EARLIER = (SleepState.ASLEEP, T0 - timedelta(minutes=1))


# ---------------------------------------------------------------- position rules


def test_stomach_on_two_checks_is_an_emergency(run: Run):
    run.feed(seen(0, STOMACH), seen(15, STOMACH), Tick(at(31)))
    assert run.alerts() == [(at(31), AlertKind.ON_STOMACH, Priority.EMERGENCY)]


def test_stomach_on_one_check_is_not_enough(run: Run):
    run.feed(seen(0, STOMACH), Tick(at(31)))
    assert run.alerts() == []


def test_face_covered_confident_then_low_confidence_does_not_alert(run: Run):
    run.feed(seen(0, COVERED), seen(15, observation(face_covered=True, confidence=0.3)))
    run.feed(Tick(at(31)))
    assert run.alerts() == []
    assert run.situation().face_covered_streak == 1


def test_low_confidence_leaves_streaks_unchanged_then_confident_extends(run: Run):
    run.feed(seen(0, STOMACH), seen(15, observation(body=BodyPosition.BACK, confidence=0.2)))
    assert run.feed(Tick(at(16))).situation().stomach_streak == 1
    run.feed(seen(30, STOMACH), Tick(at(31)))
    assert run.alerts() == [(at(31), AlertKind.ON_STOMACH, Priority.EMERGENCY)]


def test_safe_reading_resets_the_streak(run: Run):
    run.feed(seen(0, STOMACH), seen(15, observation()), seen(30, STOMACH), Tick(at(31)))
    assert run.alerts() == []
    assert run.situation().stomach_streak == 1


def test_out_of_crib_resets_every_streak(run: Run):
    run.feed(seen(0, STOMACH), seen(0, COVERED), seen(10, LEFT))
    run.feed(seen(20, observation(body=BodyPosition.STOMACH, in_crib=False)), Tick(at(21)))
    s = run.situation()
    assert (s.stomach_streak, s.face_covered_streak, s.head_left_for_s) == (0, 0, 0.0)


def test_failed_check_leaves_streaks_unchanged(run: Run):
    run.feed(seen(0, STOMACH), VisionFailed(at(15), VisionReason.CONFIRM, "timeout"))
    run.feed(seen(30, STOMACH), Tick(at(31)))
    assert run.alerts() == [(at(31), AlertKind.ON_STOMACH, Priority.EMERGENCY)]


def test_head_left_over_two_minutes_is_high(run: Run):
    run.feed(seen(10, LEFT), seen(70, LEFT)).seconds(71, 131)
    assert run.alerts() == [(at(131), AlertKind.HEAD_LEFT, Priority.HIGH)]


def test_head_turned_back_restarts_the_clock(run: Run):
    run.feed(seen(10, LEFT), seen(70, observation()), seen(100, LEFT)).seconds(101, 200)
    assert run.alerts() == []
    assert run.situation().head_left_for_s == 100.0


# ---------------------------------------------------------------- sleep rules


def test_crying_over_a_minute_is_high(run: Run):
    run.seconds(1, 70, cry=0.9)
    assert run.alerts() == [(at(65), AlertKind.CRYING, Priority.HIGH)]
    assert run.situation().sleep is SleepState.CRYING


def test_awake_during_nap_is_normal():
    run = Run(start(controls=Controls(nap_mode=True), sleep=AWAKE_EARLIER)).seconds(1, 1)
    assert run.alerts() == [(at(1), AlertKind.AWAKE_DURING_NAP, Priority.NORMAL)]


def test_awake_with_nap_off_is_silent():
    run = Run(start(sleep=AWAKE_EARLIER)).seconds(1, 5)
    assert run.alerts() == []


def test_crying_during_nap_is_not_also_awake_during_nap():
    run = Run(start(controls=Controls(nap_mode=True), sleep=ASLEEP_EARLIER))
    run.seconds(1, 70, cry=0.9)
    assert [kind for _, kind, _ in run.alerts()] == [AlertKind.CRYING]


def test_falling_asleep_records_the_change_and_sends_nothing(run: Run):
    run.seconds(1, 121)
    assert run.records() == [
        SleepChanged(at=at(121), previous=SleepState.STIRRING, current=SleepState.ASLEEP)
    ]
    assert run.alerts() == []


def test_open_eyes_wake_a_sleeping_baby():
    run = Run(start(sleep=ASLEEP_EARLIER)).feed(seen(1, observation(eyes=Eyes.OPEN)))
    run.seconds(1, 31)
    assert run.situation().sleep is SleepState.AWAKE
    assert SleepChanged(at=at(31), previous=SleepState.ASLEEP, current=SleepState.AWAKE) in (
        run.records()
    )


def test_a_stale_observation_no_longer_counts():
    run = Run(start(sleep=ASLEEP_EARLIER)).feed(seen(1, observation(eyes=Eyes.OPEN)))
    run.seconds(100, 140)
    assert run.situation().sleep is SleepState.ASLEEP


def test_a_single_cry_sample_does_not_keep_crying():
    run = Run(start(sleep=ASLEEP_EARLIER))
    run.seconds(1, 1, cry=0.9).seconds(2, 10)
    assert run.situation().sleep is SleepState.ASLEEP


def test_motion_then_quiet_stirs_and_settles():
    run = Run(start(sleep=ASLEEP_EARLIER)).seconds(1, 4, motion=0.05)
    assert run.situation().sleep is SleepState.STIRRING
    run.seconds(5, 10)
    assert run.situation().sleep is SleepState.STIRRING


def test_tick_idempotent():
    run = Run(start(sleep=ASLEEP_EARLIER)).seconds(1, 2, motion=0.05)
    before, effects_before = run.situation(), len(run.effects)
    run.feed(Tick(at(2)))
    assert (run.situation(), len(run.effects)) == (before, effects_before)
    run.seconds(3, 4, motion=0.05)
    assert run.situation().sleep is SleepState.STIRRING


# ---------------------------------------------------------------- cooldown and arming


def test_persisting_condition_realerts_exactly_at_cooldown(run: Run):
    run.feed(seen(0, STOMACH), seen(15, STOMACH)).seconds(31, 631)
    assert times(run.alerts()) == [31.0, 631.0]


def test_restored_delivery_counts_toward_cooldown():
    run = Run(start(delivered={AlertKind.ON_STOMACH: T0 - timedelta(seconds=500)}))
    run.feed(seen(0, STOMACH), seen(15, STOMACH)).seconds(31, 101)
    assert times(run.alerts()) == [100.0]


def test_disarmed_means_no_notifications():
    run = Run(start(controls=Controls(armed=False)))
    run.feed(seen(0, STOMACH), seen(15, STOMACH)).seconds(31, 40)
    assert run.alerts() == []
    run.feed(SetControls(at(41), armed=True)).seconds(42, 42)
    assert run.alerts() == [(at(42), AlertKind.ON_STOMACH, Priority.EMERGENCY)]


def test_failed_delivery_refires_on_the_next_tick(run: Run):
    run.feed(seen(0, STOMACH), seen(15, STOMACH)).seconds(31, 31)
    run.feed(DeliveryResult(at(32), AlertKind.ON_STOMACH, delivered=False)).seconds(33, 33)
    assert times(run.alerts()) == [31.0, 33.0]


def test_successful_delivery_keeps_the_cooldown(run: Run):
    run.feed(seen(0, STOMACH), seen(15, STOMACH)).seconds(31, 31)
    run.feed(DeliveryResult(at(32), AlertKind.ON_STOMACH, delivered=True)).seconds(33, 40)
    assert times(run.alerts()) == [31.0]


def test_without_push_an_undeliverable_alert_waits_out_the_cooldown():
    run = Run(start(push=False)).feed(seen(0, STOMACH), seen(15, STOMACH)).seconds(31, 31)
    run.feed(DeliveryResult(at(32), AlertKind.ON_STOMACH, delivered=False)).seconds(33, 40)
    assert times(run.alerts()) == [31.0]


# ---------------------------------------------------------------- camera and controls


def test_camera_never_connecting_alerts_after_threshold():
    run = Run(start(camera_up=False)).seconds(1, 61)
    assert run.alerts() == [(at(61), AlertKind.CAMERA_OFFLINE, Priority.HIGH)]


def test_camera_status_records_changes_only():
    run = Run(start(camera_up=False))
    run.feed(CameraStatus(at(1), True, "ok"), CameraStatus(at(2), True, "ok"))
    run.feed(CameraStatus(at(3), False, "stall"))
    assert run.records() == [
        CameraChanged(at=at(1), up=True, detail="ok"),
        CameraChanged(at=at(3), up=False, detail="stall"),
    ]


def test_controls_patch_records_once_and_keeps_other_switch(run: Run):
    run.feed(SetControls(at(1), nap_mode=True), SetControls(at(2), nap_mode=True))
    run.feed(SetControls(at(3), armed=False))
    assert run.records() == [
        ControlsChanged(at=at(1), controls=Controls(armed=True, nap_mode=True)),
        ControlsChanged(at=at(3), controls=Controls(armed=False, nap_mode=True)),
    ]


def test_vision_done_is_recorded(run: Run):
    fact = seen(5, STOMACH, VisionReason.CONFIRM)
    run.feed(fact)
    assert run.records() == [
        VisionObserved(
            at=at(5), reason=VisionReason.CONFIRM, observation=STOMACH, snapshot=fact.snapshot
        )
    ]


# ---------------------------------------------------------------- vision scheduling


def test_periodic_checks_every_minute(run: Run):
    run.seconds(1, 1).feed(VisionFailed(at(2), VisionReason.PERIODIC, "x")).seconds(2, 61)
    run.feed(VisionFailed(at(62), VisionReason.PERIODIC, "x")).seconds(62, 121)
    assert run.requests() == [
        (at(1), VisionReason.PERIODIC),
        (at(61), VisionReason.PERIODIC),
        (at(121), VisionReason.PERIODIC),
    ]


def test_confirm_checks_every_15s_while_a_streak_is_pending(run: Run):
    run.seconds(1, 1).feed(seen(2, STOMACH)).seconds(2, 16)
    assert run.requests() == [(at(1), VisionReason.PERIODIC), (at(16), VisionReason.CONFIRM)]


def test_confirmed_danger_returns_to_the_periodic_cadence(run: Run):
    run.seconds(1, 1).feed(seen(2, STOMACH)).seconds(2, 16).feed(seen(17, STOMACH))
    run.seconds(17, 76)
    assert run.requests()[2:] == [(at(76), VisionReason.PERIODIC)]


def test_settled_after_motion_requests_a_check(run: Run):
    run.seconds(1, 1).feed(VisionFailed(at(2), VisionReason.PERIODIC, "x"))
    run.seconds(2, 12, motion=0.05).seconds(13, 13)
    assert run.requests() == [
        (at(1), VisionReason.PERIODIC),
        (at(13), VisionReason.SETTLED_AFTER_MOTION),
    ]


def test_settled_check_respects_minimum_spacing(run: Run):
    run.seconds(1, 1).feed(VisionFailed(at(2), VisionReason.PERIODIC, "x"))
    run.seconds(2, 5, motion=0.05).seconds(6, 8)
    assert run.requests() == [(at(1), VisionReason.PERIODIC)]


def test_lost_request_does_not_block_vision_forever(run: Run):
    run.seconds(1, 61)
    assert run.requests() == [(at(1), VisionReason.PERIODIC), (at(61), VisionReason.PERIODIC)]


def test_in_flight_request_blocks_new_ones(run: Run):
    run.seconds(1, 1).feed(seen(2, STOMACH)).seconds(2, 16).seconds(17, 30)
    assert times(run.requests()) == [1.0, 16.0]


def test_vision_disabled_requests_nothing():
    run = Run(start(vision=False)).seconds(1, 120)
    assert run.requests() == []


# ---------------------------------------------------------------- situation


def test_situation_reports_durations_for_the_decider(run: Run):
    run.feed(seen(5, LEFT)).seconds(1, 65)
    s = run.situation()
    assert (s.at, s.sleep, s.sleep_for_s) == (at(65), SleepState.STIRRING, 65.0)
    assert (s.observation, s.observation_age_s, s.head_left_for_s) == (LEFT, 60.0, 60.0)
    assert (s.camera_up, s.camera_down_for_s) == (True, 0.0)
    assert s.capabilities.vision and s.capabilities.push


def test_situation_tracks_camera_downtime():
    run = Run(start(camera_up=False)).seconds(1, 30)
    assert run.situation().camera_down_for_s == 30.0
