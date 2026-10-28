from __future__ import annotations

import multiprocessing
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from babymon.domain import (
    Alert,
    AlertKind,
    AlertSent,
    CameraChanged,
    Controls,
    ControlsChanged,
    Priority,
    SleepChanged,
    SleepState,
    SnapshotRef,
    Still,
    VisionObserved,
    VisionReason,
)
from babymon.store import (
    EVENT_RETENTION,
    RETENTION,
    InstanceLockError,
    Store,
    acquire_instance_lock,
)
from tests.conftest import T0, at, observation

H = timedelta(hours=1)
NOW = T0 + 30 * H


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store.open(tmp_path)


def segment_name(t: datetime) -> str:
    return t.strftime("%Y%m%d-%H%M%S") + ".mp4"


def make_segments(store: Store, starts: list[datetime]) -> None:
    """Each file's mtime is the next segment's start (or NOW for the last), as ffmpeg leaves it."""
    ends = [*starts[1:], NOW]
    for start, end in zip(starts, ends, strict=True):
        path = store.segment_dir / segment_name(start)
        path.write_bytes(b"mp4")
        os.utime(path, (end.timestamp(), end.timestamp()))


def make_snapshots(store: Store, times: list[datetime]) -> list[SnapshotRef]:
    return [store.save_snapshot(Still(b"jpg", t), "vision") for t in times]


def alert_sent(
    t: datetime, kind: AlertKind = AlertKind.ON_STOMACH, delivered: bool = True
) -> AlertSent:
    alert = Alert(kind=kind, priority=Priority.EMERGENCY, at=t, title="t", message="m")
    return AlertSent(at=t, alert=alert, snapshot=None, delivered=delivered)


def sleep_changed(t: datetime, current: SleepState, previous: SleepState) -> SleepChanged:
    return SleepChanged(at=t, previous=previous, current=current)


class TestSweep:
    def test_deletes_exactly_what_is_older_than_the_horizon(self, store: Store):
        horizon = NOW - RETENTION
        hourly = [NOW - n * H for n in range(30, 0, -1)]
        make_segments(store, hourly)
        old_snap, fresh_snap = make_snapshots(store, [horizon - 1 * H, horizon + 1 * H])
        store.append(sleep_changed(horizon - 1 * H, SleepState.ASLEEP, SleepState.STIRRING))
        store.append(sleep_changed(horizon + 1 * H, SleepState.STIRRING, SleepState.ASLEEP))

        report = store.sweep(NOW)

        remaining = sorted(p.name for p in store.segment_dir.iterdir())
        assert remaining == [segment_name(t) for t in hourly if t >= horizon]
        assert not store.snapshot_path(old_snap).exists()
        assert store.snapshot_path(fresh_snap).exists()
        assert (report.segments, report.snapshots, report.events) == (6, 1, 0)

    def test_events_outlive_media_and_go_after_ninety_days(self, store: Store):
        event_horizon = NOW - EVENT_RETENTION
        old, kept = event_horizon - 1 * H, event_horizon + 1 * H
        day_old = NOW - RETENTION - 1 * H
        for t in (old, kept, day_old):
            store.append(sleep_changed(t, SleepState.ASLEEP, SleepState.STIRRING))

        report = store.sweep(NOW)

        assert report.events == 1
        assert [e.event.at for e in store.events(old - H, NOW)] == [day_old, kept]

    def test_segment_covering_the_horizon_survives(self, store: Store):
        horizon = NOW - RETENTION
        covering = horizon - timedelta(minutes=2)
        make_segments(store, [covering - 5 * H, covering - 1 * H, covering, horizon + 3 * H])

        store.sweep(NOW)

        assert sorted(p.name for p in store.segment_dir.iterdir()) == [
            segment_name(covering),
            segment_name(horizon + 3 * H),
        ]

    def test_second_sweep_deletes_nothing(self, store: Store):
        make_segments(store, [NOW - n * H for n in range(30, 0, -1)])
        make_snapshots(store, [NOW - 26 * H, NOW - 1 * H])
        store.append(sleep_changed(NOW - 26 * H, SleepState.ASLEEP, SleepState.STIRRING))
        store.sweep(NOW)

        report = store.sweep(NOW)

        assert (report.segments, report.snapshots, report.events) == (0, 0, 0)

    def test_newest_segment_is_never_swept(self, store: Store):
        make_segments(store, [NOW - 48 * H])

        assert store.sweep(NOW).segments == 0

    def test_ignores_files_that_are_not_ours(self, store: Store):
        (store.segment_dir / "notes.txt").write_text("x")
        (store.segment_dir / segment_name(NOW - 50 * H)).write_bytes(b"x")
        (store.segment_dir / segment_name(NOW - 49 * H)).write_bytes(b"x")

        store.sweep(NOW)

        assert (store.segment_dir / "notes.txt").exists()


class TestSnapshots:
    def test_name_derives_from_capture_time_and_label(self, store: Store):
        ref = store.save_snapshot(
            Still(b"abc", datetime(2026, 10, 27, 2, 3, 4, tzinfo=UTC)), "vision"
        )

        assert ref == "20261027-020304-vision.jpg"
        assert store.snapshot_path(ref).read_bytes() == b"abc"

    def test_same_still_twice_is_one_file(self, store: Store):
        still = Still(b"abc", T0)
        a = store.save_snapshot(still, "vision")
        b = store.save_snapshot(still, "vision")

        assert a == b
        assert len(list(store.snapshot_path(a).parent.iterdir())) == 1

    def test_alert_kind_works_as_label(self, store: Store):
        ref = store.save_snapshot(Still(b"abc", T0), AlertKind.ON_STOMACH)

        assert ref.endswith("-on_stomach.jpg")

    def test_label_cannot_escape_the_directory(self, store: Store):
        with pytest.raises(ValueError):
            store.save_snapshot(Still(b"abc", T0), "../evil")

    @pytest.mark.parametrize("name", ["../x.jpg", "a/b.jpg", "/etc/passwd", "", ".hidden", ".."])
    def test_snapshot_path_rejects_path_tricks(self, store: Store, name: str):
        with pytest.raises(ValueError):
            store.snapshot_path(SnapshotRef(name))

    def test_segment_path_rejects_path_tricks(self, store: Store):
        with pytest.raises(ValueError):
            store.segment_path("../babymon.sqlite3")
        assert (
            store.segment_path("20261027-020000.mp4") == store.segment_dir / "20261027-020000.mp4"
        )


class TestReference:
    def test_none_until_set_then_replaced(self, store: Store):
        assert store.reference() is None

        store.set_reference(Still(b"one", T0))
        store.set_reference(Still(b"two", T0))

        ref = store.reference()
        assert ref is not None and ref.jpeg == b"two"
        assert not list(store.segment_dir.parent.glob("*.tmp"))


class TestRestore:
    def test_empty_store_gives_defaults(self, store: Store):
        restored = store.restore(NOW, 600)

        assert (restored.controls, restored.sleep, dict(restored.delivered)) == (
            Controls(),
            None,
            {},
        )

    def test_controls_round_trip_through_the_row(self, store: Store):
        store.append(ControlsChanged(at=at(1), controls=Controls(armed=True, nap_mode=True)))
        store.append(ControlsChanged(at=at(2), controls=Controls(armed=False, nap_mode=True)))

        assert store.restore(NOW, 600).controls == Controls(armed=False, nap_mode=True)

    def test_controls_survive_sweeping_their_events(self, store: Store):
        store.append(ControlsChanged(at=T0, controls=Controls(armed=False)))
        later = T0 + EVENT_RETENTION + 2 * H
        store.sweep(later)

        assert store.events(T0 - H, later) == []
        assert store.restore(later, 600).controls == Controls(armed=False)

    def test_last_delivered_alert_per_kind_within_cooldown(self, store: Store):
        now = at(1000)
        store.append(alert_sent(at(100), AlertKind.CRYING))
        store.append(alert_sent(at(500), AlertKind.CRYING))
        store.append(alert_sent(at(600), AlertKind.ON_STOMACH))
        store.append(alert_sent(at(700), AlertKind.ON_STOMACH, delivered=False))
        store.append(alert_sent(at(800), AlertKind.FACE_COVERED, delivered=False))
        store.append(alert_sent(at(300), AlertKind.HEAD_LEFT))

        restored = store.restore(now, 600)

        assert dict(restored.delivered) == {
            AlertKind.CRYING: at(500),
            AlertKind.ON_STOMACH: at(600),
        }

    def test_alert_exactly_at_the_cooldown_edge_still_counts(self, store: Store):
        store.append(alert_sent(at(400), AlertKind.CRYING))

        assert AlertKind.CRYING in store.restore(at(1000), 600).delivered
        assert AlertKind.CRYING not in store.restore(at(1001), 600).delivered

    def test_sleep_resumes_only_under_ten_minutes(self, store: Store):
        store.append(sleep_changed(at(0), SleepState.ASLEEP, SleepState.STIRRING))
        store.append(sleep_changed(at(100), SleepState.AWAKE, SleepState.ASLEEP))

        assert store.restore(at(100 + 599), 600).sleep == (SleepState.AWAKE, at(100))
        assert store.restore(at(100 + 601), 600).sleep is None

    def test_other_event_types_do_not_disturb_restore(self, store: Store):
        store.append(CameraChanged(at=at(5), up=False, detail="x"))
        store.append(
            VisionObserved(
                at=at(6),
                reason=VisionReason.PERIODIC,
                observation=observation(),
                snapshot=SnapshotRef("a.jpg"),
            )
        )

        restored = store.restore(at(10), 600)

        assert restored.sleep is None and not restored.delivered


class TestEvents:
    def test_round_trip_newest_first_with_limit(self, store: Store):
        events = [
            CameraChanged(at=at(1), up=True, detail="a"),
            sleep_changed(at(2), SleepState.ASLEEP, SleepState.STIRRING),
            alert_sent(at(3)),
            VisionObserved(
                at=at(4),
                reason=VisionReason.CONFIRM,
                observation=observation(),
                snapshot=SnapshotRef("s.jpg"),
            ),
        ]
        for e in events:
            store.append(e)

        stored = store.events(at(0), at(10))
        assert [s.event for s in stored] == events[::-1]
        assert [s.event for s in store.events(at(2), at(3))] == [events[2], events[1]]
        assert [s.event for s in store.events(at(0), at(10), limit=2)] == events[:1:-1]
        assert len({s.id for s in stored}) == 4


class TestSleepTimeline:
    def test_spans_are_contiguous_and_cover_the_window(self, store: Store):
        store.append(sleep_changed(at(100), SleepState.ASLEEP, SleepState.STIRRING))
        store.append(sleep_changed(at(300), SleepState.CRYING, SleepState.ASLEEP))
        store.append(sleep_changed(at(900), SleepState.AWAKE, SleepState.CRYING))

        spans = store.sleep_timeline(at(0), at(1200))

        assert [(s.state, s.start, s.end) for s in spans] == [
            (None, at(0), at(100)),
            (SleepState.ASLEEP, at(100), at(300)),
            (SleepState.CRYING, at(300), at(900)),
            (SleepState.AWAKE, at(900), at(1200)),
        ]

    def test_window_starting_mid_state_takes_the_state_from_before_it(self, store: Store):
        store.append(sleep_changed(at(100), SleepState.ASLEEP, SleepState.STIRRING))
        store.append(sleep_changed(at(500), SleepState.STIRRING, SleepState.ASLEEP))

        spans = store.sleep_timeline(at(200), at(400))

        assert [(s.state, s.start, s.end) for s in spans] == [(SleepState.ASLEEP, at(200), at(400))]

    def test_change_exactly_at_since_leaves_no_empty_span(self, store: Store):
        store.append(sleep_changed(at(100), SleepState.ASLEEP, SleepState.STIRRING))

        spans = store.sleep_timeline(at(100), at(200))

        assert [(s.state, s.start, s.end) for s in spans] == [(SleepState.ASLEEP, at(100), at(200))]

    def test_no_rows_is_one_unknown_span_and_empty_window_is_empty(self, store: Store):
        assert [(s.state, s.start, s.end) for s in store.sleep_timeline(at(0), at(50))] == [
            (None, at(0), at(50))
        ]
        assert store.sleep_timeline(at(50), at(50)) == []


class TestClipAt:
    def test_picks_the_segment_and_offset(self, store: Store):
        starts = [NOW - 3 * H, NOW - 2 * H, NOW - 1 * H]
        make_segments(store, starts)

        clip = store.clip_at(NOW - 2 * H + timedelta(seconds=75))

        assert clip is not None
        assert (clip.name, clip.start, clip.offset_s) == (segment_name(starts[1]), starts[1], 75.0)
        assert clip.growing is False

    def test_instant_exactly_at_a_start_belongs_to_that_segment(self, store: Store):
        starts = [NOW - 2 * H, NOW - 1 * H]
        make_segments(store, starts)

        clip = store.clip_at(starts[1])

        assert clip is not None and clip.name == segment_name(starts[1]) and clip.offset_s == 0.0

    def test_newest_recently_written_segment_is_growing(self, store: Store):
        now = datetime.now(UTC)
        path = store.segment_dir / segment_name(now - timedelta(minutes=2))
        path.write_bytes(b"x")

        clip = store.clip_at(now - timedelta(minutes=1))

        assert clip is not None and clip.growing is True

    def test_newest_stale_segment_is_not_growing(self, store: Store):
        now = datetime.now(UTC)
        start = now - timedelta(hours=1)
        path = store.segment_dir / segment_name(start)
        path.write_bytes(b"x")
        last_write = (now - timedelta(minutes=5)).timestamp()
        os.utime(path, (last_write, last_write))

        clip = store.clip_at(start + timedelta(minutes=10))

        assert clip is not None and clip.growing is False

    def test_before_the_first_segment_and_in_a_recording_gap_have_no_clip(self, store: Store):
        first, second = NOW - 3 * H, NOW - 1 * H
        path = store.segment_dir / segment_name(first)
        path.write_bytes(b"x")
        end = (first + timedelta(minutes=5)).timestamp()
        os.utime(path, (end, end))
        (store.segment_dir / segment_name(second)).write_bytes(b"x")

        assert store.clip_at(first - timedelta(seconds=1)) is None
        assert store.clip_at(first + timedelta(hours=1)) is None

    def test_no_segments(self, store: Store):
        assert store.clip_at(NOW) is None


def _hold_lock(path: str, ready, release) -> None:
    lock = acquire_instance_lock(Path(path))
    ready.set()
    release.wait(10)
    lock.release()


class TestInstanceLock:
    def test_second_holder_is_refused_until_the_first_lets_go(self, tmp_path: Path):
        ctx = multiprocessing.get_context("spawn")
        ready, release = ctx.Event(), ctx.Event()
        holder = ctx.Process(target=_hold_lock, args=(str(tmp_path), ready, release))
        holder.start()
        try:
            assert ready.wait(20)
            with pytest.raises(InstanceLockError):
                acquire_instance_lock(tmp_path)
        finally:
            release.set()
            holder.join(10)
        with acquire_instance_lock(tmp_path):
            pass

    def test_killed_holder_leaves_no_stale_lock(self, tmp_path: Path):
        ctx = multiprocessing.get_context("spawn")
        ready, release = ctx.Event(), ctx.Event()
        holder = ctx.Process(target=_hold_lock, args=(str(tmp_path), ready, release))
        holder.start()
        assert ready.wait(20)
        holder.kill()
        holder.join(10)

        acquire_instance_lock(tmp_path).release()

    def test_creates_the_data_directory(self, tmp_path: Path):
        with acquire_instance_lock(tmp_path / "new" / "dir"):
            assert (tmp_path / "new" / "dir" / "babymon.lock").exists()
