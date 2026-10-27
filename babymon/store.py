"""Everything on disk, behind one object. The only module that knows the layout:

    <data_dir>/babymon.sqlite3     events (24 h), controls (one row, never pruned)
    <data_dir>/snapshots/          <UTC %Y%m%d-%H%M%S>-<label>.jpg   (24 h)
    <data_dir>/segments/           <UTC %Y%m%d-%H%M%S>.mp4  written by ffmpeg (24 h)
    <data_dir>/reference.jpg       the "good position" image, never pruned

File names carry their own timestamps, so the directory listing is the index: no
table of segments or snapshots to keep in sync, and retention after a crash is just
"delete what is older than the horizon" again.

Thread safety: methods are synchronous and open a short-lived sqlite3 connection
(WAL mode). Async callers use asyncio.to_thread; FastAPI sync routes run in its
thread pool. SQLite serialises the few concurrent appends.
"""

from __future__ import annotations

import fcntl
import os
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Self

from pydantic import TypeAdapter

from babymon.domain import (
    AlertKind,
    AlertSent,
    Controls,
    ControlsChanged,
    Event,
    Instant,
    Restored,
    SleepChanged,
    SleepState,
    SnapshotRef,
    Still,
    VisionObserved,
)

RETENTION = timedelta(hours=24)
SLEEP_RESUME_WINDOW = timedelta(minutes=10)
GROWING_IF_WRITTEN_WITHIN = timedelta(seconds=30)
CLIP_END_SLACK = timedelta(seconds=10)
"""A fragmented segment's mtime trails its last frame by up to one fragment."""

TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"
"""UTC. ffmpeg's strftime must run with TZ=UTC so segment names match."""

_SEGMENT_NAME = re.compile(r"^(\d{8}-\d{6})\.mp4$")
_SNAPSHOT_NAME = re.compile(r"^(\d{8}-\d{6})-[a-z0-9_]+\.jpg$")
_LABEL = re.compile(r"^[a-z0-9_]+$")
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_EVENT = TypeAdapter(Event)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    at_us INTEGER NOT NULL,
    type TEXT NOT NULL,
    payload TEXT NOT NULL,
    snapshot TEXT
);
CREATE INDEX IF NOT EXISTS events_at ON events (at_us);
CREATE INDEX IF NOT EXISTS events_type_at ON events (type, at_us);
CREATE TABLE IF NOT EXISTS controls (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    payload TEXT NOT NULL
);
"""


class InstanceLockError(Exception):
    """Another babymon process already holds the data directory."""


class InstanceLock:
    def __init__(self, handle: int) -> None:
        self._handle: int | None = handle

    def release(self) -> None:
        if self._handle is not None:
            os.close(self._handle)
            self._handle = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


def acquire_instance_lock(data_dir: Path) -> InstanceLock:
    """flock <data_dir>/babymon.lock. The kernel drops it when the process dies, so a
    crash never leaves a stale lock. Keep the returned object alive for the process."""
    data_dir.mkdir(parents=True, exist_ok=True)
    handle = os.open(data_dir / "babymon.lock", os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(handle)
        raise InstanceLockError(
            f"another babymon process holds {data_dir / 'babymon.lock'}"
        ) from None
    os.ftruncate(handle, 0)
    os.write(handle, f"{os.getpid()}\n".encode())
    return InstanceLock(handle)


def _us(at: Instant) -> int:
    return (at - _EPOCH) // timedelta(microseconds=1)


def _from_us(us: int) -> Instant:
    return _EPOCH + timedelta(microseconds=us)


def _stamp(at: Instant) -> str:
    return at.astimezone(UTC).strftime(TIMESTAMP_FORMAT)


def _parse_stamp(text: str) -> Instant:
    return datetime.strptime(text, TIMESTAMP_FORMAT).replace(tzinfo=UTC)


def _plain_name(name: str) -> str:
    if not name or name != Path(name).name or name.startswith("."):
        raise ValueError(f"not a plain file name: {name!r}")
    return name


def _write_atomically(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


@dataclass(frozen=True, slots=True)
class StoredEvent:
    id: int
    event: Event


@dataclass(frozen=True, slots=True)
class SleepSpan:
    state: SleepState | None
    """None for the part of the window before the first known state."""
    start: Instant
    end: Instant


@dataclass(frozen=True, slots=True)
class Clip:
    name: str
    start: Instant
    offset_s: float
    """Seconds from the segment start to the requested instant; the dashboard seeks here."""
    growing: bool
    """True for the segment ffmpeg is still writing."""


@dataclass(frozen=True, slots=True)
class SweepReport:
    segments: int
    snapshots: int
    events: int


class Store:
    def __init__(self, data_dir: Path) -> None:
        self._dir = data_dir
        self._db = data_dir / "babymon.sqlite3"
        self._snapshots = data_dir / "snapshots"
        self._segments = data_dir / "segments"

    @classmethod
    def open(cls, data_dir: Path) -> Store:
        """Create directories and schema if missing. Safe to call on every start."""
        store = cls(data_dir)
        store._snapshots.mkdir(parents=True, exist_ok=True)
        store._segments.mkdir(parents=True, exist_ok=True)
        with closing(store._connect()) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)
        return store

    @property
    def segment_dir(self) -> Path:
        """Handed to camera.ffmpeg_argv. The store owns the name pattern both sides use."""
        return self._segments

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db, timeout=10.0)

    # -- writes

    def append(self, event: Event) -> None:
        """Insert (at, type, payload JSON, snapshot). A ControlsChanged event also
        upserts the controls row in the same transaction, so controls cannot drift
        from the log."""
        snapshot = event.snapshot if isinstance(event, VisionObserved | AlertSent) else None
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO events (at_us, type, payload, snapshot) VALUES (?, ?, ?, ?)",
                (_us(event.at), event.type, event.model_dump_json(), snapshot),
            )
            if isinstance(event, ControlsChanged):
                conn.execute(
                    "INSERT INTO controls (id, payload) VALUES (1, ?) "
                    "ON CONFLICT (id) DO UPDATE SET payload = excluded.payload",
                    (event.controls.model_dump_json(),),
                )

    def save_snapshot(self, still: Still, label: str) -> SnapshotRef:
        """Write tmp file, rename into place. Name derives from still.at and label, so
        saving the same still twice overwrites instead of duplicating."""
        label = str(label)
        if not _LABEL.fullmatch(label):
            raise ValueError(f"snapshot label must match [a-z0-9_]+: {label!r}")
        name = f"{_stamp(still.at)}-{label}.jpg"
        _write_atomically(self._snapshots / name, still.jpeg)
        return SnapshotRef(name)

    def set_reference(self, still: Still) -> None:
        """Atomic replace of reference.jpg."""
        _write_atomically(self._dir / "reference.jpg", still.jpeg)

    def sweep(self, now: Instant) -> SweepReport:
        """Delete segments, snapshots and event rows older than now - RETENTION.
        Segments are judged by the *next* segment's start, so the file covering
        the horizon survives. Converges no matter how many times it runs."""
        horizon = now - RETENTION
        segments = self._segment_listing()
        expired_segments = [
            current
            for current, following in zip(segments, segments[1:], strict=False)
            if following[0] <= horizon
        ]
        expired_snapshots = [
            path
            for path in self._snapshots.iterdir()
            if (m := _SNAPSHOT_NAME.match(path.name)) and _parse_stamp(m[1]) < horizon
        ]
        for _, path in expired_segments:
            path.unlink(missing_ok=True)
        for path in expired_snapshots:
            path.unlink(missing_ok=True)
        with closing(self._connect()) as conn, conn:
            events = conn.execute("DELETE FROM events WHERE at_us < ?", (_us(horizon),)).rowcount
        return SweepReport(len(expired_segments), len(expired_snapshots), events)

    # -- reads

    def restore(self, now: Instant, cooldown_s: float) -> Restored:
        """Controls row (defaults if none), last SleepChanged if within 10 min, and the
        last delivered AlertSent per kind within cooldown_s."""
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT payload FROM controls WHERE id = 1").fetchone()
            sleep_row = conn.execute(
                "SELECT payload FROM events WHERE type = 'sleep_changed' AND at_us >= ? "
                "ORDER BY at_us DESC, id DESC LIMIT 1",
                (_us(now - SLEEP_RESUME_WINDOW),),
            ).fetchone()
            alert_rows = conn.execute(
                "SELECT payload FROM events WHERE type = 'alert' AND at_us >= ? ORDER BY at_us, id",
                (_us(now - timedelta(seconds=cooldown_s)),),
            ).fetchall()
        sleep: tuple[SleepState, Instant] | None = None
        if sleep_row:
            changed = _decode(sleep_row[0], SleepChanged)
            sleep = (changed.current, changed.at)
        delivered: dict[AlertKind, Instant] = {}
        for (payload,) in alert_rows:
            sent = _decode(payload, AlertSent)
            if sent.delivered:
                delivered[sent.alert.kind] = sent.alert.at
        controls = Controls.model_validate_json(row[0]) if row else Controls()
        return Restored(controls, sleep, delivered)

    def reference(self) -> Still | None:
        path = self._dir / "reference.jpg"
        try:
            return Still(path.read_bytes(), _mtime(path))
        except FileNotFoundError:
            return None

    def events(self, since: Instant, until: Instant, limit: int = 500) -> list[StoredEvent]:
        """Newest first, at most `limit`, for since <= at <= until."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT id, payload FROM events WHERE at_us BETWEEN ? AND ? "
                "ORDER BY at_us DESC, id DESC LIMIT ?",
                (_us(since), _us(until), limit),
            ).fetchall()
        return [StoredEvent(id, _EVENT.validate_json(payload)) for id, payload in rows]

    def sleep_timeline(self, since: Instant, until: Instant) -> list[SleepSpan]:
        """Derived from SleepChanged rows; contiguous spans covering [since, until]."""
        if until <= since:
            return []
        with closing(self._connect()) as conn:
            before = conn.execute(
                "SELECT payload FROM events WHERE type = 'sleep_changed' AND at_us < ? "
                "ORDER BY at_us DESC, id DESC LIMIT 1",
                (_us(since),),
            ).fetchone()
            inside = conn.execute(
                "SELECT payload FROM events WHERE type = 'sleep_changed' "
                "AND at_us BETWEEN ? AND ? ORDER BY at_us, id",
                (_us(since), _us(until)),
            ).fetchall()
        state = _decode(before[0], SleepChanged).current if before else None
        spans: list[SleepSpan] = []
        cursor = since
        for (payload,) in inside:
            changed = _decode(payload, SleepChanged)
            if changed.at > cursor:
                spans.append(SleepSpan(state, cursor, changed.at))
                cursor = changed.at
            state = changed.current
        if until > cursor:
            spans.append(SleepSpan(state, cursor, until))
        return spans

    def clip_at(self, at: Instant) -> Clip | None:
        """Segment with the greatest start <= at, from the directory listing. A segment
        covers [start, last write]; an instant past that (camera was down) has no clip."""
        segments = self._segment_listing()
        covering = [(start, path) for start, path in segments if start <= at]
        if not covering:
            return None
        start, path = covering[-1]
        try:
            written = _mtime(path)
        except FileNotFoundError:
            return None
        if at > written + CLIP_END_SLACK:
            return None
        is_newest = path == segments[-1][1]
        growing = is_newest and datetime.now(UTC) - written <= GROWING_IF_WRITTEN_WITHIN
        return Clip(path.name, start, (at - start).total_seconds(), growing)

    def snapshot_path(self, ref: SnapshotRef) -> Path:
        """Resolve inside snapshots/ only; rejects names with path separators."""
        return self._snapshots / _plain_name(ref)

    def segment_path(self, name: str) -> Path:
        return self._segments / _plain_name(name)

    def _segment_listing(self) -> list[tuple[Instant, Path]]:
        found = [
            (_parse_stamp(m[1]), path)
            for path in self._segments.iterdir()
            if (m := _SEGMENT_NAME.match(path.name))
        ]
        return sorted(found)


def _decode[E: SleepChanged | AlertSent](payload: str, kind: type[E]) -> E:
    return kind.model_validate_json(payload)


def _mtime(path: Path) -> Instant:
    return datetime.fromtimestamp(path.stat().st_mtime, UTC)
