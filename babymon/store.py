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

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from babymon.brain import Restored
from babymon.domain import Controls, Event, Instant, SleepState, SnapshotRef, Still

RETENTION = timedelta(hours=24)


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
    @classmethod
    def open(cls, data_dir: Path) -> Store:
        """Create directories and schema if missing. Safe to call on every start."""
        raise NotImplementedError

    @property
    def segment_dir(self) -> Path:
        """Handed to camera.ffmpeg_argv. The store owns the name pattern both sides use."""
        raise NotImplementedError

    # -- writes

    def append(self, event: Event) -> None:
        """Insert (at, type, payload JSON, snapshot). A ControlsChanged event also
        upserts the controls row in the same transaction, so controls cannot drift
        from the log."""
        raise NotImplementedError

    def save_snapshot(self, still: Still, label: str) -> SnapshotRef:
        """Write tmp file, rename into place. Name derives from still.at and label, so
        saving the same still twice overwrites instead of duplicating."""
        raise NotImplementedError

    def set_reference(self, still: Still) -> None:
        """Atomic replace of reference.jpg."""
        raise NotImplementedError

    def sweep(self, now: Instant) -> SweepReport:
        """Delete segments, snapshots and event rows older than now - RETENTION.
        Segments are judged by the *next* segment's start, so the file covering
        the horizon survives. Converges no matter how many times it runs."""
        raise NotImplementedError

    # -- reads

    def restore(self, now: Instant, cooldown_s: float) -> Restored:
        """Controls row (defaults if none), last SleepChanged if within 10 min, and the
        last delivered AlertSent per kind within cooldown_s."""
        raise NotImplementedError

    def reference(self) -> Still | None:
        raise NotImplementedError

    def events(self, since: Instant, until: Instant, limit: int = 500) -> list[StoredEvent]:
        raise NotImplementedError

    def sleep_timeline(self, since: Instant, until: Instant) -> list[SleepSpan]:
        """Derived from SleepChanged rows; contiguous spans covering [since, until]."""
        raise NotImplementedError

    def clip_at(self, at: Instant) -> Clip | None:
        """Segment with the greatest start <= at, from the directory listing."""
        raise NotImplementedError

    def snapshot_path(self, ref: SnapshotRef) -> Path:
        """Resolve inside snapshots/ only; rejects names with path separators."""
        raise NotImplementedError

    def segment_path(self, name: str) -> Path:
        raise NotImplementedError
