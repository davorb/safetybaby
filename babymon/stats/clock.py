"""The one interval primitive: cut a time range into local-calendar cells.

Every time-weighted statistic (seconds per state per hour, per sleep day, night against
day) asks "how many seconds of this interval fall in which local bucket". Instead of each
metric doing its own midnight and DST arithmetic, the window is tiled once into 15-minute
UTC cells and each cell carries its local attributes. Every real UTC offset is a multiple
of 15 minutes, and every bucket boundary (hour, half hour, 07:00, 19:00) is a multiple of
15 local minutes, so a cell never straddles a boundary, a DST jump included. `pieces`
then yields (cell, seconds) for any interval.

A sleep day runs from local 07:00 to the next 07:00 and is labelled by its start date.
Its night is 19:00 to 07:00, so the night sits inside the day it began in.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta, tzinfo

from babymon.domain import Instant

CELL = timedelta(minutes=15)
SLEEP_DAY_START_MIN = 7 * 60
NIGHT_START_MIN = 19 * 60
NIGHT_END_MIN = 7 * 60
SLOT_MIN = 30
SLOTS_PER_DAY = 24 * 60 // SLOT_MIN
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class Cell:
    start: Instant
    hour: int
    """Local hour 0..23."""
    day: date
    """Label of the sleep day this cell belongs to."""
    slot: int
    """Half-hour index inside the sleep day, 0 = 07:00. A repeated DST hour shares slots."""
    night: bool


def local_minutes(at: Instant, tz: tzinfo) -> int:
    local = at.astimezone(tz)
    return local.hour * 60 + local.minute


class Calendar:
    def __init__(self, start: Instant, end: Instant, tz: tzinfo) -> None:
        first = _EPOCH + ((start - _EPOCH) // CELL) * CELL
        self._cells: list[Cell] = []
        t = first
        while t <= end:
            self._cells.append(_cell(t, tz))
            t += CELL
        self._starts = [c.start for c in self._cells]

    def cell_at(self, at: Instant) -> Cell:
        return self._cells[max(0, bisect_right(self._starts, at) - 1)]

    def pieces(self, start: Instant, end: Instant) -> Iterator[tuple[Cell, float]]:
        """(cell, seconds of [start, end) inside it), in time order."""
        if end <= start:
            return
        index = max(0, bisect_right(self._starts, start) - 1)
        while index < len(self._cells):
            cell = self._cells[index]
            if cell.start >= end:
                return
            lo, hi = max(start, cell.start), min(end, cell.start + CELL)
            if hi > lo:
                yield cell, (hi - lo).total_seconds()
            index += 1

    @property
    def days(self) -> list[date]:
        return sorted({c.day for c in self._cells})


def _cell(start: Instant, tz: tzinfo) -> Cell:
    local = start.astimezone(tz)
    minutes = local.hour * 60 + local.minute
    shifted = local.replace(tzinfo=None) - timedelta(minutes=SLEEP_DAY_START_MIN)
    return Cell(
        start=start,
        hour=local.hour,
        day=shifted.date(),
        slot=((minutes - SLEEP_DAY_START_MIN) % (24 * 60)) // SLOT_MIN,
        night=minutes >= NIGHT_START_MIN or minutes < NIGHT_END_MIN,
    )
