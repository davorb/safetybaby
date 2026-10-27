"""Single-writer, many-reader cell holding the newest value. The only channel type
besides the brain's inbox queue.

Used where only the newest value matters and a slow reader must never stall the
writer: stills, PCM windows, vision requests, the published Situation. The writer
replaces an immutable reference; readers never see a half-written value.
"""

from __future__ import annotations

import asyncio


class Latest[T]:
    def __init__(self) -> None:
        self._value: T | None = None
        self._version = 0
        self._changed = asyncio.Event()

    @property
    def version(self) -> int:
        return self._version

    def set(self, value: T) -> None:
        """Replace the value and bump the version. Never blocks."""
        self._value = value
        self._version += 1
        self._wake_waiters()

    def _wake_waiters(self) -> None:
        self._changed.set()
        self._changed = asyncio.Event()

    def get(self) -> T | None:
        return self._value

    async def newer_than(self, version: int) -> tuple[int, T]:
        """Wait until the version exceeds `version`; return (version, value).
        Intermediate values a slow reader missed are dropped on purpose."""
        while self._version <= version:
            await self._changed.wait()
        assert self._value is not None
        return self._version, self._value
