"""Single-writer, many-reader cell holding the newest value. The only channel type
besides the brain's inbox queue.

Used where only the newest value matters and a slow reader must never stall the
writer: stills, PCM windows, vision requests, the published Situation. The writer
replaces an immutable reference; readers never see a half-written value.
"""

from __future__ import annotations


class Latest[T]:
    def __init__(self) -> None:
        raise NotImplementedError

    def set(self, value: T) -> None:
        """Replace the value and bump the version. Never blocks."""
        raise NotImplementedError

    def get(self) -> T | None:
        raise NotImplementedError

    async def newer_than(self, version: int) -> tuple[int, T]:
        """Wait until the version exceeds `version`; return (version, value).
        Intermediate values a slow reader missed are dropped on purpose."""
        raise NotImplementedError
