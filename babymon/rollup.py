"""Per-minute signal rollup. Pure: (accumulator, fact, tuning) -> (accumulator, finished).

Motion arrives at 4 Hz and audio at 1 Hz; the statistics page needs only a few numbers
per minute. A minute is emitted when the first fact of a later minute arrives, and Tick
guarantees one every second, so a finished minute is never held back by silence. A minute
with no samples has no accumulator and emits nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from babymon.domain import AudioSample, Fact, Instant, MinuteRollup, MotionSample, Tick, Tuning


@dataclass(frozen=True, slots=True)
class Accumulator:
    minute: Instant
    motion_n: int = 0
    motion_sum: float = 0.0
    motion_peak: float = 0.0
    audio_n: int = 0
    loudness_sum: float = 0.0
    loudness_peak: float = 0.0
    cry_peak: float = 0.0
    cry_seconds: int = 0


def _floor(at: Instant) -> Instant:
    return at.replace(second=0, microsecond=0)


def finish(acc: Accumulator) -> MinuteRollup:
    return MinuteRollup(
        at=acc.minute,
        motion_samples=acc.motion_n,
        motion_mean=acc.motion_sum / acc.motion_n if acc.motion_n else None,
        motion_peak=acc.motion_peak if acc.motion_n else None,
        audio_samples=acc.audio_n,
        loudness_mean_db=acc.loudness_sum / acc.audio_n if acc.audio_n else None,
        loudness_peak_db=acc.loudness_peak if acc.audio_n else None,
        cry_peak=acc.cry_peak if acc.audio_n else None,
        cry_seconds=acc.cry_seconds,
    )


def _fold(acc: Accumulator, fact: MotionSample | AudioSample, tuning: Tuning) -> Accumulator:
    if isinstance(fact, MotionSample):
        peak = fact.level if acc.motion_n == 0 else max(acc.motion_peak, fact.level)
        return replace(
            acc, motion_n=acc.motion_n + 1, motion_sum=acc.motion_sum + fact.level, motion_peak=peak
        )
    first = acc.audio_n == 0
    return replace(
        acc,
        audio_n=acc.audio_n + 1,
        loudness_sum=acc.loudness_sum + fact.loudness_db,
        loudness_peak=fact.loudness_db if first else max(acc.loudness_peak, fact.loudness_db),
        cry_peak=fact.cry_probability if first else max(acc.cry_peak, fact.cry_probability),
        cry_seconds=acc.cry_seconds + (fact.cry_probability >= tuning.cry_probability),
    )


def step(
    acc: Accumulator | None, fact: Fact, tuning: Tuning
) -> tuple[Accumulator | None, MinuteRollup | None]:
    """A fact older than the open minute joins it rather than reopening a closed one."""
    if not isinstance(fact, MotionSample | AudioSample | Tick):
        return acc, None
    minute = _floor(fact.at)
    finished = None
    if acc is not None and minute > acc.minute:
        finished, acc = finish(acc), None
    if isinstance(fact, Tick):
        return acc, finished
    return _fold(acc or Accumulator(minute), fact, tuning), finished
