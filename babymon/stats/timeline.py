"""Sleep spans to the things worth counting: observed time, bouts, crying episodes.

Observed time is the store's sleep timeline with every stretch the camera was known to be
down blanked to None. The brain keeps its last state through an outage, so without the
blanking a night with the camera unplugged would read as hours of confident sleep.

A bout is how a parent sees sleep: ASLEEP runs joined across brief interruptions. The
state machine drops ASLEEP to STIRRING on three seconds of movement, so counting raw
ASLEEP runs would split one night into dozens of "naps".
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import timedelta

from babymon.domain import CameraChanged, Instant, SleepSpan, SleepState

BRIEF_AROUSAL = timedelta(minutes=3)

type Interval = tuple[Instant, Instant]


@dataclass(frozen=True, slots=True)
class CameraTimeline:
    up: list[Interval]
    down: list[Interval]
    unknown: list[Interval]


def camera_timeline(
    events: Sequence[CameraChanged], before: CameraChanged | None, start: Instant, now: Instant
) -> CameraTimeline:
    """Time before the first event is unknown unless `before` (the last event ahead of the
    window) says otherwise."""
    up, down, unknown = [], [], []
    state = before.up if before else None
    cursor = start
    for event in [*events, None]:
        edge = event.at if event else now
        if edge > cursor:
            (unknown if state is None else up if state else down).append((cursor, edge))
            cursor = edge
        if event:
            state = event.up
    return CameraTimeline(up, down, unknown)


def observed_spans(
    spans: Iterable[SleepSpan], down: Sequence[Interval], start: Instant, now: Instant
) -> list[SleepSpan]:
    """Contiguous spans over [start, now], camera-down time blanked, equal neighbours merged."""
    out: list[SleepSpan] = []
    for span in spans:
        lo, hi = max(span.start, start), min(span.end, now)
        cursor = lo
        for d_start, d_end in down:
            if d_end <= cursor or d_start >= hi:
                continue
            if d_start > cursor:
                _push(out, span.state, cursor, d_start)
            _push(out, None, max(cursor, d_start), min(hi, d_end))
            cursor = min(hi, d_end)
        if hi > cursor:
            _push(out, span.state, cursor, hi)
    return out


def _push(out: list[SleepSpan], state: SleepState | None, lo: Instant, hi: Instant) -> None:
    if hi <= lo:
        return
    if out and out[-1].state == state and out[-1].end == lo:
        out[-1] = replace(out[-1], end=hi)
    else:
        out.append(SleepSpan(state, lo, hi))


def subtract(lo: Instant, hi: Instant, holes: Sequence[Interval]) -> list[Interval]:
    """[lo, hi) minus the holes."""
    out, cursor = [], lo
    for h_start, h_end in holes:
        if h_end <= cursor or h_start >= hi:
            continue
        if h_start > cursor:
            out.append((cursor, h_start))
        cursor = max(cursor, h_end)
    if hi > cursor:
        out.append((cursor, hi))
    return out


@dataclass(frozen=True, slots=True)
class Bout:
    start: Instant
    end: Instant
    brief_arousals: int
    censored: bool
    """Touches unknown time, the window start or now, so the true length is longer."""
    woke: bool
    """The interruption that follows holds AWAKE or CRYING: a real waking, not a stir."""

    @property
    def duration_s(self) -> float:
        return (self.end - self.start).total_seconds()


@dataclass(frozen=True, slots=True)
class Episode:
    start: Instant
    end: Instant
    censored: bool

    @property
    def duration_s(self) -> float:
        return (self.end - self.start).total_seconds()


def _hidden_edge(spans: Sequence[SleepSpan], i: int) -> bool:
    """True when span i has no known neighbour on a side (window edge or unknown time)."""
    before = spans[i - 1].state if i > 0 else None
    after = spans[i + 1].state if i + 1 < len(spans) else None
    return before is None or after is None


def bouts(spans: Sequence[SleepSpan]) -> list[Bout]:
    asleep = [i for i, s in enumerate(spans) if s.state is SleepState.ASLEEP]
    out: list[Bout] = []
    first = 0
    while first < len(asleep):
        last, arousals = first, 0
        while last + 1 < len(asleep) and _absorbable(spans[asleep[last] + 1 : asleep[last + 1]]):
            last, arousals = last + 1, arousals + 1
        head, tail = asleep[first], asleep[last]
        left_hidden = head == 0 or spans[head - 1].state is None
        right_hidden = tail + 1 >= len(spans) or spans[tail + 1].state is None
        out.append(
            Bout(
                spans[head].start,
                spans[tail].end,
                arousals,
                left_hidden or right_hidden,
                _woke_after(spans, tail),
            )
        )
        first = last + 1
    return out


def _absorbable(gap: Sequence[SleepSpan]) -> bool:
    quiet = all(s.state in (SleepState.STIRRING, SleepState.AWAKE) for s in gap)
    return quiet and gap[-1].end - gap[0].start < BRIEF_AROUSAL


def _woke_after(spans: Sequence[SleepSpan], tail: int) -> bool:
    for span in spans[tail + 1 :]:
        if span.state in (SleepState.AWAKE, SleepState.CRYING):
            return True
        if span.state in (SleepState.ASLEEP, None):
            return False
    return False


def crying_episodes(spans: Sequence[SleepSpan]) -> list[Episode]:
    return [
        Episode(s.start, s.end, _hidden_edge(spans, i))
        for i, s in enumerate(spans)
        if s.state is SleepState.CRYING
    ]
