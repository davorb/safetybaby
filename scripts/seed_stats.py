"""Build a synthetic 21-day data dir for a 4-month-old through the real Store API.

    uv run python scripts/seed_stats.py [--out DIR] [--days 21] [--seed 7]

Local clock is Europe/Stockholm; stored instants are UTC. The data ends at the current
minute, so /api/stats?days=N sees it as live. Snapshots exist only for the last 24 h,
like the real ring buffer; older vision rows keep their (dangling) names.
"""

from __future__ import annotations

import argparse
import math
import random
import shutil
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from babymon.domain import (
    Alert,
    AlertKind,
    AlertSent,
    BabyLocation,
    BodyPosition,
    CameraChanged,
    Eyes,
    HeadDirection,
    MinuteRollup,
    Observation,
    Priority,
    SleepChanged,
    SleepState,
    SnapshotRef,
    Still,
    VisionObserved,
    VisionReason,
)
from babymon.store import Store

LOCAL = ZoneInfo("Europe/Stockholm")
A, S, W, C = SleepState.ASLEEP, SleepState.STIRRING, SleepState.AWAKE, SleepState.CRYING
MIN = timedelta(minutes=1)


@dataclass
class Seg:
    state: SleepState
    start: datetime
    end: datetime
    body: BodyPosition = BodyPosition.BACK
    head: HeadDirection = HeadDirection.CENTER


def pick(rng: random.Random, table: dict) -> object:
    return rng.choices(list(table), weights=list(table.values()))[0]


class Plan:
    """A gapless list of state segments, built by appending durations."""

    def __init__(self, rng: random.Random, start: datetime) -> None:
        self.rng, self.cursor, self.segs = rng, start, []

    def add(self, state, minutes, body=BodyPosition.BACK, head=HeadDirection.CENTER) -> None:
        end = self.cursor + timedelta(minutes=minutes)
        self.segs.append(Seg(state, self.cursor, end, body, head))
        self.cursor = end

    def until(self, local_time: datetime, fussy: float = 0.0) -> None:
        """Awake until local_time, with an optional crying spell."""
        t = local_time.astimezone(UTC)
        gap = (t - self.cursor).total_seconds() / 60
        if gap <= 0:
            return
        if gap > 25 and self.rng.random() < fussy:
            before = self.rng.uniform(5, gap - 15)
            self.add(W, before)
            self.add(C, self.rng.uniform(0.7, 6))
            gap = (t - self.cursor).total_seconds() / 60
        if gap > 0:
            self.add(W, gap)

    def sleep_bout(self, minutes: float, arousal: bool = True) -> None:
        rng = self.rng
        body = pick(
            rng,
            {
                BodyPosition.BACK: 80,
                BodyPosition.SIDE_LEFT: 8,
                BodyPosition.SIDE_RIGHT: 8,
                BodyPosition.STOMACH: 2.5,
                BodyPosition.UNKNOWN: 1.5,
            },
        )
        head = pick(
            rng,
            {
                HeadDirection.CENTER: 55,
                HeadDirection.LEFT: 22,
                HeadDirection.RIGHT: 20,
                HeadDirection.UNKNOWN: 3,
            },
        )
        shrink = {
            BodyPosition.STOMACH: 0.6,
            BodyPosition.SIDE_LEFT: 0.85,
            BodyPosition.SIDE_RIGHT: 0.85,
        }
        minutes *= shrink.get(body, 1.0) * (0.9 if head is HeadDirection.LEFT else 1.0)
        pieces = [minutes]
        if arousal and minutes > 60:
            for _ in range(rng.choice([0, 1, 1, 2])):
                cut = rng.uniform(0.2, 0.8) * pieces[-1]
                pieces[-1:] = [cut, pieces[-1] - cut]
        for i, p in enumerate(pieces):
            if i:
                self.add(S, rng.uniform(1.0, 2.5))
            self.add(A, p, body, head)

    def wake_up(self, cry_prob: float, awake_min: float) -> None:
        self.add(S, self.rng.uniform(2, 4))
        if self.rng.random() < cry_prob:
            self.add(W, self.rng.uniform(0.3, 1.5))
            self.add(C, self.rng.uniform(0.6, 6))
        self.add(W, awake_min)


def local(d: date, hh: int, mm: int = 0) -> datetime:
    return datetime(d.year, d.month, d.day, hh, mm, tzinfo=LOCAL)


def build_plan(rng: random.Random, start: datetime, end: datetime) -> list[Seg]:
    plan = Plan(rng, start)
    day = start.astimezone(LOCAL).date()
    while plan.cursor < end:
        j = lambda lo, hi: rng.uniform(lo, hi)  # noqa: E731
        naps = [(9, 15, j(35, 55)), (12, 20, j(70, 100)), (15, 35, j(45, 70))]
        if rng.random() < 0.55:
            naps.append((17, 40, j(20, 35)))
        for hh, mm, length in naps:
            plan.until(local(day, hh, mm) + timedelta(minutes=j(-20, 20)), fussy=0.3)
            plan.sleep_bout(length)
            plan.wake_up(0.2, j(3, 10))
        plan.until(local(day, 18, 40) + timedelta(minutes=j(-15, 15)), fussy=0.55)
        plan.add(W, j(20, 40))
        bed = plan.cursor
        wakings = rng.choice([2, 2, 3, 3, 4])
        wake_local = local(day + timedelta(days=1), 6, 30) + timedelta(minutes=j(-25, 25))
        total = (wake_local.astimezone(UTC) - bed).total_seconds() / 60
        weights = [j(2.2, 3.5)] + [j(1.2, 2.2) for _ in range(wakings - 1)] + [j(1.5, 2.5)]
        scale = (total - wakings * 22) / sum(weights)
        for i, w in enumerate(weights):
            plan.sleep_bout(w * scale)
            if i < wakings:
                plan.wake_up(0.45, j(8, 22))
        day += timedelta(days=1)
    return plan.segs


def state_at(segs: list[Seg], t: datetime) -> Seg | None:
    lo, hi = 0, len(segs) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if segs[mid].end <= t:
            lo = mid + 1
        elif segs[mid].start > t:
            hi = mid - 1
        else:
            return segs[mid]
    return None


def observation(rng: random.Random, seg: Seg) -> Observation:
    confident = rng.random() < 0.8
    conf = rng.uniform(0.6, 0.97) if confident else rng.uniform(0.15, 0.49)
    night = seg.start.astimezone(LOCAL).hour >= 19 or seg.start.astimezone(LOCAL).hour < 7
    if seg.state is A:
        return Observation(
            body=seg.body,
            head=seg.head,
            face_covered=rng.random() < 0.003,
            location=BabyLocation.CRIB,
            eyes=Eyes.OPEN if rng.random() < 0.04 else Eyes.CLOSED,
            confidence=conf,
        )
    held = rng.random() < (0.5 if seg.state is C else 0.25)
    adult = held or rng.random() < (0.7 if night else 0.25)
    location = (
        BabyLocation.HELD
        if held
        else pick(
            rng, {BabyLocation.CRIB: 80, BabyLocation.PARENTS_BED: 6, BabyLocation.NOT_VISIBLE: 14}
        )
    )
    return Observation(
        body=BodyPosition.UNKNOWN
        if held
        else pick(
            rng,
            {
                BodyPosition.BACK: 70,
                BodyPosition.SIDE_LEFT: 12,
                BodyPosition.SIDE_RIGHT: 12,
                BodyPosition.STOMACH: 6,
            },
        ),
        head=pick(
            rng,
            {
                HeadDirection.CENTER: 50,
                HeadDirection.LEFT: 20,
                HeadDirection.RIGHT: 20,
                HeadDirection.UNKNOWN: 10,
            },
        ),
        face_covered=False,
        location=location,
        eyes=pick(rng, {Eyes.OPEN: 85, Eyes.CLOSED: 12, Eyes.UNKNOWN: 3}),
        adult_present=adult,
        confidence=conf,
    )


def minute_row(rng: random.Random, at: datetime, state: SleepState) -> MinuteRollup:
    mean, peak, loud, cry = {
        A: ((0.0015, 0.0045), (0.008, 0.03), (0.0, 2.0), (0.01, 0.18)),
        S: ((0.008, 0.03), (0.03, 0.12), (1.0, 5.0), (0.03, 0.3)),
        W: ((0.025, 0.09), (0.09, 0.45), (2.0, 9.0), (0.05, 0.4)),
        C: ((0.04, 0.14), (0.2, 0.6), (9.0, 24.0), (0.75, 0.98)),
    }[state]
    m = rng.uniform(*mean) * math.exp(rng.gauss(0, 0.3))
    p = max(m, rng.uniform(*peak))
    cry_peak = rng.uniform(*cry)
    cry_s = (
        int(rng.uniform(35, 60))
        if state is C
        else (rng.choice([0, 0, 0, 3]) if cry_peak > 0.6 else 0)
    )
    return MinuteRollup(
        at, 240, m, p, 60, rng.uniform(*loud), rng.uniform(*loud) + 3, cry_peak, cry_s
    )


def jpeg(text: str) -> bytes:
    image = np.full((180, 320, 3), 40, np.uint8)
    cv2.putText(image, text, (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    return buf.tobytes()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(".dev/seed-data"))
    ap.add_argument("--days", type=int, default=21)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    if args.out.exists():
        shutil.rmtree(args.out)
    store = Store.open(args.out)
    end = datetime.now(UTC).replace(second=0, microsecond=0)
    start = end - timedelta(days=args.days)
    segs = build_plan(rng, start, end + timedelta(days=2))

    outage_at = local(start.astimezone(LOCAL).date() + timedelta(days=9), 2, 10).astimezone(UTC)
    outage = (outage_at, outage_at + 40 * MIN)
    restart_at = local(start.astimezone(LOCAL).date() + timedelta(days=14), 13, 40).astimezone(UTC)
    process_down = (restart_at - 12 * MIN, restart_at)

    def away(t: datetime) -> bool:
        return outage[0] <= t < outage[1] or process_down[0] <= t < process_down[1]

    store.append(CameraChanged(at=start, up=True, detail=""))
    store.append(SleepChanged(at=start + timedelta(seconds=1), previous=S, current=segs[0].state))
    previous = segs[0]
    for seg in segs[1:]:
        if seg.start > end:
            break
        if previous.state is not seg.state:
            store.append(SleepChanged(at=seg.start, previous=previous.state, current=seg.state))
        previous = seg
    held = state_at(segs, restart_at)
    if held and held.state is not S and restart_at - held.start >= 10 * MIN:
        # a fresh process starts in STIRRING and only writes when that changes
        store.append(SleepChanged(at=restart_at + 2 * MIN, previous=S, current=held.state))
    store.append(CameraChanged(at=outage[0], up=False, detail="[in#0/rtsp] Connection timed out"))
    store.append(CameraChanged(at=outage[1], up=True, detail=""))
    store.append(CameraChanged(at=restart_at, up=True, detail=""))

    # vision checks
    t, snap_from = start + 5 * MIN, end - timedelta(hours=24)
    n_snap = 0
    while t < end:
        seg = state_at(segs, t)
        if seg and not away(t):
            reason = (
                VisionReason.PERIODIC if rng.random() < 0.85 else VisionReason.SETTLED_AFTER_MOTION
            )
            obs = observation(rng, seg)
            name = SnapshotRef(f"{t:%Y%m%d-%H%M%S}-{reason.value}.jpg")
            if t >= snap_from:
                store.save_snapshot(
                    Still(jpeg(f"{t:%m-%d %H:%M:%S} {seg.state.value}"), t), reason.value
                )
                n_snap += 1
            store.append(VisionObserved(at=t, reason=reason, observation=obs, snapshot=name))
        step = rng.uniform(11, 17) if seg and seg.state is A else rng.uniform(3, 7)
        t += timedelta(minutes=step)

    # alerts
    crying = [
        s
        for s in segs
        if s.state is C and s.end - s.start > 4 * MIN and s.end < end and not away(s.start)
    ]
    sent = 0
    for s in crying:
        at = s.start + timedelta(seconds=3)
        alert = Alert(
            kind=AlertKind.CRYING,
            priority=Priority.HIGH,
            at=at,
            title="Crying",
            message="Crying for 3 s",
        )
        store.append(AlertSent(at=at, alert=alert, snapshot=None, delivered=True))
        sent += 1
    extras = [
        (AlertKind.HEAD_LEFT, Priority.HIGH, 3, 2, True),
        (AlertKind.HEAD_LEFT, Priority.HIGH, 11, 3, True),
        (AlertKind.ON_STOMACH, Priority.EMERGENCY, 6, 1, False),
        (AlertKind.ON_STOMACH, Priority.EMERGENCY, 6, 1, True),
        (AlertKind.CAMERA_OFFLINE, Priority.HIGH, 9, 2, True),
    ]
    for kind, priority, day, hour, delivered in extras:
        at = (
            (start + timedelta(days=day))
            .astimezone(LOCAL)
            .replace(hour=hour, minute=20)
            .astimezone(UTC)
        )
        if kind is AlertKind.CAMERA_OFFLINE:
            at = outage[0] + MIN
        alert = Alert(kind=kind, priority=priority, at=at, title=kind.value, message="synthetic")
        store.append(
            AlertSent(
                at=at,
                alert=alert,
                snapshot=None,
                delivered=delivered,
                detail="" if delivered else "phone unreachable",
            )
        )

    # minute rollups
    t, rows = start, 0
    while t < end:
        seg = state_at(segs, t + timedelta(seconds=30))
        if seg and not away(t):
            state = seg.state
            store.append_minute(minute_row(rng, t, state))
            rows += 1
        t += MIN
    print(
        f"wrote {args.out}: {len(segs)} segments, {rows} minutes, "
        f"{n_snap} snapshots, {sent} crying alerts"
    )


if __name__ == "__main__":
    main()
