"""Serve the real dashboard over synthetic data, for the end-to-end tests in e2e/.

    uv run python scripts/e2e_server.py [--port 8090] [--data DIR]

No camera, no brain. The data dir is rebuilt on every start: 21 days from seed_stats.py
plus a few fixed events in the last hour that the specs name. The seed never writes a
covered face outside the crib, so the fixed vision row is the only one of its kind. The
Situation is fixed (asleep, camera up, vision and push disabled). POST /api/controls
patches it and logs a ControlsChanged, as the brain would.
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import uvicorn

from babymon.domain import (
    Alert,
    AlertKind,
    AlertSent,
    BabyLocation,
    BodyPosition,
    Capabilities,
    Controls,
    ControlsChanged,
    Eyes,
    Fact,
    HeadDirection,
    Observation,
    Priority,
    SetControls,
    Situation,
    SleepChanged,
    SleepState,
    Still,
    VisionObserved,
    VisionReason,
)
from babymon.latest import Latest
from babymon.store import Store
from babymon.web import WebDeps, create_app

SEED = Path(__file__).parent / "seed_stats.py"


def jpeg(text: str) -> bytes:
    image = np.full((360, 640, 3), 40, np.uint8)
    cv2.putText(image, text, (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 2)
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    return buf.tobytes()


def add_known_events(store: Store, now: datetime) -> None:
    woke = now - timedelta(minutes=25)
    store.append(SleepChanged(at=woke, previous=SleepState.STIRRING, current=SleepState.ASLEEP))
    seen = now - timedelta(minutes=12)
    snapshot = store.save_snapshot(Still(jpeg("e2e vision check"), seen), "periodic")
    observation = Observation(
        body=BodyPosition.STOMACH,
        head=HeadDirection.LEFT,
        face_covered=True,
        location=BabyLocation.PARENTS_BED,
        eyes=Eyes.CLOSED,
        confidence=0.9,
    )
    store.append(
        VisionObserved(
            at=seen, reason=VisionReason.PERIODIC, observation=observation, snapshot=snapshot
        )
    )
    cried = now - timedelta(minutes=5)
    alert = Alert(
        kind=AlertKind.CRYING,
        priority=Priority.HIGH,
        at=cried,
        title="Crying",
        message="E2E crying for 4 s",
    )
    store.append(
        AlertSent(at=cried, alert=alert, snapshot=None, delivered=False, detail="push disabled")
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--data", type=Path, help="default: a temp dir, deleted on exit")
    args = ap.parse_args()
    # uvicorn re-raises SIGTERM after shutdown; the default handler would skip the cleanup.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    with tempfile.TemporaryDirectory(prefix="babymon-e2e-") as tmp:
        serve(args.data or Path(tmp) / "data", args.port)


def serve(data: Path, port: int) -> None:
    # The seed commits row by row: seconds on tmpfs, minutes on a disk that fsyncs.
    subprocess.run([sys.executable, str(SEED), "--out", str(data)], check=True)
    store = Store.open(data)
    now = datetime.now(UTC)
    add_known_events(store, now)

    situation: Latest[Situation] = Latest()
    situation.set(
        Situation(
            at=now,
            controls=Controls(),
            capabilities=Capabilities(vision=False, push=False),
            camera_up=True,
            camera_down_for_s=0.0,
            sleep=SleepState.ASLEEP,
            sleep_for_s=25 * 60,
            crying_for_s=0.0,
            observation=None,
            observation_age_s=None,
            face_covered_streak=0,
            stomach_streak=0,
            head_left_for_s=0.0,
        )
    )
    stills: Latest[Still] = Latest()
    stills.set(Still(jpeg("babymon e2e camera"), now))

    def publish(fact: Fact) -> None:
        if not isinstance(fact, SetControls):
            return
        current = situation.get()
        assert current is not None
        controls = current.controls.model_copy(
            update={
                k: v
                for k, v in (("armed", fact.armed), ("nap_mode", fact.nap_mode))
                if v is not None
            }
        )
        store.append(ControlsChanged(at=fact.at, controls=controls))
        situation.set(current.model_copy(update={"controls": controls, "at": fact.at}))

    app = create_app(WebDeps(publish=publish, situation=situation, stills=stills, store=store))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
