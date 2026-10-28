from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from babymon.domain import (
    Alert,
    AlertKind,
    AlertSent,
    CameraChanged,
    Fact,
    Priority,
    SetControls,
    Situation,
    SleepChanged,
    SleepState,
    SnapshotRef,
    Still,
    VisionObserved,
    VisionReason,
)
from babymon.latest import Latest
from babymon.store import Clip, SleepSpan, StoredEvent
from babymon.web import INDEX, WebDeps, create_app
from tests.conftest import T0, Run, at, observation, start

NOW = at(3600)
SEGMENT = bytes(range(256)) * 4


def jpeg() -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((9, 16, 3), 128, np.uint8))
    assert ok
    return buf.tobytes()


class FakeStore:
    """The slice of Store that web.py reads, backed by a tmp dir like the real one."""

    def __init__(self, root: Path) -> None:
        self.snapshots = root / "snapshots"
        self.segments = root / "segments"
        self.snapshots.mkdir()
        self.segments.mkdir()
        self.stored: list[StoredEvent] = []
        self.spans: list[SleepSpan] = []
        self.clip: Clip | None = None
        self.reference_still: Still | None = None

    def _resolve(self, root: Path, name: str) -> Path:
        if "/" in name or "\\" in name or name in ("", ".", ".."):
            raise ValueError(name)
        return root / name

    def snapshot_path(self, ref: SnapshotRef) -> Path:
        return self._resolve(self.snapshots, ref)

    def segment_path(self, name: str) -> Path:
        return self._resolve(self.segments, name)

    def events(self, since, until, limit: int = 500):
        return [s for s in self.stored if since <= s.event.at <= until]

    def sleep_timeline(self, since, until):
        return self.spans

    def clip_at(self, when):
        return self.clip

    def set_reference(self, still: Still) -> None:
        self.reference_still = still


@dataclass
class Rig:
    store: FakeStore
    situation: Latest[Situation]
    stills: Latest[Still]
    published: list[Fact] = field(default_factory=list)

    @property
    def client(self) -> TestClient:
        deps = WebDeps(self.published.append, self.situation, self.stills, self.store, lambda: NOW)
        return TestClient(create_app(deps))


@pytest.fixture
def rig(tmp_path) -> Rig:
    return Rig(FakeStore(tmp_path), Latest(), Latest())


def published_situation(**state_kwargs) -> Situation:
    return Run(start(**state_kwargs)).seconds(1, 5).situation()


def test_index_references_only_routes_that_exist(rig):
    html = rig.client.get("/")
    assert html.status_code == 200
    assert html.text == INDEX.read_text()
    served = {r.path for r in rig.client.app.routes}
    served_media = {p.rsplit("/", 1)[0] for p in served if p.startswith("/media/")}
    used = set(re.findall(r"""['"](/(?:api|media)/[\w./-]*)""", html.text))
    assert used
    for path in used:
        known = path in served or path.rstrip("/") in served_media
        assert known, path


def test_state_is_503_until_published_then_round_trips(rig):
    assert rig.client.get("/api/state").status_code == 503
    s = published_situation()
    rig.situation.set(s)
    r = rig.client.get("/api/state")
    assert r.status_code == 200
    assert Situation.model_validate(r.json()) == s


def test_frame_serves_the_latest_still_uncached(rig):
    rig.stills.set(Still(jpeg(), at(1)))
    rig.situation.set(published_situation())
    r = rig.client.get("/api/frame.jpg")
    assert (r.status_code, r.headers["content-type"]) == (200, "image/jpeg")
    assert r.headers["cache-control"] == "no-store"
    assert r.content == jpeg()
    newer = jpeg() + b"\x00"
    rig.stills.set(Still(newer, at(2)))
    assert rig.client.get("/api/frame.jpg").content == newer


def test_frame_is_503_without_a_still_or_with_the_camera_down(rig):
    r = rig.client.get("/api/frame.jpg")
    assert (r.status_code, r.headers["cache-control"]) == (503, "no-store")
    rig.stills.set(Still(jpeg(), at(1)))
    rig.situation.set(published_situation(camera_up=False))
    r = rig.client.get("/api/frame.jpg")
    assert (r.status_code, r.headers["cache-control"]) == (503, "no-store")


def test_timeline_lists_spans_with_unknown_as_null(rig):
    rig.store.spans = [
        SleepSpan(None, NOW - timedelta(hours=2), NOW - timedelta(hours=1)),
        SleepSpan(SleepState.ASLEEP, NOW - timedelta(hours=1), NOW),
    ]
    spans = rig.client.get("/api/timeline?hours=24").json()
    assert [s["state"] for s in spans] == [None, "asleep"]
    assert spans[1]["end"].startswith("2026-10-27T03:00:00")


@pytest.mark.parametrize("query", ["hours=0", "hours=25", "hours=abc"])
def test_hours_outside_the_retention_window_is_rejected(rig, query):
    assert rig.client.get(f"/api/events?{query}").status_code == 422
    assert rig.client.get(f"/api/timeline?{query}").status_code == 422


def test_events_are_newest_first_with_snapshot_and_clip_urls(rig):
    alert = Alert(
        kind=AlertKind.CRYING, priority=Priority.HIGH, at=at(300), title="Crying", message="m"
    )
    rig.store.stored = [
        StoredEvent(
            1, SleepChanged(at=at(100), previous=SleepState.ASLEEP, current=SleepState.AWAKE)
        ),
        StoredEvent(
            2,
            VisionObserved(
                at=at(200),
                reason=VisionReason.PERIODIC,
                observation=observation(),
                snapshot=SnapshotRef("20261027-020330-periodic.jpg"),
            ),
        ),
        StoredEvent(3, AlertSent(at=at(300), alert=alert, snapshot=None, delivered=True)),
        StoredEvent(4, CameraChanged(at=at(400), up=False, detail="stalled")),
    ]
    rig.store.clip = Clip("20261027-020000.mp4", T0, offset_s=12.5, growing=False)
    events = rig.client.get("/api/events").json()
    assert [e["id"] for e in events] == [4, 3, 2, 1]
    by_id = {e["id"]: e for e in events}
    assert by_id[2]["snapshot_url"] == "/media/snapshots/20261027-020330-periodic.jpg"
    assert by_id[2]["event"]["type"] == "vision"
    assert by_id[1]["snapshot_url"] is None
    assert by_id[3]["snapshot_url"] is None
    assert by_id[1]["clip_url"] == "/media/segments/20261027-020000.mp4#t=12.5"


def test_since_id_keeps_only_newer_events(rig):
    rig.store.stored = [
        StoredEvent(i, CameraChanged(at=at(i * 10), up=True, detail="")) for i in (1, 2, 3)
    ]
    assert [e["id"] for e in rig.client.get("/api/events?since_id=1").json()] == [3, 2]
    assert rig.client.get("/api/events?since_id=3").json() == []
    assert rig.client.get("/api/events?since_id=-1").status_code == 422


def test_event_without_a_covering_segment_has_no_clip_url(rig):
    rig.store.stored = [StoredEvent(1, CameraChanged(at=at(5), up=True, detail=""))]
    assert rig.client.get("/api/events").json()[0]["clip_url"] is None


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"nap_mode": True}, SetControls(NOW, armed=None, nap_mode=True)),
        ({"armed": False}, SetControls(NOW, armed=False, nap_mode=None)),
        ({"armed": True, "nap_mode": False}, SetControls(NOW, armed=True, nap_mode=False)),
    ],
)
def test_controls_publish_one_patch_with_only_the_given_fields(rig, body, expected):
    r = rig.client.post("/api/controls", json=body)
    assert r.status_code == 202
    assert rig.published == [expected]


@pytest.mark.parametrize("body", [{}, {"armed": None}, {"armed": "maybe"}, {"volume": 3}])
def test_a_bad_controls_body_publishes_nothing(rig, body):
    assert rig.client.post("/api/controls", json=body).status_code == 422
    assert rig.published == []


def test_reference_stores_the_current_still(rig):
    assert rig.client.post("/api/reference").status_code == 503
    still = Still(jpeg(), at(9))
    rig.stills.set(still)
    assert rig.client.post("/api/reference").status_code == 200
    assert rig.store.reference_still == still


def test_snapshot_is_served(rig):
    (rig.store.snapshots / "a.jpg").write_bytes(jpeg())
    r = rig.client.get("/media/snapshots/a.jpg")
    assert (r.status_code, r.headers["content-type"], r.content) == (200, "image/jpeg", jpeg())
    assert rig.client.get("/media/snapshots/missing.jpg").status_code == 404


@pytest.mark.parametrize("route", ["snapshots", "segments"])
@pytest.mark.parametrize("name", ["..%2Fsecret.txt", "..%2F..%2Fsecret.txt", "%2E%2E", "a%5Cb"])
def test_media_never_leaves_its_directory(rig, tmp_path, route, name):
    (tmp_path / "secret.txt").write_text("secret")
    r = rig.client.get(f"/media/{route}/{name}")
    assert r.status_code == 404
    assert "secret" not in r.text


def test_segment_supports_range_requests(rig):
    (rig.store.segments / "s.mp4").write_bytes(SEGMENT)
    c = rig.client
    whole = c.get("/media/segments/s.mp4")
    assert (whole.status_code, whole.content) == (200, SEGMENT)
    assert whole.headers["accept-ranges"] == "bytes"
    part = c.get("/media/segments/s.mp4", headers={"Range": "bytes=10-19"})
    assert (part.status_code, part.content) == (206, SEGMENT[10:20])
    assert part.headers["content-range"] == f"bytes 10-19/{len(SEGMENT)}"
    tail = c.get("/media/segments/s.mp4", headers={"Range": "bytes=1000-"})
    assert (tail.status_code, tail.content) == (206, SEGMENT[1000:])
    past_end = c.get("/media/segments/s.mp4", headers={"Range": "bytes=5000-"})
    assert past_end.status_code == 416
