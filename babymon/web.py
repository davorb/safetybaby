"""Dashboard. Reads; never writes monitor state.

Reads: the published Situation (current state), the latest Still, and the Store
(timeline, events, clips, snapshots). Writes: controls go into the brain inbox as a
SetControls fact, the same path the CLI flag uses; the reference image goes to the
Store. No route touches MonitorState.

    GET  /                       single HTML page, polls the JSON routes
    GET  /api/state              Situation as JSON
    GET  /api/frame.jpg          latest Still (503 when the camera is down)
    GET  /api/timeline?hours=24  list[SleepSpan]
    GET  /api/events?hours=24    events with snapshot URL and clip URL#t=offset;
                                 since_id=N keeps only ids above N
    GET  /api/stats?days=7&tz=UTC  Stats over the last `days` (1..90), local-time fields
                                 in the IANA zone `tz` (400 if unknown)
    GET  /stats                  single HTML page, reads /api/stats
    GET  /static/{name}          the shared base.css and tip.js, plain .css/.js names only
    POST /api/controls           {"armed"?: bool, "nap_mode"?: bool}
    POST /api/reference          store the current Still as the good-position reference
    GET  /media/snapshots/{name}, /media/segments/{name}
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, ConfigDict, model_validator

from babymon.domain import (
    AlertSent,
    CameraChanged,
    Fact,
    Instant,
    SetControls,
    Situation,
    SnapshotRef,
    Still,
    Tuning,
    VisionObserved,
)
from babymon.latest import Latest
from babymon.stats import Stats, StatsInput, compute
from babymon.store import Store

INDEX = Path(__file__).parent / "static" / "index.html"
STATS_PAGE = Path(__file__).parent / "static" / "stats.html"
ASSETS = Path(__file__).parent / "static"
ASSET_SUFFIXES = {".css", ".js"}
NO_STORE = {"Cache-Control": "no-store"}
Hours = Annotated[float, Query(gt=0, le=24)]
Days = Annotated[int, Query(ge=1, le=90)]


def _utcnow() -> Instant:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class WebDeps:
    publish: Callable[[Fact], None]
    situation: Latest[Situation]
    stills: Latest[Still]
    store: Store
    now: Callable[[], Instant] = _utcnow
    tuning: Tuning = Tuning()


class ControlsPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    armed: bool | None = None
    nap_mode: bool | None = None

    @model_validator(mode="after")
    def _not_empty(self) -> ControlsPatch:
        if self.armed is None and self.nap_mode is None:
            raise ValueError("set armed, nap_mode, or both")
        return self


def _unavailable(detail: str) -> HTTPException:
    return HTTPException(503, detail, headers=NO_STORE)


def _media(resolve: Callable[[str], Path], name: str) -> FileResponse:
    try:
        path = resolve(name)
    except ValueError:
        raise HTTPException(404) from None
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)


def _asset(name: str) -> Path:
    if "/" in name or "\\" in name or Path(name).suffix not in ASSET_SUFFIXES:
        raise ValueError(name)
    return ASSETS / name


def create_app(deps: WebDeps) -> FastAPI:
    app = FastAPI(title="babymon", docs_url=None, redoc_url=None, openapi_url=None)
    store = deps.store

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX.read_text()

    @app.get("/stats", response_class=HTMLResponse)
    def stats_page() -> str:
        return STATS_PAGE.read_text()

    @app.get("/static/{name}")
    def static_asset(name: str) -> FileResponse:
        return _media(_asset, name)

    @app.get("/api/state")
    def state() -> Situation:
        situation = deps.situation.get()
        if situation is None:
            raise _unavailable("no situation published yet")
        return situation

    @app.get("/api/frame.jpg")
    def frame() -> Response:
        still = deps.stills.get()
        situation = deps.situation.get()
        if still is None or (situation is not None and not situation.camera_up):
            raise _unavailable("camera is down")
        return Response(still.jpeg, media_type="image/jpeg", headers=NO_STORE)

    @app.get("/api/timeline")
    def timeline(hours: Hours = 24) -> list[dict[str, Any]]:
        until = deps.now()
        spans = store.sleep_timeline(until - timedelta(hours=hours), until)
        return [{"state": s.state, "start": s.start, "end": s.end} for s in spans]

    @app.get("/api/events")
    def events(hours: Hours = 24, since_id: int = Query(0, ge=0)) -> list[dict[str, Any]]:
        until = deps.now()
        stored = store.events(until - timedelta(hours=hours), until)
        out = []
        for item in sorted(stored, key=lambda s: s.event.at, reverse=True):
            if item.id <= since_id:
                continue
            event = item.event
            snapshot = event.snapshot if isinstance(event, VisionObserved | AlertSent) else None
            clip = store.clip_at(event.at)
            clip_url = f"/media/segments/{clip.name}#t={clip.offset_s:g}" if clip else None
            out.append(
                {
                    "id": item.id,
                    "event": event.model_dump(mode="json"),
                    "snapshot_url": f"/media/snapshots/{snapshot}" if snapshot else None,
                    "clip_url": clip_url,
                }
            )
        return out

    @app.get("/api/stats")
    def stats(days: Days = 7, tz: str = "UTC") -> Stats:
        try:
            zone = ZoneInfo(tz)
        except (ZoneInfoNotFoundError, ValueError, OSError):
            raise HTTPException(400, f"unknown time zone: {tz!r}") from None
        until = deps.now()
        since = until - timedelta(days=days)
        return compute(
            StatsInput(
                start=since,
                now=until,
                tz=zone,
                tuning=deps.tuning,
                spans=store.sleep_timeline(since, until),
                visions=store.events_of(VisionObserved, since, until),
                alerts=store.events_of(AlertSent, since, until),
                camera=store.events_of(CameraChanged, since, until),
                camera_before=store.last_event_before(CameraChanged, since),
                minutes=store.minutes(since, until),
            )
        )

    @app.post("/api/controls", status_code=202)
    def controls(patch: ControlsPatch) -> ControlsPatch:
        deps.publish(SetControls(deps.now(), armed=patch.armed, nap_mode=patch.nap_mode))
        return patch

    @app.post("/api/reference")
    def reference() -> dict[str, Any]:
        still = deps.stills.get()
        if still is None:
            raise _unavailable("no frame to store")
        store.set_reference(still)
        return {"at": still.at}

    @app.get("/media/snapshots/{name}")
    def snapshot(name: str) -> FileResponse:
        return _media(lambda n: store.snapshot_path(SnapshotRef(n)), name)

    @app.get("/media/segments/{name}")
    def segment(name: str) -> FileResponse:
        return _media(store.segment_path, name)

    return app
