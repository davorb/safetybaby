"""Dashboard. Reads; never writes monitor state.

Reads: the published Situation (current state), the latest Still, and the Store
(timeline, events, clips, snapshots). Writes: controls go into the brain inbox as a
SetControls fact, the same path the CLI flag uses; the reference image goes to the
Store. No route touches MonitorState.

    GET  /                       single HTML page, polls the JSON routes
    GET  /api/state              Situation as JSON
    GET  /api/frame.jpg          latest Still (503 when the camera is down)
    GET  /api/timeline?hours=24  list[SleepSpan]
    GET  /api/events?hours=24    events with snapshot URL and clip URL#t=offset
    POST /api/controls           {"armed"?: bool, "nap_mode"?: bool}
    POST /api/reference          store the current Still as the good-position reference
    GET  /media/snapshots/{name}, /media/segments/{name}
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from fastapi import FastAPI

from babymon.domain import Fact, Situation, Still
from babymon.latest import Latest
from babymon.store import Store


@dataclass(frozen=True, slots=True)
class WebDeps:
    publish: Callable[[Fact], None]
    situation: Latest[Situation]
    stills: Latest[Still]
    store: Store


def create_app(deps: WebDeps) -> FastAPI:
    raise NotImplementedError
