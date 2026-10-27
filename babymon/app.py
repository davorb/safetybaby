"""Wiring. The only module that knows every actor. No logic beyond plumbing.

Actors and what each owns:

    camera      ffmpeg process, pipes, MotionMeter        -> facts, stills, pcm
    audio       classifier, NoiseFloor                    -> facts
    vision      one in-flight model call                  -> facts
    brain       MonitorState (sole owner)                 -> effects, published Situation
    recorder    (inside camera's ffmpeg; files on disk)
    writer      ordered event appends                     -> Store
    alerter     delivery + AlertSent rows                 -> phones, Store
    retention   periodic Store.sweep                      -> disk
    ticker      1 Hz Tick                                 -> facts
    web         HTTP                                      -> facts (SetControls), Store (reference)

Channels: one asyncio.Queue[Fact] into the brain (many writers, append-only, the
structural serialiser); Latest cells for newest-wins data; two queues out of the
brain (events to the writer, alerts to the alerter).

Failure policy: actors handle their expected failures (ffmpeg exits, model errors,
HTTP errors). An unexpected exception ends the TaskGroup and the process; Docker's
restart policy brings it back, and startup converges from disk (Store.restore,
Store.sweep, fresh ffmpeg). Restart is the recovery path, so it is the tested path.
"""

from __future__ import annotations

import asyncio
from typing import Never

from babymon.brain import MonitorState
from babymon.config import Settings
from babymon.domain import Alert, Effect, Event, Fact, RequestVision, SetControls, Situation, Tuning
from babymon.latest import Latest
from babymon.rules import Decider
from babymon.store import Store


async def run(settings: Settings, initial_controls: SetControls | None = None) -> Never:
    """
    store = Store.open(settings.data_dir); store.sweep(now)
    state = brain.initial_state(store.restore(now, cooldown), settings.capabilities(), now)
    build channels; build backends (ClaudeVision or none, Pushover or LogOnly)
    if initial_controls: facts.put_nowait(initial_controls)     (CLI --nap / --disarm)
    async with TaskGroup: camera.run, audio.run_audio, vision.run_vision (if enabled),
        run_brain, run_writer, notify.run_alerter, run_ticker, run_retention, uvicorn
    """
    raise NotImplementedError


async def run_brain(
    state: MonitorState,
    decider: Decider,
    tuning: Tuning,
    facts: asyncio.Queue[Fact],
    published: Latest[Situation],
    events: asyncio.Queue[Event],
    alerts: asyncio.Queue[Alert],
    vision_requests: Latest[RequestVision],
) -> Never:
    """
    loop:
        fact = await facts.get()
        state, effects = brain.observe(state, fact, tuning)
        if fact is Tick:
            s = brain.situation(state, tuning)
            state, notifies = brain.gate(state, await decider.decide(s), tuning)
            effects += notifies
            published.set(s)
        for e in effects: dispatch(e, ...)
    """
    raise NotImplementedError


def dispatch(
    effect: Effect,
    events: asyncio.Queue[Event],
    alerts: asyncio.Queue[Alert],
    vision_requests: Latest[RequestVision],
) -> None:
    """match effect: Record -> events.put_nowait; Notify -> alerts.put_nowait;
    RequestVision -> vision_requests.set; case _: assert_never. Never awaits, so the
    brain never waits on I/O."""
    raise NotImplementedError


async def run_writer(store: Store, events: asyncio.Queue[Event]) -> Never:
    """Drain events in order into Store.append via to_thread."""
    raise NotImplementedError


async def run_ticker(facts: asyncio.Queue[Fact]) -> Never:
    raise NotImplementedError


async def run_retention(store: Store, every_s: float = 300.0) -> Never:
    raise NotImplementedError
