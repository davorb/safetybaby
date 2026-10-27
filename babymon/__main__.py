"""CLI. `uv run babymon ...`

    babymon run [--nap | --no-nap] [--arm | --disarm] [--record-facts FILE]
        Start the service. Flags become one SetControls fact at startup; without
        flags the persisted controls apply.

    babymon probe camera [--seconds 30]
        Run only the camera actor; print frames/s, PCM s/s, stills/s, motion levels,
        and the segment files that appeared. The check for unit 5.
    babymon probe audio FILE.wav
        Print P(cry) and loudness per second. The check for unit 7.
    babymon probe vision FRAME.jpg [--reference REF.jpg]
        Print the typed Observation. The check for unit 8.
    babymon probe alert --priority {normal,high,emergency}
        Send a test notification with the latest still. The check for unit 9.
    babymon reference
        Grab one still from the camera and store it as the reference image.
    babymon replay FACTS.jsonl
        Feed recorded facts through brain + RulesDecider and print effects. Lets a
        real night's facts become a regression fixture.

Each subcommand imports the module that owns its work only when it runs, so `--help`
and `replay` need no camera, model or network library. The owning functions are:

    app.run(settings, initial_controls, record_facts)         async
    camera.probe(settings, seconds)                           async
    audio.probe(settings, path)                               async
    vision.probe(settings, frame_path, reference_path)        async
    notify.probe(settings, priority)                          async
    camera.grab_still(settings) -> Still                      async

A probe reports failure by raising SystemExit with its message.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import assert_never

from babymon import brain, factlog
from babymon.domain import (
    Capabilities,
    Controls,
    Effect,
    Event,
    Fact,
    Instant,
    Notify,
    Priority,
    Record,
    RequestVision,
    Restored,
    SetControls,
    Tick,
    Tuning,
)
from babymon.rules import RulesDecider

RECORD_FACTS_HELP = (
    "append every fact the brain receives to FILE as JSONL, one object per line: "
    '{"type": <tick|motion|audio|vision_done|vision_failed|camera|set_controls|delivery>, '
    '"at": <ISO 8601 with offset>, ...the fact\'s own fields}. '
    "`babymon replay FILE` reads it back. Field layout: babymon/factlog.py"
)


REPLAY_TUNING = Tuning()
REPLAY_CAPABILITIES = Capabilities(vision=True, push=True)


def replay_facts(facts: Iterable[Fact]) -> list[tuple[Instant, Effect]]:
    """Drive the core the way app.run_brain does: observe every fact, and on a Tick
    also situation -> decide -> gate. Starts from default controls, an empty ledger and
    unknown sleep, one second before the first fact. Tuning is the default, and vision
    and push count as enabled, whatever the recording process had."""
    tuning = REPLAY_TUNING
    decider = RulesDecider(tuning)
    state = None
    effects: list[tuple[Instant, Effect]] = []
    for fact in facts:
        if state is None:
            start = fact.at - timedelta(seconds=1)
            state = brain.initial_state(Restored(Controls(), None, {}), REPLAY_CAPABILITIES, start)
        state, observed = brain.observe(state, fact, tuning)
        effects.extend((fact.at, e) for e in observed)
        if isinstance(fact, Tick):
            verdicts = decider.evaluate(brain.situation(state, tuning))
            state, notifies = brain.gate(state, verdicts, tuning)
            effects.extend((fact.at, n) for n in notifies)
    return effects


def _event_summary(event: Event) -> str:
    match event.type:
        case "sleep_changed":
            return f"{event.previous} -> {event.current}"
        case "vision":
            o = event.observation
            return (
                f"{event.reason} body={o.body} head={o.head} eyes={o.eyes} "
                f"face_covered={o.face_covered} in_crib={o.in_crib} confidence={o.confidence}"
            )
        case "alert":
            return f"{event.alert.kind} delivered={event.delivered}"
        case "controls":
            return f"armed={event.controls.armed} nap_mode={event.controls.nap_mode}"
        case "camera":
            return f"up={event.up} {event.detail}".rstrip()
        case _:
            assert_never(event)


def format_effect(at: Instant, effect: Effect) -> str:
    stamp = at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    match effect:
        case Notify(alert=alert):
            body = f"notify {alert.priority} {alert.kind} {alert.title!r}"
        case Record(event=event):
            body = f"record {event.type} {_event_summary(event)}"
        case RequestVision(reason=reason):
            body = f"request_vision {reason}"
        case _:
            assert_never(effect)
    return f"{stamp}  {body}"


def _existing_file(text: str) -> Path:
    path = Path(text)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"no such file: {text}")
    return path


def _positive(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="babymon", description="AI baby monitor.")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="start the service")
    nap = run.add_mutually_exclusive_group()
    nap.add_argument("--nap", dest="nap_mode", action="store_true", default=None)
    nap.add_argument("--no-nap", dest="nap_mode", action="store_false")
    arm = run.add_mutually_exclusive_group()
    arm.add_argument("--arm", dest="armed", action="store_true", default=None)
    arm.add_argument("--disarm", dest="armed", action="store_false")
    run.add_argument("--record-facts", type=Path, metavar="FILE", help=RECORD_FACTS_HELP)

    probe = commands.add_parser("probe", help="check one subsystem on its own")
    targets = probe.add_subparsers(dest="target", required=True)
    camera = targets.add_parser("camera", help="run only the camera actor")
    camera.add_argument("--seconds", type=_positive, default=30.0)
    audio = targets.add_parser("audio", help="print P(cry) and loudness per second")
    audio.add_argument("file", type=_existing_file, metavar="FILE.wav")
    vision = targets.add_parser("vision", help="print the typed Observation of one frame")
    vision.add_argument("frame", type=_existing_file, metavar="FRAME.jpg")
    vision.add_argument("--reference", type=_existing_file, metavar="REF.jpg")
    alert = targets.add_parser("alert", help="send a test notification")
    alert.add_argument("--priority", type=Priority, choices=list(Priority), required=True)

    commands.add_parser("reference", help="store one camera frame as the reference image")

    replay = commands.add_parser("replay", help="print the effects of recorded facts")
    replay.add_argument("facts", type=_existing_file, metavar="FACTS.jsonl")
    return parser


def _initial_controls(args: argparse.Namespace) -> SetControls | None:
    if args.armed is None and args.nap_mode is None:
        return None
    return SetControls(datetime.now(UTC), armed=args.armed, nap_mode=args.nap_mode)


def _cmd_run(args: argparse.Namespace) -> None:
    from babymon import app, config

    asyncio.run(app.run(config.load(), _initial_controls(args), args.record_facts))


def _cmd_probe(args: argparse.Namespace) -> None:
    from babymon import config

    settings = config.load()
    match args.target:
        case "camera":
            from babymon import camera

            asyncio.run(camera.probe(settings, args.seconds))
        case "audio":
            from babymon import audio

            asyncio.run(audio.probe(settings, args.file))
        case "vision":
            from babymon import vision

            asyncio.run(vision.probe(settings, args.frame, args.reference))
        case "alert":
            from babymon import notify

            asyncio.run(notify.probe(settings, args.priority))


def _cmd_reference(args: argparse.Namespace) -> None:
    from babymon import camera, config
    from babymon.store import Store

    settings = config.load()
    still = asyncio.run(camera.grab_still(settings))
    Store.open(settings.data_dir).set_reference(still)
    print(f"reference stored from the frame at {still.at.isoformat(timespec='seconds')}")


def _cmd_replay(args: argparse.Namespace) -> None:
    try:
        effects = replay_facts(factlog.read(args.facts))
    except ValueError as e:
        raise SystemExit(f"babymon: {e}") from None
    for at, effect in effects:
        print(format_effect(at, effect))


COMMANDS = {
    "run": _cmd_run,
    "probe": _cmd_probe,
    "reference": _cmd_reference,
    "replay": _cmd_replay,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        COMMANDS[args.command](args)
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
