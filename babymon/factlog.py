"""JSONL encoding of facts: `run --record-facts` writes it, `replay` reads it.

One JSON object per line. `"type"` names the fact class and every other key is that
class's field, with datetimes as ISO 8601 with offset and enums as their string value:

    {"type": "tick", "at": "2026-10-27T02:00:01Z"}
    {"type": "motion", "at": "...", "level": 0.03}
    {"type": "audio", "at": "...", "loudness_db": 4.1, "cry_probability": 0.08}
    {"type": "vision_done", "at": "...", "reason": "periodic", "observation": {...},
     "snapshot": "20261027-020001-periodic.jpg"}
    {"type": "vision_failed", "at": "...", "reason": "periodic", "error": "timeout"}
    {"type": "camera", "at": "...", "up": true, "detail": "connected"}
    {"type": "set_controls", "at": "...", "armed": null, "nap_mode": true}
    {"type": "delivery", "at": "...", "kind": "crying", "delivered": true}
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, get_args

from pydantic import TypeAdapter

from babymon.domain import (
    AudioSample,
    CameraStatus,
    DeliveryResult,
    Fact,
    MotionSample,
    SetControls,
    Tick,
    VisionDone,
    VisionFailed,
)

FACT_TYPES: dict[str, type] = {
    "tick": Tick,
    "motion": MotionSample,
    "audio": AudioSample,
    "vision_done": VisionDone,
    "vision_failed": VisionFailed,
    "camera": CameraStatus,
    "set_controls": SetControls,
    "delivery": DeliveryResult,
}
_NAME_OF = {cls: name for name, cls in FACT_TYPES.items()}
_ADAPTERS = {cls: TypeAdapter(cls) for cls in FACT_TYPES.values()}

assert set(FACT_TYPES.values()) == set(get_args(Fact.__value__)), "a Fact has no log name"


def encode(fact: Fact) -> str:
    body: dict[str, Any] = _ADAPTERS[type(fact)].dump_python(fact, mode="json")
    return json.dumps({"type": _NAME_OF[type(fact)], **body}, separators=(",", ":"))


def decode(line: str) -> Fact:
    """Raises ValueError for an unknown type, a malformed line or a naive timestamp."""
    raw = json.loads(line)
    if not isinstance(raw, dict) or raw.get("type") not in FACT_TYPES:
        raise ValueError(f"not a fact line: {line[:80]!r}")
    cls = FACT_TYPES[raw.pop("type")]
    fact = _ADAPTERS[cls].validate_python(raw)
    if fact.at.tzinfo is None:
        raise ValueError(f"timestamp without offset: {line[:80]!r}")
    return fact


def read(path: Path) -> Iterator[Fact]:
    """Blank lines are skipped. A bad line raises ValueError naming its line number."""
    with path.open() as f:
        for number, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield decode(line)
                except ValueError as e:
                    raise ValueError(f"{path}:{number}: {e}") from e
