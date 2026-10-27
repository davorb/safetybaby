"""Vision: Still (+ reference Still) -> Observation.

VisionBackend is the swap point. ClaudeVision is the only module that imports
`anthropic`; its request and response shapes (_WireObservation, content blocks) stay
private here and are parsed into domain.Observation before leaving.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Never, Protocol

from pydantic import BaseModel, Field

from babymon.domain import (
    BodyPosition,
    Eyes,
    Fact,
    HeadDirection,
    Observation,
    RequestVision,
    Still,
)
from babymon.latest import Latest
from babymon.store import Store


class VisionError(Exception):
    """Any failure to produce an Observation: transport, refusal, unparseable output."""


class VisionBackend(Protocol):
    async def observe(self, frame: Still, reference: Still | None) -> Observation:
        """Raise VisionError on failure. Never return a guessed Observation; uncertainty
        belongs in the UNKNOWN members and in confidence."""
        ...


class _WireObservation(BaseModel):
    """Structured-output schema sent to Claude. Field descriptions carry the rubric.
    Kept separate from Observation so prompt wording can change without touching the
    domain type."""

    body: BodyPosition = Field(description="...")
    head: HeadDirection = Field(
        description="Direction the face points, from the baby's own left/right."
    )
    face_covered: bool = Field(description="...")
    in_crib: bool
    eyes: Eyes
    confidence: float = Field(ge=0.0, le=1.0)


SYSTEM_PROMPT = """You watch a 4-month-old in a sleep sack, side/top view, often infrared.
The first image, when present, is a reference of the correct, safe position. ..."""


class ClaudeVision:
    def __init__(self, api_key: str, model: str, max_tokens: int = 512) -> None:
        """Holds an `anthropic.Anthropic` client. The sync client runs in a worker
        thread so only the SDK names the brief allows are used."""
        raise NotImplementedError

    async def observe(self, frame: Still, reference: Still | None) -> Observation:
        """await to_thread(client.messages.parse, model=..., system=SYSTEM_PROMPT,
        messages=[{role: user, content: [base64 image block(reference)?,
        base64 image block(frame), text]}], output_format=_WireObservation)
        -> parsed _WireObservation -> Observation. Any exception -> VisionError."""
        raise NotImplementedError


async def run_vision(
    backend: VisionBackend,
    requests: Latest[RequestVision],
    stills: Latest[Still],
    store: Store,
    publish: Callable[[Fact], None],
    max_still_age_s: float = 5.0,
) -> Never:
    """One call in flight at a time.

    for each newer request:
        still = stills.get(); stale or missing -> publish VisionFailed
        reference = store.reference()                       (file on disk, may be None)
        observation = await backend.observe(still, reference)   (VisionError -> VisionFailed)
        snapshot = store.save_snapshot(still, label="vision")
        publish(VisionDone(at=still.at, reason=req.reason, observation, snapshot))
    `at` is the still's capture time, not the response time, so streak timing reflects
    when the baby was seen.
    """
    raise NotImplementedError
