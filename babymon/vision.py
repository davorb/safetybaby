"""Vision: Still (+ reference Still) -> Observation.

VisionBackend is the swap point. ClaudeVision is the only module that imports
`anthropic`; its request and response shapes (_WireObservation, content blocks) stay
private here and are parsed into domain.Observation before leaving.
"""

from __future__ import annotations

import asyncio
import base64
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Never, Protocol

import anthropic
from pydantic import BaseModel, Field

from babymon.config import VisionConfig
from babymon.domain import (
    BodyPosition,
    Eyes,
    Fact,
    HeadDirection,
    Instant,
    Observation,
    RequestVision,
    Still,
    VisionDone,
    VisionFailed,
)
from babymon.latest import Latest
from babymon.store import Store

REQUEST_TIMEOUT_S = 30.0


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

    body: BodyPosition = Field(
        description=(
            "How the baby lies. back: face up. side_left or side_right: on the baby's own "
            "left or right side. stomach: face down. unknown: cannot tell."
        )
    )
    head: HeadDirection = Field(
        description=(
            "Direction the face points, from the baby's own left and right. center: face "
            "straight up or toward the ceiling. unknown: cannot tell."
        )
    )
    face_covered: bool = Field(
        description="True when the nose or mouth is hidden by fabric, a hand, or an object."
    )
    in_crib: bool = Field(
        description="False when the baby is absent from the crib or is being held."
    )
    eyes: Eyes = Field(description="open, closed, or unknown when the eyes cannot be seen.")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "How sure you are of the whole reading, 0 to 1. Use a low value when the image "
            "is dark, blurred, or the baby is partly out of view."
        ),
    )


_BASE_PROMPT = """You watch a 4-month-old baby in a crib through a camera. The baby wears a \
sleep sack and is placed on the back or at an angle. The image may be infrared and grayscale.

Report what you see for the current frame only.

Left and right are the baby's own, not the image's. When the baby faces the camera, the \
baby's left appears on the viewer's right. Work out where the head and feet are and which way \
the baby is oriented, then reason from that orientation. Do not read left or right off the \
image edges.

face_covered means the nose or mouth is hidden by fabric, a hand, or an object. in_crib is \
false when the baby is absent or is being held.

When you cannot tell, use the unknown value and a low confidence. Do not guess."""

_REFERENCE_PROMPT = """

The first image is the parents' reference of the correct, safe position. The second image is \
the current frame. Use the reference to judge what the correct position looks like from this \
camera. Describe only the current frame."""


def system_prompt(has_reference: bool) -> str:
    return _BASE_PROMPT + (_REFERENCE_PROMPT if has_reference else "")


def _image(still: Still) -> dict[str, object]:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": base64.standard_b64encode(still.jpeg).decode("ascii"),
        },
    }


def _text(text: str) -> dict[str, object]:
    return {"type": "text", "text": text}


def _content(frame: Still, reference: Still | None) -> list[dict[str, object]]:
    blocks: list[dict[str, object]] = []
    if reference is not None:
        blocks += [_text("Reference image of the correct position:"), _image(reference)]
    blocks += [_image(frame), _text("Assess the current frame.")]
    return blocks


def _explain(error: Exception, model: str) -> str:
    """One line per failure, built from the exception type and status. Never from str(error)
    for transport errors, and never including a credential."""
    match error:
        case anthropic.AuthenticationError() | anthropic.PermissionDeniedError():
            return f"Anthropic rejected the API key (HTTP {error.status_code})"
        case anthropic.NotFoundError():
            return f"model {model!r} not found or not available"
        case anthropic.BadRequestError():
            return f"bad request (HTTP 400): {error.message}"
        case anthropic.RateLimitError():
            return "rate limited (HTTP 429)"
        case anthropic.APIStatusError():
            return f"Anthropic API error (HTTP {error.status_code})"
        case anthropic.APITimeoutError():
            return "request to Anthropic timed out"
        case anthropic.APIConnectionError():
            return "could not reach Anthropic"
        case anthropic.AnthropicError():
            return f"Anthropic SDK error ({type(error).__name__})"
        case _:
            return "model output did not match the observation schema"


class ClaudeVision:
    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int = 4096,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        """Holds an `anthropic.Anthropic` client. The sync client runs in a worker
        thread so only the SDK names the brief allows are used. `client` is the test seam."""
        self._client = client or anthropic.Anthropic(api_key=api_key, timeout=REQUEST_TIMEOUT_S)
        self._model = model
        self._max_tokens = max_tokens

    def __repr__(self) -> str:
        return f"ClaudeVision(model={self._model!r})"

    async def observe(self, frame: Still, reference: Still | None) -> Observation:
        """Reference (with a label) first when present, then the frame, then the instruction.
        Every failure, including a refusal or a truncated answer, becomes VisionError."""
        failure: str | None = None
        try:
            response = await asyncio.to_thread(
                self._client.messages.parse,
                model=self._model,
                max_tokens=self._max_tokens,
                output_config={"effort": "low"},
                system=system_prompt(reference is not None),
                messages=[{"role": "user", "content": _content(frame, reference)}],
                output_format=_WireObservation,
            )
        except (anthropic.AnthropicError, ValueError) as e:
            failure = _explain(e, self._model)
        if failure is not None:
            # raised outside the except block so the SDK exception is not chained
            raise VisionError(failure)

        if response.stop_reason != "end_turn":
            raise VisionError(f"model stopped with {response.stop_reason!r}, no observation")
        wire = response.parsed_output
        if wire is None:
            raise VisionError("model returned no structured output")
        return Observation.model_validate(wire.model_dump())


async def run_vision(
    backend: VisionBackend,
    requests: Latest[RequestVision],
    stills: Latest[Still],
    store: Store,
    publish: Callable[[Fact], None],
    max_still_age_s: float = 5.0,
    now: Callable[[], Instant] = lambda: datetime.now(UTC),
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
    version = 0
    while True:
        version, request = await requests.newer_than(version)
        still = stills.get()
        if still is None:
            publish(VisionFailed(now(), request.reason, "no frame available"))
            continue
        age_s = (now() - still.at).total_seconds()
        if age_s > max_still_age_s:
            publish(VisionFailed(now(), request.reason, f"latest frame is {age_s:.0f} s old"))
            continue
        reference = await asyncio.to_thread(store.reference)
        try:
            observation = await backend.observe(still, reference)
        except VisionError as e:
            publish(VisionFailed(now(), request.reason, str(e)))
            continue
        snapshot = await asyncio.to_thread(store.save_snapshot, still, "vision")
        publish(VisionDone(still.at, request.reason, observation, snapshot))


async def probe_vision(frame: Path, reference: Path | None, config: VisionConfig | None) -> int:
    """`babymon probe vision FRAME.jpg [--reference REF.jpg]`. Prints the Observation as
    JSON and returns the exit code."""
    if config is None:
        print("vision disabled: no API key", file=sys.stderr)
        return 1
    backend = ClaudeVision(config.api_key.get_secret_value(), config.model)
    captured = datetime.now(UTC)
    ref = Still(reference.read_bytes(), captured) if reference else None
    try:
        observation = await backend.observe(Still(frame.read_bytes(), captured), ref)
    except VisionError as e:
        print(f"vision failed: {e}", file=sys.stderr)
        return 1
    print(observation.model_dump_json(indent=2))
    return 0
