from __future__ import annotations

import asyncio
import base64
import json
from datetime import timedelta
from pathlib import Path

import anthropic
import httpx2
import pytest
from pydantic import SecretStr

from babymon.config import VisionConfig
from babymon.domain import (
    BodyPosition,
    Eyes,
    HeadDirection,
    Observation,
    RequestVision,
    Still,
    VisionDone,
    VisionFailed,
    VisionReason,
)
from babymon.latest import Latest
from babymon.store import Store
from babymon.vision import ClaudeVision, VisionError, probe_vision, run_vision, system_prompt
from tests.conftest import T0, at, observation

SECRET = "sk-ant-secret-key-do-not-leak"
FRAME = Still(b"\xff\xd8frame", T0)
REFERENCE = Still(b"\xff\xd8reference", T0 - timedelta(days=1))

READING = {
    "body": "side_left",
    "head": "left",
    "face_covered": False,
    "in_crib": True,
    "eyes": "closed",
    "confidence": 0.82,
}


def message(text: str, stop_reason: str = "end_turn") -> dict:
    return {
        "id": "msg_01ABC",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 1800, "output_tokens": 60},
    }


class Server:
    """An httpx2 MockTransport standing in for api.anthropic.com, recording requests."""

    def __init__(self, reply) -> None:
        self.requests: list[httpx2.Request] = []
        self._reply = reply

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self._reply(request)

    @property
    def body(self) -> dict:
        return json.loads(self.requests[-1].content)

    def vision(self, api_key: str = SECRET) -> ClaudeVision:
        client = anthropic.Anthropic(
            api_key=api_key,
            max_retries=0,
            http_client=httpx2.Client(transport=httpx2.MockTransport(self)),
        )
        return ClaudeVision(api_key, "claude-opus-5-5", client=client)


def ok(payload: dict | None = None, **kw) -> Server:
    return Server(
        lambda _: httpx2.Response(200, json=message(json.dumps(payload or READING), **kw))
    )


def status(code: int, error_type: str, text: str = "nope") -> Server:
    body = {"type": "error", "error": {"type": error_type, "message": text}}
    return Server(lambda _: httpx2.Response(code, json=body))


class TestClaudeVision:
    async def test_reading_is_mapped_into_the_domain_observation(self):
        server = ok()

        result = await server.vision().observe(FRAME, None)

        assert result == Observation(
            body=BodyPosition.SIDE_LEFT,
            head=HeadDirection.LEFT,
            face_covered=False,
            in_crib=True,
            eyes=Eyes.CLOSED,
            confidence=0.82,
        )

    async def test_request_carries_model_limit_schema_and_no_temperature(self):
        server = ok()

        await server.vision().observe(FRAME, None)

        body = server.body
        assert (body["model"], body["max_tokens"]) == ("claude-opus-5-5", 4096)
        assert "temperature" not in body
        assert body["output_config"]["effort"] == "low"
        schema = body["output_config"]["format"]["schema"]
        assert set(schema["properties"]) == set(READING)
        assert SECRET in server.requests[-1].headers["authorization"]

    async def test_frame_alone_is_one_image_then_the_instruction(self):
        server = ok()

        await server.vision().observe(FRAME, None)

        blocks = server.body["messages"][0]["content"]
        assert [b["type"] for b in blocks] == ["image", "text"]
        assert blocks[0]["source"]["media_type"] == "image/jpeg"
        assert base64.b64decode(blocks[0]["source"]["data"]) == FRAME.jpeg

    async def test_reference_goes_first_with_a_label_then_the_frame(self):
        server = ok()

        await server.vision().observe(FRAME, REFERENCE)

        blocks = server.body["messages"][0]["content"]
        assert [b["type"] for b in blocks] == ["text", "image", "image", "text"]
        decoded = [base64.b64decode(b["source"]["data"]) for b in blocks if b["type"] == "image"]
        assert decoded == [REFERENCE.jpeg, FRAME.jpeg]

    async def test_system_prompt_mentions_the_reference_only_when_one_is_passed(self):
        server = ok()
        vision = server.vision()

        await vision.observe(FRAME, None)
        without = server.body["system"]
        await vision.observe(FRAME, REFERENCE)
        with_reference = server.body["system"]

        assert "reference" not in without.lower()
        assert "reference" in with_reference.lower()
        assert without == system_prompt(False) and with_reference == system_prompt(True)

    def test_prompt_states_the_babys_own_left_and_the_infrared_caveat(self):
        prompt = system_prompt(False)

        assert "4-month-old" in prompt and "sleep sack" in prompt
        assert "infrared" in prompt
        assert "baby's left appears on the viewer's right" in prompt
        assert "nose or mouth" in prompt

    @pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
    async def test_a_stop_other_than_end_turn_is_a_vision_error(self, stop_reason: str):
        server = ok(stop_reason=stop_reason)

        with pytest.raises(VisionError, match=stop_reason):
            await server.vision().observe(FRAME, None)

    @pytest.mark.parametrize(
        "text",
        [
            "not json at all",
            json.dumps({**READING, "body": "upside_down"}),
            json.dumps({**READING, "confidence": 1.7}),
            json.dumps({"body": "back"}),
        ],
    )
    async def test_output_that_breaks_the_schema_is_a_vision_error(self, text: str):
        server = Server(lambda _: httpx2.Response(200, json=message(text)))

        with pytest.raises(VisionError):
            await server.vision().observe(FRAME, None)

    @pytest.mark.parametrize(
        ("code", "error_type", "expected"),
        [
            (401, "authentication_error", "API key"),
            (403, "permission_error", "API key"),
            (404, "not_found_error", "not found"),
            (400, "invalid_request_error", "bad request"),
            (429, "rate_limit_error", "rate limited"),
            (500, "api_error", "HTTP 500"),
            (529, "overloaded_error", "HTTP 529"),
        ],
    )
    async def test_http_errors_become_vision_errors(self, code, error_type, expected):
        with pytest.raises(VisionError, match=expected):
            await status(code, error_type).vision().observe(FRAME, None)

    async def test_connection_and_timeout_failures_become_vision_errors(self):
        def refuse(request):
            raise httpx2.ConnectError("boom", request=request)

        def slow(request):
            raise httpx2.ReadTimeout("slow", request=request)

        with pytest.raises(VisionError, match="reach"):
            await Server(refuse).vision().observe(FRAME, None)
        with pytest.raises(VisionError, match="timed out"):
            await Server(slow).vision().observe(FRAME, None)

    async def test_the_key_never_appears_in_an_error_or_a_repr(self):
        servers = [
            status(401, "authentication_error", f"bad key {SECRET}"),
            ok(stop_reason="refusal"),
        ]
        for server in servers:
            vision = server.vision()
            with pytest.raises(VisionError) as caught:
                await vision.observe(FRAME, None)
            assert SECRET not in str(caught.value) and SECRET not in repr(caught.value)
            assert SECRET not in repr(vision)
            assert caught.value.__cause__ is None and caught.value.__context__ is None


class ScriptedBackend:
    def __init__(self, *results: Observation | VisionError) -> None:
        self.results = list(results)
        self.calls: list[tuple[Still, Still | None]] = []
        self.gate = asyncio.Event()
        self.gate.set()
        self.in_flight = 0
        self.max_in_flight = 0

    async def observe(self, frame: Still, reference: Still | None) -> Observation:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        self.calls.append((frame, reference))
        try:
            await self.gate.wait()
            result = self.results.pop(0)
            if isinstance(result, VisionError):
                raise result
            return result
        finally:
            self.in_flight -= 1


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store.open(tmp_path)


class Harness:
    def __init__(self, backend: ScriptedBackend, store: Store, clock_s: float = 2.0) -> None:
        self.requests: Latest[RequestVision] = Latest()
        self.stills: Latest[Still] = Latest()
        self.facts: list = []
        self.clock_s = clock_s
        self.backend = backend
        self.task = asyncio.create_task(
            run_vision(
                backend,
                self.requests,
                self.stills,
                store,
                self.facts.append,
                now=lambda: at(self.clock_s),
            )
        )

    async def settle(self, until=None) -> None:
        """Worker threads run in real time, so wait on the condition, not on loop turns."""
        if until is None:
            await asyncio.sleep(0.1)
            return
        for _ in range(1000):
            if until():
                return
            await asyncio.sleep(0.005)
        raise AssertionError("condition not reached")

    async def close(self) -> None:
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)


@pytest.fixture
async def harness(store: Store):
    made: list[Harness] = []

    def make(*results, clock_s: float = 2.0) -> Harness:
        h = Harness(ScriptedBackend(*results), store, clock_s)
        made.append(h)
        return h

    yield make
    for h in made:
        await h.close()


class TestRunVision:
    async def test_publishes_done_at_the_stills_capture_time_and_saves_the_snapshot(
        self, harness, store
    ):
        h = harness(observation())
        h.stills.set(Still(b"jpeg", at(0)))

        h.requests.set(RequestVision(at(1), VisionReason.CONFIRM))
        await h.settle(lambda: h.facts)

        (fact,) = h.facts
        assert isinstance(fact, VisionDone)
        assert (fact.at, fact.reason, fact.observation) == (
            at(0),
            VisionReason.CONFIRM,
            observation(),
        )
        assert store.snapshot_path(fact.snapshot).read_bytes() == b"jpeg"

    async def test_stale_still_is_a_failure_and_the_backend_is_not_called(self, harness):
        h = harness(observation(), clock_s=6.5)
        h.stills.set(Still(b"jpeg", at(0)))

        h.requests.set(RequestVision(at(6), VisionReason.PERIODIC))
        await h.settle(lambda: h.facts)

        (fact,) = h.facts
        assert isinstance(fact, VisionFailed) and fact.reason is VisionReason.PERIODIC
        assert h.backend.calls == []

    async def test_missing_still_is_a_failure(self, harness):
        h = harness(observation())

        h.requests.set(RequestVision(at(1), VisionReason.PERIODIC))
        await h.settle(lambda: h.facts)

        assert isinstance(h.facts[0], VisionFailed)

    async def test_backend_error_is_a_failure_and_the_loop_survives(self, harness):
        h = harness(VisionError("rate limited (HTTP 429)"), observation())
        h.stills.set(Still(b"jpeg", at(0)))

        h.requests.set(RequestVision(at(1), VisionReason.PERIODIC))
        await h.settle(lambda: len(h.facts) == 1)
        h.requests.set(RequestVision(at(2), VisionReason.PERIODIC))
        await h.settle(lambda: len(h.facts) == 2)

        assert isinstance(h.facts[0], VisionFailed) and "429" in h.facts[0].error
        assert isinstance(h.facts[1], VisionDone)

    async def test_failure_leaves_no_snapshot_behind(self, harness, store):
        h = harness(VisionError("x"))
        h.stills.set(Still(b"jpeg", at(0)))

        h.requests.set(RequestVision(at(1), VisionReason.PERIODIC))
        await h.settle(lambda: h.facts)

        assert (
            list(
                store.snapshot_path(
                    __import__("babymon.domain").domain.SnapshotRef("x")
                ).parent.iterdir()
            )
            == []
        )

    async def test_reference_from_the_store_is_passed_when_present(self, harness, store):
        store.set_reference(REFERENCE)
        h = harness(observation())
        h.stills.set(Still(b"jpeg", at(0)))

        h.requests.set(RequestVision(at(1), VisionReason.PERIODIC))
        await h.settle(lambda: h.facts)

        (_, reference) = h.backend.calls[0]
        assert reference is not None and reference.jpeg == REFERENCE.jpeg

    async def test_one_call_in_flight_and_only_the_newest_pending_request_runs(self, harness):
        h = harness(observation(), observation(), observation())
        h.backend.gate.clear()
        h.stills.set(Still(b"jpeg", at(0)))

        h.requests.set(RequestVision(at(1), VisionReason.PERIODIC))
        await h.settle(lambda: h.backend.calls)
        h.requests.set(RequestVision(at(1.1), VisionReason.PERIODIC))
        h.requests.set(RequestVision(at(1.2), VisionReason.CONFIRM))
        await h.settle()
        assert len(h.backend.calls) == 1

        h.backend.gate.set()
        await h.settle(lambda: len(h.facts) == 2)

        assert h.backend.max_in_flight == 1
        assert [f.reason for f in h.facts] == [VisionReason.PERIODIC, VisionReason.CONFIRM]


class TestProbe:
    async def test_without_a_key_it_says_vision_is_disabled(self, tmp_path: Path, capsys):
        frame = tmp_path / "crib.jpg"
        frame.write_bytes(b"x")

        code = await probe_vision(frame, None, None)

        assert code == 1
        assert "vision disabled: no API key" in capsys.readouterr().err

    def test_config_keeps_the_key_out_of_its_repr(self):
        assert SECRET not in repr(VisionConfig(api_key=SecretStr(SECRET)))
