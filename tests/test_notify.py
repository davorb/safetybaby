from __future__ import annotations

import asyncio
import re
from pathlib import Path

import httpx
import pytest

from babymon.domain import (
    Alert,
    AlertKind,
    AlertSent,
    DeliveryResult,
    Priority,
    Still,
)
from babymon.latest import Latest
from babymon.notify import (
    DeliveryError,
    LogOnly,
    Pushover,
    pushover_priority,
    run_alerter,
)
from babymon.store import Store
from tests.conftest import T0, at

TOKEN = "app-token-secret"
USERS = ("uQiRzpo4DXghDmr9QzzfQu27cmVRsG", "gznej3rKEVAvPUxu9vvNnqpmZpokzF")
STILL = Still(b"\xff\xd8jpeg", T0)


def alert(priority: Priority = Priority.HIGH, kind: AlertKind = AlertKind.CRYING) -> Alert:
    return Alert(kind=kind, priority=priority, at=T0, title="Baby crying", message="For 60 s")


def fields(request: httpx.Request) -> dict[str, str | bytes]:
    """Decode a multipart body into {name: value}; the attachment keeps its raw bytes."""
    content_type = request.headers["content-type"]
    assert content_type.startswith("multipart/form-data")
    boundary = content_type.split("boundary=")[1].encode()
    found: dict[str, str | bytes] = {}
    for part in request.content.split(b"--" + boundary)[1:-1]:
        head, _, body = part.partition(b"\r\n\r\n")
        body = body.removesuffix(b"\r\n")
        name = re.search(rb'name="([^"]+)"', head)[1].decode()
        found[name] = body if b"filename=" in head else body.decode()
        if b"filename=" in head:
            found[f"{name}:head"] = head.decode()
    return found


class Phone:
    """An httpx MockTransport standing in for api.pushover.net."""

    def __init__(self, *replies: httpx.Response | Exception) -> None:
        self.requests: list[httpx.Request] = []
        self._replies = list(replies)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self._replies.pop(0) if len(self._replies) > 1 else self._replies[0]
        if isinstance(reply, Exception):
            raise reply
        return reply

    def pushover(self, retry: int = 30, expire: int = 600) -> Pushover:
        self.slept: list[float] = []
        self.now = 0.0

        async def sleep(seconds: float) -> None:
            self.slept.append(seconds)
            self.now += seconds

        return Pushover(
            TOKEN,
            USERS,
            retry,
            expire,
            transport=httpx.MockTransport(self),
            sleep=sleep,
            clock=lambda: self.now,
        )


def accepted() -> httpx.Response:
    return httpx.Response(200, json={"status": 1, "request": "abc"})


class TestPushoverFields:
    @pytest.mark.parametrize(
        ("priority", "wire"),
        [(Priority.NORMAL, "0"), (Priority.HIGH, "1"), (Priority.EMERGENCY, "2")],
    )
    async def test_maps_the_domain_priority_to_pushovers_integer(self, priority, wire):
        phone = Phone(accepted())

        await phone.pushover().deliver(alert(priority), None)

        sent = fields(phone.requests[0])
        assert sent["priority"] == wire
        assert (sent["token"], sent["title"], sent["message"]) == (TOKEN, "Baby crying", "For 60 s")
        assert sent["user"] == ",".join(USERS) and " " not in sent["user"]
        assert str(phone.requests[0].url) == "https://api.pushover.net/1/messages.json"

    async def test_only_emergency_carries_retry_and_expire(self):
        phone = Phone(accepted())
        pushover = phone.pushover(retry=45, expire=900)

        await pushover.deliver(alert(Priority.HIGH), None)
        await pushover.deliver(alert(Priority.EMERGENCY), None)

        high, emergency = (fields(r) for r in phone.requests)
        assert "retry" not in high and "expire" not in high
        assert (emergency["retry"], emergency["expire"]) == ("45", "900")

    async def test_retry_and_expire_are_clamped_to_pushovers_limits(self):
        phone = Phone(accepted())

        await phone.pushover(retry=5, expire=99999).deliver(alert(Priority.EMERGENCY), None)

        sent = fields(phone.requests[0])
        assert (sent["retry"], sent["expire"]) == ("30", "10800")

    async def test_snapshot_is_attached_as_a_jpeg_file_part(self):
        phone = Phone(accepted())

        await phone.pushover().deliver(alert(), STILL)

        sent = fields(phone.requests[0])
        assert sent["attachment"] == STILL.jpeg
        assert 'filename="snapshot.jpg"' in sent["attachment:head"]
        assert "image/jpeg" in sent["attachment:head"]

    async def test_no_snapshot_means_no_attachment_but_still_multipart(self):
        phone = Phone(accepted())

        await phone.pushover().deliver(alert(), None)

        assert "attachment" not in fields(phone.requests[0])

    async def test_snapshot_over_5_mb_is_skipped(self):
        phone = Phone(accepted())
        big = Still(b"x" * (5 * 1024 * 1024 + 1), T0)
        at_limit = Still(b"x" * (5 * 1024 * 1024), T0)

        await phone.pushover().deliver(alert(), big)
        await phone.pushover().deliver(alert(), at_limit)

        assert "attachment" not in fields(phone.requests[0])
        assert "attachment" in fields(phone.requests[1])

    async def test_overlong_text_is_truncated_to_pushovers_limits(self):
        phone = Phone(accepted())
        long = alert().model_copy(update={"title": "t" * 400, "message": "m" * 2000})

        await phone.pushover().deliver(long, None)

        sent = fields(phone.requests[0])
        assert (len(sent["title"]), len(sent["message"])) == (250, 1024)

    def test_user_key_count_is_validated(self):
        with pytest.raises(ValueError):
            Pushover(TOKEN, (), 30, 600)
        with pytest.raises(ValueError):
            Pushover(TOKEN, tuple(f"u{i}" for i in range(51)), 30, 600)
        Pushover(TOKEN, tuple(f"u{i}" for i in range(50)), 30, 600)


class TestPushoverOutcomes:
    async def test_status_zero_in_a_200_is_a_failure_naming_the_errors(self):
        reply = httpx.Response(200, json={"status": 0, "errors": ["user identifier is invalid"]})
        phone = Phone(reply)

        with pytest.raises(DeliveryError, match="user identifier is invalid"):
            await phone.pushover().deliver(alert(), None)
        assert len(phone.requests) == 1

    async def test_a_400_fails_at_once_without_retrying(self):
        reply = httpx.Response(400, json={"status": 0, "errors": ["application token is invalid"]})
        phone = Phone(reply)

        with pytest.raises(DeliveryError, match="application token is invalid"):
            await phone.pushover().deliver(alert(), None)
        assert len(phone.requests) == 1

    @pytest.mark.parametrize("code", [429, 500, 503])
    async def test_429_and_5xx_retry_with_backoff_then_fail_within_60_s(self, code):
        phone = Phone(httpx.Response(code, json={"status": 0, "errors": ["busy"]}))
        pushover = phone.pushover()

        with pytest.raises(DeliveryError, match=str(code)):
            await pushover.deliver(alert(), None)

        assert len(phone.requests) > 3
        assert phone.slept[:4] == [1, 2, 4, 8]
        assert sum(phone.slept) == 60

    async def test_transport_errors_retry_then_succeed(self):
        phone = Phone(httpx.ConnectError("down"), httpx.ReadTimeout("slow"), accepted())

        await phone.pushover().deliver(alert(), None)

        assert len(phone.requests) == 3

    async def test_a_5xx_then_success_delivers(self):
        phone = Phone(httpx.Response(502, text="bad gateway"), accepted())

        await phone.pushover().deliver(alert(), None)

        assert len(phone.requests) == 2

    async def test_non_json_200_is_a_failure(self):
        phone = Phone(httpx.Response(200, text="<html>"))

        with pytest.raises(DeliveryError):
            await phone.pushover().deliver(alert(), None)

    async def test_the_token_never_appears_in_errors_or_repr(self):
        for reply in (
            httpx.Response(400, json={"status": 0, "errors": ["bad"]}),
            httpx.ConnectError(f"cannot reach with {TOKEN}"),
        ):
            phone = Phone(reply)
            pushover = phone.pushover()
            with pytest.raises(DeliveryError) as caught:
                await pushover.deliver(alert(), None)
            assert TOKEN not in str(caught.value) and TOKEN not in repr(pushover)
            assert TOKEN not in repr(caught.value.__context__)
            assert all(u not in repr(pushover) for u in USERS)


class TestLogOnly:
    async def test_logs_and_raises(self, caplog):
        with caplog.at_level("WARNING"), pytest.raises(DeliveryError, match="disabled"):
            await LogOnly().deliver(alert(), STILL)

        assert "Baby crying" in caplog.text


def test_pushover_priority_covers_every_domain_priority():
    assert {p: pushover_priority(p) for p in Priority} == {
        Priority.NORMAL: 0,
        Priority.HIGH: 1,
        Priority.EMERGENCY: 2,
    }


class Recorder:
    def __init__(self, *outcomes: bool | Exception) -> None:
        self.outcomes = list(outcomes)
        self.delivered: list[tuple[Alert, Still | None]] = []

    async def deliver(self, alert: Alert, snapshot: Still | None) -> None:
        self.delivered.append((alert, snapshot))
        outcome = self.outcomes.pop(0)
        if outcome is False:
            raise DeliveryError("Pushover HTTP 400: nope")
        if isinstance(outcome, Exception):
            raise outcome


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store.open(tmp_path)


class Alerter:
    def __init__(self, notifier: Recorder, store: Store, still: Still | None = STILL) -> None:
        self.inbox: asyncio.Queue[Alert] = asyncio.Queue()
        self.stills: Latest[Still] = Latest()
        if still:
            self.stills.set(still)
        self.facts: list = []
        self.task = asyncio.create_task(
            run_alerter(
                self.inbox,
                notifier,
                self.stills,
                store,
                self.facts.append,
                now=lambda: at(500),
            )
        )

    async def drain(self) -> None:
        await asyncio.wait_for(self.inbox.join(), 5)

    async def close(self) -> None:
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)


class TestRunAlerter:
    async def test_delivered_alert_writes_a_true_row_and_a_delivery_result(self, store: Store):
        recorder = Recorder(True)
        a = Alerter(recorder, store)
        try:
            a.inbox.put_nowait(alert(Priority.EMERGENCY, AlertKind.ON_STOMACH))
            await a.drain()
        finally:
            await a.close()

        (stored,) = store.events(at(-10), at(10))
        row = stored.event
        assert isinstance(row, AlertSent)
        assert (row.delivered, row.detail, row.at) == (True, "", T0)
        assert row.snapshot is not None
        assert store.snapshot_path(row.snapshot).read_bytes() == STILL.jpeg
        assert row.snapshot.endswith("-on_stomach.jpg")
        assert recorder.delivered == [(alert(Priority.EMERGENCY, AlertKind.ON_STOMACH), STILL)]
        assert a.facts == [DeliveryResult(at(500), AlertKind.ON_STOMACH, True)]

    async def test_failed_delivery_writes_a_false_row_with_the_error(self, store: Store):
        a = Alerter(Recorder(False), store)
        try:
            a.inbox.put_nowait(alert())
            await a.drain()
        finally:
            await a.close()

        (stored,) = store.events(at(-10), at(10))
        assert isinstance(stored.event, AlertSent)
        assert (stored.event.delivered, stored.event.detail) == (False, "Pushover HTTP 400: nope")
        assert a.facts == [DeliveryResult(at(500), AlertKind.CRYING, False)]

    async def test_cooldown_recovery_counts_only_the_delivered_row(self, store: Store):
        a = Alerter(Recorder(False, True), store)
        try:
            a.inbox.put_nowait(alert(kind=AlertKind.CRYING))
            a.inbox.put_nowait(alert(kind=AlertKind.HEAD_LEFT))
            await a.drain()
        finally:
            await a.close()

        assert set(store.restore(at(60), 600).delivered) == {AlertKind.HEAD_LEFT}

    async def test_no_still_means_no_snapshot_and_delivery_still_happens(self, store: Store):
        recorder = Recorder(True)
        a = Alerter(recorder, store, still=None)
        try:
            a.inbox.put_nowait(alert())
            await a.drain()
        finally:
            await a.close()

        assert recorder.delivered[0][1] is None
        (stored,) = store.events(at(-10), at(10))
        assert isinstance(stored.event, AlertSent) and stored.event.snapshot is None

    async def test_alerts_are_delivered_in_order_and_the_loop_survives_failures(self, store: Store):
        recorder = Recorder(False, True, True)
        a = Alerter(recorder, store)
        kinds = [AlertKind.FACE_COVERED, AlertKind.CRYING, AlertKind.CAMERA_OFFLINE]
        try:
            for kind in kinds:
                a.inbox.put_nowait(alert(kind=kind))
            await a.drain()
        finally:
            await a.close()

        assert [d[0].kind for d in recorder.delivered] == kinds
        assert [f.delivered for f in a.facts] == [False, True, True]

    async def test_log_only_records_undelivered_rows(self, store: Store):
        a = Alerter(Recorder(), store)
        a.task.cancel()
        await asyncio.gather(a.task, return_exceptions=True)
        a.task = asyncio.create_task(
            run_alerter(a.inbox, LogOnly(), a.stills, store, a.facts.append, now=lambda: at(1))
        )
        try:
            a.inbox.put_nowait(alert())
            await a.drain()
        finally:
            await a.close()

        (stored,) = store.events(at(-10), at(10))
        assert isinstance(stored.event, AlertSent) and stored.event.delivered is False
        assert "disabled" in stored.event.detail
