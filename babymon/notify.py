"""Alerter: Alert -> phones. Notifier is the swap point; Pushover is the only module
that knows Pushover's form fields."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Never, Protocol, assert_never

import httpx

from babymon.domain import (
    Alert,
    AlertSent,
    DeliveryResult,
    Fact,
    Instant,
    Priority,
    SnapshotRef,
    Still,
)
from babymon.latest import Latest
from babymon.store import Store

log = logging.getLogger(__name__)

PUSHOVER_URL = "https://api.pushover.net/1/messages.json"
MAX_USER_KEYS = 50
MIN_RETRY_S = 30
MAX_EXPIRE_S = 10800
MAX_ATTACHMENT_BYTES = 5 * 1024 * 1024
MAX_TITLE_CHARS = 250
MAX_MESSAGE_CHARS = 1024
RETRY_BUDGET_S = 60.0
FIRST_BACKOFF_S = 1.0
REQUEST_TIMEOUT_S = 10.0


class Notifier(Protocol):
    async def deliver(self, alert: Alert, snapshot: Still | None) -> None:
        """Raise DeliveryError when no recipient accepted it after retries."""
        ...


class DeliveryError(Exception):
    pass


def pushover_priority(priority: Priority) -> int:
    match priority:
        case Priority.NORMAL:
            return 0
        case Priority.HIGH:
            return 1
        case Priority.EMERGENCY:
            return 2
        case _:
            assert_never(priority)


class _Retryable(Exception):
    pass


class Pushover:
    """POST https://api.pushover.net/1/messages.json, multipart, one request addressed to
    every user key (comma-separated).

    Priority maps to Pushover's scale (NORMAL 0, HIGH 1, EMERGENCY 2). EMERGENCY adds
    retry (at least 30 s) and expire (at most 10800 s); the domain never sees those fields.
    Snapshot goes as `attachment`. Transport errors, 429 and 5xx retry with backoff for up
    to 60 s per alert; other 4xx fail at once.
    """

    def __init__(
        self,
        app_token: str,
        user_keys: tuple[str, ...],
        emergency_retry_s: int,
        emergency_expire_s: int,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not user_keys:
            raise ValueError("Pushover needs at least one user key")
        if len(user_keys) > MAX_USER_KEYS:
            raise ValueError(f"Pushover accepts at most {MAX_USER_KEYS} user keys")
        self._token = app_token
        self._users = ",".join(user_keys)
        self._retry_s = max(emergency_retry_s, MIN_RETRY_S)
        self._expire_s = min(emergency_expire_s, MAX_EXPIRE_S)
        self._transport = transport
        self._sleep = sleep
        self._clock = clock

    def __repr__(self) -> str:
        return f"Pushover(users={len(self._users.split(','))})"

    def _parts(self, alert: Alert, snapshot: Still | None) -> dict[str, tuple]:
        fields = {
            "token": self._token,
            "user": self._users,
            "title": alert.title[:MAX_TITLE_CHARS],
            "message": alert.message[:MAX_MESSAGE_CHARS],
            "priority": str(pushover_priority(alert.priority)),
        }
        if alert.priority is Priority.EMERGENCY:
            fields["retry"] = str(self._retry_s)
            fields["expire"] = str(self._expire_s)
        parts: dict[str, tuple] = {name: (None, value) for name, value in fields.items()}
        if snapshot is not None and len(snapshot.jpeg) <= MAX_ATTACHMENT_BYTES:
            parts["attachment"] = ("snapshot.jpg", snapshot.jpeg, "image/jpeg")
        return parts

    async def deliver(self, alert: Alert, snapshot: Still | None) -> None:
        parts = self._parts(alert, snapshot)
        deadline = self._clock() + RETRY_BUDGET_S
        backoff = FIRST_BACKOFF_S
        async with httpx.AsyncClient(transport=self._transport, timeout=REQUEST_TIMEOUT_S) as http:
            while True:
                try:
                    await self._post(http, parts)
                    return
                except _Retryable as failure:
                    remaining = deadline - self._clock()
                    if remaining <= 0:
                        raise DeliveryError(f"{failure} (gave up after retries)") from None
                    await self._sleep(min(backoff, remaining))
                    backoff *= 2

    async def _post(self, http: httpx.AsyncClient, parts: dict[str, tuple]) -> None:
        failure: str | None = None
        try:
            response = await http.post(PUSHOVER_URL, files=parts)
        except httpx.HTTPError as e:
            failure = f"transport error ({type(e).__name__})"
        if failure is not None:
            raise _Retryable(failure)

        if response.status_code >= 500 or response.status_code == 429:
            raise _Retryable(f"Pushover HTTP {response.status_code}")
        try:
            body = response.json()
        except ValueError:
            body = None
        if response.status_code == 200 and isinstance(body, dict) and body.get("status") == 1:
            return
        errors = body.get("errors") if isinstance(body, dict) else None
        detail = "; ".join(map(str, errors)) if isinstance(errors, list) and errors else "no detail"
        raise DeliveryError(f"Pushover HTTP {response.status_code}: {detail}")


class LogOnly:
    """Used when Pushover keys are absent. Logs, then raises DeliveryError so the event
    row records delivered=False and the dashboard shows push as disabled."""

    async def deliver(self, alert: Alert, snapshot: Still | None) -> None:
        log.warning("push disabled, alert not delivered: [%s] %s", alert.kind, alert.title)
        raise DeliveryError("push is disabled: no Pushover keys configured")


async def run_alerter(
    inbox: asyncio.Queue[Alert],
    notifier: Notifier,
    stills: Latest[Still],
    store: Store,
    publish: Callable[[Fact], None],
    now: Callable[[], Instant] = lambda: datetime.now(UTC),
) -> Never:
    """Sequential, in decision order.

    for alert in inbox:
        still = stills.get()                                   (latest frame, may be None)
        snapshot = store.save_snapshot(still, label=alert.kind) if still
        try deliver -> delivered=True; except DeliveryError -> delivered=False
        store.append(AlertSent(at=alert.at, alert, snapshot, delivered, detail))
        publish(DeliveryResult(at=now, kind=alert.kind, delivered))
    The row is written after the attempt so cooldown recovery counts only alerts that
    reached a phone. DeliveryResult tells the brain the same thing in-process.
    """
    while True:
        alert = await inbox.get()
        still = stills.get()
        ref: SnapshotRef | None = None
        if still is not None:
            ref = await asyncio.to_thread(store.save_snapshot, still, alert.kind)
        delivered, detail = True, ""
        try:
            await notifier.deliver(alert, still)
        except DeliveryError as e:
            delivered, detail = False, str(e)
            log.warning("alert %s not delivered: %s", alert.kind, detail)
        sent = AlertSent(at=alert.at, alert=alert, snapshot=ref, delivered=delivered, detail=detail)
        await asyncio.to_thread(store.append, sent)
        publish(DeliveryResult(now(), alert.kind, delivered))
        inbox.task_done()
