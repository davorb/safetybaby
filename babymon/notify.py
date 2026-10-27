"""Alerter: Alert -> phones. Notifier is the swap point; Pushover is the only module
that knows Pushover's form fields."""

from __future__ import annotations

import asyncio
from typing import Never, Protocol

from babymon.domain import Alert, Still
from babymon.latest import Latest
from babymon.store import Store


class Notifier(Protocol):
    async def deliver(self, alert: Alert, snapshot: Still | None) -> None:
        """Raise DeliveryError when no recipient accepted it after retries."""
        ...


class DeliveryError(Exception):
    pass


class Pushover:
    """POST https://api.pushover.net/1/messages.json, multipart, once per user key.

    Priority.EMERGENCY adds retry=emergency_retry_s (30) and expire=emergency_expire_s
    (600); the domain never sees those fields. Snapshot goes as `attachment`.
    Transport errors retry with backoff for up to 60 s per alert.
    """

    def __init__(
        self, app_token: str, user_keys: tuple[str, ...], emergency_retry_s: int, emergency_expire_s: int
    ) -> None:
        raise NotImplementedError

    async def deliver(self, alert: Alert, snapshot: Still | None) -> None:
        raise NotImplementedError


class LogOnly:
    """Used when Pushover keys are absent. Logs, then raises DeliveryError so the event
    row records delivered=False and the dashboard shows push as disabled."""

    async def deliver(self, alert: Alert, snapshot: Still | None) -> None:
        raise NotImplementedError


async def run_alerter(
    inbox: asyncio.Queue[Alert], notifier: Notifier, stills: Latest[Still], store: Store
) -> Never:
    """Sequential, in decision order.

    for alert in inbox:
        still = stills.get()                                   (latest frame, may be None)
        snapshot = store.save_snapshot(still, label=alert.kind) if still
        try deliver -> delivered=True; except DeliveryError -> delivered=False
        store.append(AlertSent(at=alert.at, alert, snapshot, delivered, detail))
    The row is written after the attempt so cooldown recovery counts only alerts that
    reached a phone.
    """
    raise NotImplementedError
