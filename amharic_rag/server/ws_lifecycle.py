#!/usr/bin/env python3
"""
Per-WebSocket request lifecycle: IDs, overlap protection, cancel, stale filtering.

One connection may only run one pipeline at a time. New audio/text while busy is
rejected with ``code=busy``. Clients may send ``{"type":"cancel"}`` to stop the
active request cooperatively. Every outbound event is stamped with ``request_id``
so the browser can ignore stale events after a reconnect or race.
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any, Awaitable, Callable, Dict, Optional

from server.ws_streaming import sanitize_event, stream_sync_generator

SendJson = Callable[[Dict[str, Any]], Awaitable[None]]
GenFactory = Callable[[], Any]


class WSRequestLifecycle:
    """Mutable per-connection request state (single-threaded asyncio use)."""

    def __init__(self) -> None:
        self.active_request_id: Optional[str] = None
        self.cancel_event: Optional[asyncio.Event] = None

    @property
    def is_busy(self) -> bool:
        return self.active_request_id is not None

    @staticmethod
    def new_request_id(client_id: Optional[str] = None) -> str:
        if client_id and isinstance(client_id, str) and client_id.strip():
            return client_id.strip()
        return uuid.uuid4().hex[:12]

    def try_begin(self, request_id: str) -> tuple[bool, Optional[str]]:
        """
        Attempt to start ``request_id``.

        Returns ``(True, None)`` on success, or ``(False, active_id)`` if busy.
        """
        if self.active_request_id is not None:
            return False, self.active_request_id
        self.active_request_id = request_id
        self.cancel_event = asyncio.Event()
        return True, None

    def is_current(self, request_id: str) -> bool:
        return self.active_request_id == request_id

    def request_cancel(self, request_id: Optional[str] = None) -> bool:
        """
        Signal cooperative cancel for the active request.

        If ``request_id`` is given and does not match the active request, no-op.
        Returns True when a cancel signal was set.
        """
        if self.active_request_id is None or self.cancel_event is None:
            return False
        if request_id is not None and request_id != self.active_request_id:
            return False
        self.cancel_event.set()
        return True

    def complete(self, request_id: str) -> None:
        """Clear active state when ``request_id`` finishes (success, error, or cancel)."""
        if self.active_request_id == request_id:
            self.active_request_id = None
            self.cancel_event = None

    def on_disconnect(self) -> None:
        """Cancel any in-flight work when the socket drops."""
        self.request_cancel()


def busy_error_payload(
    *,
    active_request_id: str,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "type": "error",
        "code": "busy",
        "message": "A request is already in progress on this connection",
        "active_request_id": active_request_id,
    }
    if request_id is not None:
        payload["request_id"] = request_id
    return payload


async def drive_pipeline_events(
    *,
    request_id: str,
    lifecycle: WSRequestLifecycle,
    gen_factory: GenFactory,
    send_json: SendJson,
    timeout_s: Optional[float] = None,
) -> None:
    """
    Run the sync pipeline off-loop, stamp events with ``request_id``, and stop
    sending when cancelled, superseded, or ``timeout_s`` elapses.
    """
    from server.config import PIPELINE_TIMEOUT_S

    if timeout_s is None:
        timeout_s = PIPELINE_TIMEOUT_S

    cancel_event = lifecycle.cancel_event
    if cancel_event is None:
        raise RuntimeError("drive_pipeline_events requires try_begin() first")

    try:
        await send_json({"type": "accepted", "request_id": request_id})

        stopped_early = False

        async def _pump() -> None:
            nonlocal stopped_early
            async for ev in stream_sync_generator(
                gen_factory, cancel_event=cancel_event
            ):
                if cancel_event.is_set() or not lifecycle.is_current(request_id):
                    stopped_early = True
                    break
                payload = sanitize_event(ev)
                payload["request_id"] = request_id
                await send_json(payload)
                await asyncio.sleep(0)

        try:
            await asyncio.wait_for(_pump(), timeout=timeout_s)
        except asyncio.TimeoutError:
            cancel_event.set()
            if lifecycle.is_current(request_id):
                await send_json(
                    {
                        "type": "error",
                        "code": "timeout",
                        "message": f"Pipeline exceeded {timeout_s:.0f}s",
                        "request_id": request_id,
                    }
                )
            return

        if lifecycle.is_current(request_id) and (
            stopped_early or cancel_event.is_set()
        ):
            await send_json({"type": "cancelled", "request_id": request_id})
    except Exception as exc:
        if lifecycle.is_current(request_id):
            await send_json(
                {
                    "type": "error",
                    "code": "pipeline_error",
                    "message": str(exc),
                    "request_id": request_id,
                }
            )
        raise
    finally:
        lifecycle.complete(request_id)
