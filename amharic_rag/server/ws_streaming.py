#!/usr/bin/env python3
"""
Bridge synchronous pipeline generators into the asyncio event loop.

The ASR / retrieval / LLM / TTS pipeline is synchronous. Running it directly
inside an async WebSocket handler blocks disconnect detection, control frames,
and other connections. This module runs the generator in a worker thread and
yields each event on the event loop as soon as it is produced.
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np

_SENTINEL = object()


def sanitize_event(ev: Dict[str, Any]) -> Dict[str, Any]:
    """Convert numpy values so events are JSON-serializable."""
    clean: Dict[str, Any] = {}
    for key, value in ev.items():
        if isinstance(value, np.ndarray):
            clean[key] = value.tolist()
        elif isinstance(value, (np.floating, np.integer)):
            clean[key] = value.item()
        else:
            clean[key] = value
    return clean


async def stream_sync_generator(
    gen_factory: Callable[[], Iterable[Any]],
    *,
    cancel_event: Optional[asyncio.Event] = None,
):
    """
    Run ``gen_factory()`` in a worker thread and yield items as they are produced.

    ``gen_factory`` must return a fresh iterable/generator when called. Items are
    pushed onto an asyncio queue via ``call_soon_threadsafe`` so the event loop
    stays free for WebSocket keepalives, disconnects, and other clients.

    If ``cancel_event`` is set, the worker stops putting new items after the
    current ``next()`` returns (cooperative; does not abort in-flight GPU work).
    """
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    errors: list = []

    def _worker() -> None:
        try:
            for item in gen_factory():
                if cancel_event is not None and cancel_event.is_set():
                    break
                loop.call_soon_threadsafe(queue.put_nowait, item)
        except Exception as exc:  # surface to the async consumer
            errors.append(exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

    fut = loop.run_in_executor(None, _worker)
    finished_normally = False
    try:
        while True:
            item = await queue.get()
            if item is _SENTINEL:
                finished_normally = True
                break
            yield item
        if errors:
            raise errors[0]
    finally:
        # Signal the worker only when the consumer aborts early (break / cancel /
        # exception). Never set cancel_event after a normal sentinel drain — that
        # would look like a user cancel to Phase 3 lifecycle code.
        if cancel_event is not None and not finished_normally:
            cancel_event.set()
        await fut
