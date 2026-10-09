"""
Non-GPU tests for the WebSocket streaming bridge (Phase 2).

These verify that synchronous generators run off the asyncio event loop and
that events are yielded incrementally — without loading ASR/LLM/TTS models.
"""
import asyncio
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from server.ws_streaming import sanitize_event, stream_sync_generator


def test_sanitize_event_converts_numpy():
    ev = {
        "type": "audio",
        "scores": np.array([0.1, 0.2], dtype=np.float32),
        "count": np.int64(3),
        "rtf": np.float64(1.25),
        "text": "ሰላም",
    }
    clean = sanitize_event(ev)
    assert clean["scores"] == pytest.approx([0.1, 0.2], abs=1e-5)
    assert clean["count"] == 3
    assert isinstance(clean["count"], int)
    assert clean["rtf"] == pytest.approx(1.25)
    assert isinstance(clean["rtf"], float)
    assert clean["text"] == "ሰላም"


@pytest.mark.asyncio
async def test_stream_sync_generator_yields_incrementally():
    """Events must arrive as produced, not after the whole generator finishes."""
    arrived_at = []

    def slow_gen():
        for i in range(3):
            time.sleep(0.05)
            yield {"type": "tick", "i": i}

    t0 = time.perf_counter()
    async for ev in stream_sync_generator(slow_gen):
        arrived_at.append((ev["i"], time.perf_counter() - t0))

    assert [i for i, _ in arrived_at] == [0, 1, 2]
    # First event well before the full ~150ms would finish if buffered.
    assert arrived_at[0][1] < 0.12
    assert arrived_at[2][1] >= 0.12


@pytest.mark.asyncio
async def test_stream_sync_generator_runs_off_event_loop():
    """Worker must not occupy the asyncio thread — loop stays responsive."""
    worker_thread = {"id": None}
    loop_ticks = {"n": 0}
    stop = asyncio.Event()

    def blocking_gen():
        worker_thread["id"] = threading.get_ident()
        time.sleep(0.15)
        yield {"type": "done"}

    async def ticker():
        while not stop.is_set():
            loop_ticks["n"] += 1
            await asyncio.sleep(0.01)

    tick_task = asyncio.create_task(ticker())
    main_tid = threading.get_ident()

    events = []
    async for ev in stream_sync_generator(blocking_gen):
        events.append(ev)

    stop.set()
    await tick_task

    assert events == [{"type": "done"}]
    assert worker_thread["id"] is not None
    assert worker_thread["id"] != main_tid
    # Event loop kept ticking while the worker slept ~150ms.
    assert loop_ticks["n"] >= 8


@pytest.mark.asyncio
async def test_stream_sync_generator_propagates_errors():
    def bad_gen():
        yield {"type": "ok"}
        raise RuntimeError("pipeline boom")

    received = []
    with pytest.raises(RuntimeError, match="pipeline boom"):
        async for ev in stream_sync_generator(bad_gen):
            received.append(ev)

    assert received == [{"type": "ok"}]


@pytest.mark.asyncio
async def test_stream_sync_generator_respects_cancel_event():
    cancel = asyncio.Event()
    produced = []

    def long_gen():
        for i in range(20):
            produced.append(i)
            yield {"i": i}
            time.sleep(0.03)

    received = []
    async for ev in stream_sync_generator(long_gen, cancel_event=cancel):
        received.append(ev["i"])
        if ev["i"] == 1:
            cancel.set()

    assert received[0] == 0
    assert 1 in received
    # Cooperative cancel: may get one more item already in flight, but not all 20.
    assert len(received) < 10
    assert len(produced) < 20


@pytest.mark.asyncio
async def test_event_loop_can_interleave_while_streaming():
    """Simulate another async client task making progress during a slow stream."""
    other_progress = []

    def slow_gen():
        for i in range(4):
            time.sleep(0.04)
            yield {"i": i}

    async def other_client():
        for n in range(5):
            await asyncio.sleep(0.025)
            other_progress.append(n)

    other_task = asyncio.create_task(other_client())
    stream_events = [ev async for ev in stream_sync_generator(slow_gen)]
    await other_task

    assert [e["i"] for e in stream_events] == [0, 1, 2, 3]
    assert other_progress == [0, 1, 2, 3, 4]


def test_websocket_handler_does_not_call_stream_query_inline():
    """Regression: handler must use stream_sync_generator, not a bare for-loop.

    Reads source text instead of importing ``server.app`` (avoids loading the
    full model/service import graph on CPU-only machines).
    """
    app_path = ROOT_DIR / "server" / "app.py"
    src = app_path.read_text(encoding="utf-8")

    # Locate the websocket handler body.
    start = src.find("async def websocket_s2s")
    assert start != -1
    rest = src[start:]
    # Next top-level def/decorator after the handler ends the slice.
    end_candidates = [
        rest.find("\n@app.get"),
        rest.find("\ndef serve_ui"),
        rest.find("\n# -------------------------------------------------------------"),
    ]
    ends = [e for e in end_candidates if e > 0]
    handler_src = rest[: min(ends)] if ends else rest

    assert "drive_pipeline_events" in handler_src
    assert "WSRequestLifecycle" in handler_src
    assert "run_gen" not in handler_src
    assert "for ev in orchestrator.stream_query" not in handler_src
