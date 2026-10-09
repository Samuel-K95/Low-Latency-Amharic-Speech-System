"""
Non-GPU tests for WebSocket request lifecycle (Phase 3).
"""
import asyncio
import sys
import time
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from server.ws_lifecycle import (
    WSRequestLifecycle,
    busy_error_payload,
    drive_pipeline_events,
)


def test_new_request_id_uses_client_or_generates():
    assert WSRequestLifecycle.new_request_id("abc123") == "abc123"
    assert WSRequestLifecycle.new_request_id("  x  ") == "x"
    generated = WSRequestLifecycle.new_request_id(None)
    assert isinstance(generated, str) and len(generated) >= 8
    assert WSRequestLifecycle.new_request_id("") != ""


def test_try_begin_and_busy_rejection():
    life = WSRequestLifecycle()
    ok, busy = life.try_begin("r1")
    assert ok and busy is None
    assert life.is_busy
    assert life.is_current("r1")

    ok2, busy2 = life.try_begin("r2")
    assert not ok2
    assert busy2 == "r1"

    life.complete("r1")
    assert not life.is_busy
    ok3, _ = life.try_begin("r2")
    assert ok3


def test_complete_ignores_stale_id():
    life = WSRequestLifecycle()
    life.try_begin("r1")
    life.complete("other")
    assert life.is_current("r1")
    life.complete("r1")
    assert not life.is_busy


def test_request_cancel_matching_and_mismatch():
    life = WSRequestLifecycle()
    life.try_begin("r1")
    assert not life.request_cancel("r2")
    assert not life.cancel_event.is_set()
    assert life.request_cancel("r1")
    assert life.cancel_event.is_set()


def test_on_disconnect_cancels_active():
    life = WSRequestLifecycle()
    life.try_begin("r1")
    life.on_disconnect()
    assert life.cancel_event.is_set()


def test_busy_error_payload_shape():
    payload = busy_error_payload(active_request_id="a1", request_id="b2")
    assert payload["type"] == "error"
    assert payload["code"] == "busy"
    assert payload["active_request_id"] == "a1"
    assert payload["request_id"] == "b2"


@pytest.mark.asyncio
async def test_drive_pipeline_stamps_request_id_and_accepted():
    life = WSRequestLifecycle()
    assert life.try_begin("req-1")[0]
    sent = []

    async def send_json(payload):
        sent.append(payload)

    def gen_factory():
        yield {"type": "transcript", "text": "ሰላም"}
        yield {"type": "done", "full_text": "ሰላም", "total_compute_s": 0.1}

    await drive_pipeline_events(
        request_id="req-1",
        lifecycle=life,
        gen_factory=gen_factory,
        send_json=send_json,
    )

    assert sent[0] == {"type": "accepted", "request_id": "req-1"}
    assert all(
        e.get("request_id") == "req-1" for e in sent if e["type"] != "accepted"
    ) or sent[0]["request_id"] == "req-1"
    assert sent[1]["type"] == "transcript" and sent[1]["request_id"] == "req-1"
    assert sent[2]["type"] == "done" and sent[2]["request_id"] == "req-1"
    assert not any(e["type"] == "cancelled" for e in sent)
    assert not life.is_busy


@pytest.mark.asyncio
async def test_drive_pipeline_cancel_emits_cancelled_not_done_tail():
    life = WSRequestLifecycle()
    assert life.try_begin("req-c")[0]
    sent = []

    async def send_json(payload):
        sent.append(payload)
        if payload.get("type") == "transcript":
            life.request_cancel("req-c")

    def gen_factory():
        yield {"type": "transcript", "text": "a"}
        time.sleep(0.05)
        yield {"type": "audio", "sentence": "should-skip", "audio_b64": "xx"}
        yield {"type": "done", "full_text": "nope"}

    await drive_pipeline_events(
        request_id="req-c",
        lifecycle=life,
        gen_factory=gen_factory,
        send_json=send_json,
    )

    types = [e["type"] for e in sent]
    assert "accepted" in types
    assert "transcript" in types
    assert "cancelled" in types
    # After cancel, no further pipeline events should be delivered.
    assert "done" not in types
    assert not life.is_busy


@pytest.mark.asyncio
async def test_drive_pipeline_stale_filter_stops_sending():
    """If lifecycle is superseded after the first event, stop emitting."""
    life = WSRequestLifecycle()
    assert life.try_begin("old")[0]
    sent = []

    async def send_json(payload):
        sent.append(payload)
        # Supersede from the async consumer after the first real event.
        if payload.get("type") == "transcript":
            life.complete("old")
            life.try_begin("new")

    def gen_factory():
        yield {"type": "transcript", "text": "1"}
        time.sleep(0.05)
        yield {"type": "done", "full_text": "stale"}

    await drive_pipeline_events(
        request_id="old",
        lifecycle=life,
        gen_factory=gen_factory,
        send_json=send_json,
    )

    types = [e["type"] for e in sent]
    assert "transcript" in types
    assert "done" not in types
    assert life.is_current("new")


@pytest.mark.asyncio
async def test_drive_pipeline_error_event():
    life = WSRequestLifecycle()
    assert life.try_begin("err1")[0]
    sent = []

    async def send_json(payload):
        sent.append(payload)

    def gen_factory():
        yield {"type": "transcript", "text": "x"}
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await drive_pipeline_events(
            request_id="err1",
            lifecycle=life,
            gen_factory=gen_factory,
            send_json=send_json,
        )

    err = [e for e in sent if e.get("type") == "error"]
    assert err and err[0]["code"] == "pipeline_error"
    assert err[0]["request_id"] == "err1"
    assert not life.is_busy


def test_app_websocket_uses_lifecycle_source():
    src = (ROOT_DIR / "server" / "app.py").read_text(encoding="utf-8")
    start = src.find("async def websocket_s2s")
    assert start != -1
    handler = src[start : start + 4500]
    assert "WSRequestLifecycle" in handler
    assert "drive_pipeline_events" in handler
    assert "busy_error_payload" in handler
    assert '"cancel"' in handler or "'cancel'" in handler
    assert "create_task" in handler
    assert "activeRequestId" in src
    assert "isStaleEvent" in src
    assert "requestInFlight" in src
    assert "ensureMicStream" in src
    assert "scheduleReconnect" in src
