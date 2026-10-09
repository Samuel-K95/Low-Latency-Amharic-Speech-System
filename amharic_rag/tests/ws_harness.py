"""
Minimal FastAPI WebSocket app mirroring production lifecycle (no GPU / no models).

Used by Phase 6 regression tests so we never import ``server.app`` (heavy deps).
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from server.ws_lifecycle import (
    WSRequestLifecycle,
    busy_error_payload,
    drive_pipeline_events,
)


GenFactoryBuilder = Callable[
    [Optional[str], Optional[str], int],
    Callable[[], Any],
]


def build_ws_app(make_gen_factory: GenFactoryBuilder) -> FastAPI:
    """
    ``make_gen_factory(question, audio_b64, k)`` must return a zero-arg callable
    that yields pipeline events (same contract as orchestrator.stream_query).
    """
    app = FastAPI()

    @app.websocket("/ws/s2s")
    async def websocket_s2s(websocket: WebSocket):
        await websocket.accept()
        lifecycle = WSRequestLifecycle()
        pipeline_task: Optional[asyncio.Task] = None

        async def send_json(payload: Dict[str, Any]) -> None:
            await websocket.send_text(json.dumps(payload))

        async def run_one(
            request_id: str,
            question_text: Optional[str],
            audio_b64: Optional[str],
            k: int,
        ) -> None:
            gen_factory = make_gen_factory(question_text, audio_b64, k)
            try:
                await drive_pipeline_events(
                    request_id=request_id,
                    lifecycle=lifecycle,
                    gen_factory=gen_factory,
                    send_json=send_json,
                    timeout_s=30.0,
                )
            except Exception:
                pass

        try:
            while True:
                raw_msg = await websocket.receive_text()
                try:
                    data = json.loads(raw_msg)
                except json.JSONDecodeError:
                    await send_json(
                        {
                            "type": "error",
                            "code": "invalid_json",
                            "message": "Message must be JSON",
                        }
                    )
                    continue

                msg_type = data.get("type")
                if msg_type == "cancel":
                    cancelled = lifecycle.request_cancel(data.get("request_id"))
                    await send_json(
                        {
                            "type": "cancel_ack",
                            "cancelled": cancelled,
                            "request_id": data.get("request_id")
                            or lifecycle.active_request_id,
                        }
                    )
                    continue

                if msg_type not in ("audio", "text"):
                    await send_json(
                        {
                            "type": "error",
                            "code": "invalid_type",
                            "message": f"Unsupported type: {msg_type}",
                        }
                    )
                    continue

                request_id = WSRequestLifecycle.new_request_id(data.get("request_id"))
                started, busy_id = lifecycle.try_begin(request_id)
                if not started:
                    await send_json(
                        busy_error_payload(
                            active_request_id=busy_id or "",
                            request_id=request_id,
                        )
                    )
                    continue

                question_text = data.get("text") if msg_type == "text" else None
                audio_b64 = data.get("audio_b64") if msg_type == "audio" else None
                k = int(data.get("k", 3))
                pipeline_task = asyncio.create_task(
                    run_one(request_id, question_text, audio_b64, k)
                )
        except WebSocketDisconnect:
            lifecycle.on_disconnect()
            if pipeline_task is not None and not pipeline_task.done():
                pipeline_task.cancel()
                try:
                    await pipeline_task
                except (asyncio.CancelledError, Exception):
                    pass

    return app


def is_stale_event(ev: Dict[str, Any], active_request_id: Optional[str]) -> bool:
    """Python mirror of the browser stale-event filter (Phase 3/4)."""
    if not ev or not ev.get("request_id"):
        return False
    if not active_request_id:
        return False
    if ev.get("type") == "error" and ev.get("code") == "busy":
        return False
    return ev["request_id"] != active_request_id


def make_minimal_wav(sample_rate: int = 16000, n_samples: int = 1600) -> bytes:
    """Build a valid mono 16-bit PCM WAV without soundfile."""
    import struct

    data_size = n_samples * 2
    buf = bytearray()
    buf.extend(b"RIFF")
    buf.extend(struct.pack("<I", 36 + data_size))
    buf.extend(b"WAVE")
    buf.extend(b"fmt ")
    buf.extend(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
    buf.extend(b"data")
    buf.extend(struct.pack("<I", data_size))
    buf.extend(b"\x00" * data_size)
    return bytes(buf)
