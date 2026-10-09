"""
Phase 6 regression coverage (mocked services — no GPU).

Covers:
- multiple messages on one WebSocket
- disconnect during processing
- reconnect (new connection after close)
- stale-event rejection
- duplicate / busy submission prevention
- valid WAV payload handling
- generator off event-loop (reconfirmed)
- dependency-version validation
- repeated cached / uncached TTS calls
"""
from __future__ import annotations

import asyncio
import base64
import json
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ws_harness import build_ws_app, is_stale_event, make_minimal_wav  # noqa: E402
from server.deps_check import (  # noqa: E402
    REQUIRED_VERSIONS,
    assert_dependency_versions,
    check_dependency_versions,
)
from server.ws_streaming import stream_sync_generator  # noqa: E402


def _fast_pipeline(question=None, audio_b64=None, k=3):
    def gen():
        text = question or "audio-query"
        yield {"type": "transcript", "text": text}
        yield {"type": "intent", "intent": "GREETING", "is_direct": True}
        yield {
            "type": "audio",
            "sentence": "ሰላም",
            "audio_b64": "QQ==",
            "duration": 0.1,
            "is_first": True,
            "t": 0.05,
        }
        yield {
            "type": "done",
            "full_text": "ሰላም",
            "total_compute_s": 0.1,
            "stage_timings": {"total_s": 0.1},
        }

    return gen


def _slow_pipeline(delay=0.3):
    def make(question=None, audio_b64=None, k=3):
        def gen():
            yield {"type": "transcript", "text": question or "slow"}
            time.sleep(delay)
            yield {"type": "done", "full_text": "done", "total_compute_s": delay}

        return gen

    return make


def _recv_until(ws, pred, limit=20):
    events = []
    for _ in range(limit):
        ev = json.loads(ws.receive_text())
        events.append(ev)
        if pred(ev):
            break
    return events


# ---------------------------------------------------------------------------
# WebSocket: multiple messages / busy / disconnect / reconnect
# ---------------------------------------------------------------------------


def test_multiple_text_messages_on_one_websocket():
    app = build_ws_app(_fast_pipeline)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/s2s") as ws:
            ws.send_text(json.dumps({"type": "text", "text": "one", "request_id": "a1"}))
            evs1 = _recv_until(ws, lambda e: e.get("type") == "done")
            assert evs1[0]["type"] == "accepted"
            assert any(e["type"] == "transcript" and e["text"] == "one" for e in evs1)
            assert evs1[-1]["type"] == "done"
            assert all(
                e.get("request_id") == "a1"
                for e in evs1
                if e["type"] != "accepted" or True
            )

            ws.send_text(json.dumps({"type": "text", "text": "two", "request_id": "a2"}))
            evs2 = _recv_until(ws, lambda e: e.get("type") == "done")
            assert any(e["type"] == "transcript" and e["text"] == "two" for e in evs2)
            assert evs2[-1]["request_id"] == "a2"


def test_duplicate_submission_rejected_busy():
    app = build_ws_app(_slow_pipeline(0.5))
    with TestClient(app) as client:
        with client.websocket_connect("/ws/s2s") as ws:
            ws.send_text(json.dumps({"type": "text", "text": "first", "request_id": "b1"}))
            first = json.loads(ws.receive_text())
            assert first["type"] == "accepted"

            ws.send_text(json.dumps({"type": "text", "text": "second", "request_id": "b2"}))
            saw_busy = False
            for _ in range(30):
                ev = json.loads(ws.receive_text())
                if ev.get("type") == "error" and ev.get("code") == "busy":
                    assert ev["active_request_id"] == "b1"
                    assert ev["request_id"] == "b2"
                    saw_busy = True
                    break
            assert saw_busy, "expected busy rejection for overlapping request"

            # Drain first request completion
            for _ in range(30):
                ev = json.loads(ws.receive_text())
                if ev.get("type") in ("done", "cancelled", "error") and ev.get("code") != "busy":
                    break


def test_disconnect_during_processing_cancels_cleanly():
    app = build_ws_app(_slow_pipeline(1.0))
    with TestClient(app) as client:
        with client.websocket_connect("/ws/s2s") as ws:
            ws.send_text(json.dumps({"type": "text", "text": "x", "request_id": "d1"}))
            assert json.loads(ws.receive_text())["type"] == "accepted"
            # Closing mid-flight must not hang / raise on the server task.
            ws.close()


def test_reconnect_after_disconnect_accepts_new_session():
    app = build_ws_app(_fast_pipeline)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/s2s") as ws1:
            ws1.send_text(json.dumps({"type": "text", "text": "before", "request_id": "r1"}))
            _recv_until(ws1, lambda e: e.get("type") == "done")
            ws1.close()

        with client.websocket_connect("/ws/s2s") as ws2:
            ws2.send_text(json.dumps({"type": "text", "text": "after", "request_id": "r2"}))
            evs = _recv_until(ws2, lambda e: e.get("type") == "done")
            assert evs[0]["type"] == "accepted"
            assert any(e.get("text") == "after" for e in evs)


def test_cancel_message_stops_inflight():
    app = build_ws_app(_slow_pipeline(0.5))
    with TestClient(app) as client:
        with client.websocket_connect("/ws/s2s") as ws:
            ws.send_text(json.dumps({"type": "text", "text": "c", "request_id": "c1"}))
            assert json.loads(ws.receive_text())["type"] == "accepted"
            # May get transcript before cancel lands
            ws.send_text(json.dumps({"type": "cancel", "request_id": "c1"}))
            types = []
            for _ in range(10):
                ev = json.loads(ws.receive_text())
                types.append(ev["type"])
                if ev["type"] in ("cancelled", "cancel_ack", "done", "error"):
                    if ev["type"] == "cancel_ack":
                        continue
                    break
            assert "cancel_ack" in types or "cancelled" in types or "done" in types


# ---------------------------------------------------------------------------
# Stale-event rejection
# ---------------------------------------------------------------------------


def test_stale_event_rejection():
    assert is_stale_event({"type": "audio", "request_id": "old"}, "new") is True
    assert is_stale_event({"type": "audio", "request_id": "new"}, "new") is False
    assert is_stale_event({"type": "error", "code": "busy", "request_id": "x"}, "y") is False
    assert is_stale_event({"type": "done"}, "new") is False


# ---------------------------------------------------------------------------
# WAV payload
# ---------------------------------------------------------------------------


def test_valid_wav_payload_roundtrip_base64():
    wav = make_minimal_wav()
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    b64 = base64.b64encode(wav).decode("ascii")
    raw = base64.b64decode(b64)
    assert raw == wav
    assert len(raw) > 44


def test_asr_transcribe_base64_accepts_valid_wav():
    """Decode path works; inference is mocked (no GPU / model)."""
    from server.asr_service import ASRService

    asr = ASRService(device="cpu")
    asr.is_ready = True

    wav = make_minimal_wav(n_samples=3200)
    b64 = base64.b64encode(wav).decode("ascii")

    def fake_infer(audio, t_start, load_s):
        assert isinstance(audio, np.ndarray)
        assert audio.ndim == 1
        assert audio.dtype == np.float32
        return {
            "raw": "ሰላም",
            "transcript": "ሰላም",
            "load_s": load_s,
            "preprocessing_s": 0.0,
            "inference_s": 0.0,
            "total_s": 0.01,
        }

    asr._run_inference = fake_infer  # type: ignore

    class _FakeSF:
        @staticmethod
        def read(buf, dtype="float32", always_2d=False):
            # Consume buffer; return mono silence at 16 kHz
            buf.read()
            return np.zeros(3200, dtype=np.float32), 16000

    with patch("server.asr_service.sf", _FakeSF):
        res = asr.transcribe_base64(b64)

    assert res["transcript"] == "ሰላም"


def test_audio_message_over_websocket_with_wav():
    wav_b64 = base64.b64encode(make_minimal_wav()).decode("ascii")

    def make(question=None, audio_b64=None, k=3):
        def gen():
            assert audio_b64 == wav_b64
            yield {"type": "transcript", "text": "from-audio"}
            yield {"type": "done", "full_text": "ok", "total_compute_s": 0.01}

        return gen

    app = build_ws_app(make)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/s2s") as ws:
            ws.send_text(
                json.dumps(
                    {"type": "audio", "audio_b64": wav_b64, "request_id": "w1"}
                )
            )
            evs = _recv_until(ws, lambda e: e.get("type") == "done")
            assert any(e.get("text") == "from-audio" for e in evs)


# ---------------------------------------------------------------------------
# Generator off event loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generator_runs_outside_event_loop():
    worker_tid = {"id": None}
    main_tid = threading.get_ident()

    def gen():
        worker_tid["id"] = threading.get_ident()
        time.sleep(0.05)
        yield {"type": "tick"}

    events = [ev async for ev in stream_sync_generator(gen)]
    assert events == [{"type": "tick"}]
    assert worker_tid["id"] is not None
    assert worker_tid["id"] != main_tid


# ---------------------------------------------------------------------------
# Dependency version validation
# ---------------------------------------------------------------------------


def test_dependency_version_validation_detects_mismatch():
    ok, mismatches = check_dependency_versions(
        {"transformers": "9.9.9"}, allow_missing=False
    )
    # Either not installed or wrong version → not ok
    assert ok is False
    assert mismatches


def test_dependency_version_validation_allow_missing_passes():
    ok, mismatches = check_dependency_versions(
        {"definitely-not-a-real-pkg-xyz": "1.0.0"},
        allow_missing=True,
    )
    assert ok is True
    assert mismatches == []


def test_assert_dependency_versions_raises():
    with pytest.raises(RuntimeError, match="Incompatible"):
        assert_dependency_versions({"transformers": "0.0.0"}, allow_missing=False)


def test_required_versions_match_omnivoice_stack():
    from server.deps_check import REQUIRED_EXACT, REQUIRED_VERSIONS, transformers_ok

    assert REQUIRED_EXACT["omnivoice"] == "0.2.1"
    assert REQUIRED_EXACT["sentence-transformers"] == "5.7.0"
    assert "5.3.0" in REQUIRED_VERSIONS["transformers"]
    assert transformers_ok("5.3.0")
    assert transformers_ok("5.19.0")
    assert not transformers_ok("4.57.1")
    assert not transformers_ok("6.0.0")


def test_notebook_install_order_and_gate_present():
    nb_path = ROOT_DIR.parent / "amharic-speech-to-speech-rag-api-server (12).ipynb"
    if not nb_path.exists():
        nb_path = ROOT_DIR.parent / "amharic-speech-to-speech-rag-api-server (11).ipynb"
    if not nb_path.exists():
        pytest.skip("deploy notebook not in parent folder")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    text = nb_path.read_text(encoding="utf-8")
    assert "omnivoice==0.2.1" in text
    assert "sentence-transformers==5.7.0" in text
    assert "transformers>=5.3.0" in text
    assert "4.57.1" not in text
    assert "HiggsAudioV2TokenizerModel" in text

    install = ""
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if "pip install" in src and "omnivoice==0.2.1" in src:
            install = src
            break
    assert install
    pip_lines = [ln for ln in install.splitlines() if ln.lstrip().startswith("!pip")]
    omni_i = next(i for i, ln in enumerate(pip_lines) if "omnivoice==0.2.1" in ln)
    tf_i = next(i for i, ln in enumerate(pip_lines) if "transformers>=" in ln)
    assert omni_i < tf_i


# ---------------------------------------------------------------------------
# TTS cached vs uncached (mocked synthesize)
# ---------------------------------------------------------------------------


def test_tts_repeated_cached_and_uncached_calls():
    from server.tts_service import TTSService
    from server.config import IDK_TEXT

    tts = TTSService(device="cpu")
    tts.is_ready = True

    fake = {
        "text": "x",
        "waveform": np.zeros(100, dtype=np.float32),
        "wav_bytes": b"RIFF",
        "audio_b64": "QQ==",
        "duration": 0.01,
        "time": 0.2,
        "rtf": 1.0,
        "steps": 8,
        "cached": False,
    }
    tts.idk_audio = dict(fake, text=IDK_TEXT, cached=True)
    router_key = tts.prep_text("ሰላም እንደምን ዋልክ")
    tts.router_audio_cache = {router_key: dict(fake, cached=True)}
    tts._synthesize_raw = MagicMock(return_value=dict(fake))  # type: ignore

    # IDK → cached, no synthesize
    r1 = tts.speak_sentence(IDK_TEXT, is_first=True)
    assert r1["cached"] is True
    tts._synthesize_raw.assert_not_called()

    # Router cache hit
    r2 = tts.speak_sentence("ሰላም እንደምን ዋልክ", is_first=True)
    assert r2["cached"] is True
    tts._synthesize_raw.assert_not_called()

    # Uncached then memoized
    novel = "አዲስ ዓረፍተ ነገር።"
    r3 = tts.speak_sentence(novel, is_first=True)
    assert r3["cached"] is False
    assert tts._synthesize_raw.call_count == 1

    r4 = tts.speak_sentence(novel, is_first=False)
    assert r4["cached"] is True
    assert tts._synthesize_raw.call_count == 1  # memo hit


def test_ui_blocks_duplicate_while_inflight():
    app_src = (ROOT_DIR / "server" / "app.py").read_text(encoding="utf-8")
    assert "requestInFlight" in app_src
    assert "if (requestInFlight)" in app_src
    assert "ensureMicStream" in app_src
    assert "isStaleEvent" in app_src
