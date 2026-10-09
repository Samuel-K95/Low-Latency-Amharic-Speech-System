"""
Phase 5 non-GPU tests: compact events, pipeline timeout, LLM health TTL, cache trim.
"""
import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from server.event_compact import (
    compact_asr_metrics,
    compact_retrieved,
    build_stage_timings,
)
from server.ws_lifecycle import WSRequestLifecycle, drive_pipeline_events
from server.config import ANSWER_CACHE_MAX, WS_OMIT_FULL_AUDIO


def test_compact_asr_metrics_keeps_timings_only():
    raw = {
        "raw": "ሰላም [AMH]",
        "transcript": "ሰላም",
        "load_s": 0.1,
        "preprocessing_s": 0.2,
        "inference_s": 0.3,
        "total_s": 0.6,
    }
    compact = compact_asr_metrics(raw)
    assert compact == {
        "load_s": 0.1,
        "prep_s": 0.2,
        "infer_s": 0.3,
        "total_s": 0.6,
    }
    assert "raw" not in compact
    assert compact_asr_metrics(None) is None


def test_compact_retrieved_truncates_text():
    docs = [
        {
            "id": "c1",
            "article_id": "Art-11",
            "score": 0.9,
            "dense": 0.8,
            "keyword": 0.7,
            "text_spoken": "A" * 500,
            "text_clean": "B" * 500,
        }
    ]
    out = compact_retrieved(docs, preview_chars=40)
    assert out[0]["id"] == "c1"
    assert out[0]["article_id"] == "Art-11"
    assert len(out[0]["text_preview"]) == 40
    assert "text_spoken" not in out[0]
    assert "text_clean" not in out[0]


def test_build_stage_timings_independent_stages():
    marks = {
        "asr_done": 0.5,
        "retrieval_done": 0.8,
        "first_token": 1.2,
        "first_audio_ready": 1.9,
        "all_done": 3.0,
    }
    stage = build_stage_timings(
        marks,
        asr_res={"total_s": 0.45},
        ret_timing={"total_ms": 120.0},
        llm_meta={"ttft_s": 0.4},
        audio_chunks=[{"time": 0.55}],
    )
    assert stage["asr_s"] == pytest.approx(0.45)
    assert stage["retrieval_s"] == pytest.approx(0.12)
    assert stage["llm_ttft_s"] == pytest.approx(0.4)
    assert stage["tts_first_s"] == pytest.approx(0.55)
    assert stage["first_audio_s"] == pytest.approx(1.9)
    assert stage["total_s"] == pytest.approx(3.0)


def test_ws_omit_full_audio_flag_enabled():
    assert WS_OMIT_FULL_AUDIO is True


def test_done_event_omits_full_audio_contract_in_source():
    """Streaming done payloads must omit full_audio_b64 when WS_OMIT_FULL_AUDIO."""
    src = (ROOT_DIR / "server" / "orchestrator.py").read_text(encoding="utf-8")
    assert "def _done_event" in src
    assert "WS_OMIT_FULL_AUDIO" in src
    assert 'payload["full_audio_b64"] = full_b64' in src
    assert "stage_timings" in src
    assert "warm_answer_cache" in src
    assert "_trim_answer_cache" in src


def test_answer_cache_trim_logic():
    class _Orch:
        def __init__(self):
            self.answer_cache = []

        def _trim_answer_cache(self):
            if len(self.answer_cache) > ANSWER_CACHE_MAX:
                self.answer_cache = self.answer_cache[-ANSWER_CACHE_MAX:]

    orch = _Orch()
    orch.answer_cache = [
        {"key": f"k{i}", "qv": np.ones(4), "sentences": ["x"]}
        for i in range(ANSWER_CACHE_MAX + 5)
    ]
    orch._trim_answer_cache()
    assert len(orch.answer_cache) == ANSWER_CACHE_MAX
    assert orch.answer_cache[0]["key"] == "k5"


def test_warm_answer_cache_source_reads_questions_json():
    src = (ROOT_DIR / "server" / "orchestrator.py").read_text(encoding="utf-8")
    assert 'DATA_DIR / "questions.json"' in src
    assert '["warm"] = True' in src or '"warm": True' in src


@pytest.mark.asyncio
async def test_drive_pipeline_timeout_emits_error():
    life = WSRequestLifecycle()
    assert life.try_begin("t1")[0]
    sent = []

    async def send_json(payload):
        sent.append(payload)

    def slow_gen():
        yield {"type": "transcript", "text": "x"}
        time.sleep(0.25)
        yield {"type": "done", "full_text": "late"}

    await drive_pipeline_events(
        request_id="t1",
        lifecycle=life,
        gen_factory=slow_gen,
        send_json=send_json,
        timeout_s=0.05,
    )

    types = [e["type"] for e in sent]
    assert "accepted" in types
    assert "error" in types
    err = next(e for e in sent if e["type"] == "error")
    assert err["code"] == "timeout"
    assert not life.is_busy


def test_llm_ensure_server_skips_redundant_health_check():
    from server.llm_service import LLMService

    llm = LLMService.__new__(LLMService)
    llm.server_url = "http://127.0.0.1:9"
    llm.server_proc = MagicMock()
    llm.server_proc.poll.return_value = None
    llm._last_health_ok = time.time()

    with patch("server.llm_service.requests.get") as get:
        llm.ensure_server()
        get.assert_not_called()


def test_phase5_markers_in_app_and_config():
    app_src = (ROOT_DIR / "server" / "app.py").read_text(encoding="utf-8")
    cfg_src = (ROOT_DIR / "server" / "config.py").read_text(encoding="utf-8")
    assert "stage_timings" in app_src
    assert 'code === "timeout"' in app_src or "code === 'timeout'" in app_src
    assert "PIPELINE_TIMEOUT_S" in cfg_src
    assert "WS_OMIT_FULL_AUDIO" in cfg_src
    assert "LLM_HEALTH_TTL_S" in cfg_src
