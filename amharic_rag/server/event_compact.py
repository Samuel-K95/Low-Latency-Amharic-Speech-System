#!/usr/bin/env python3
"""
Compact WebSocket / stream event helpers (Phase 5 latency & bandwidth).

Keeps sentence-level audio streaming intact while trimming redundant fields
(full chunk text, raw ASR dumps, duplicate full WAV on ``done``).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def compact_asr_metrics(asr_res: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    """Keep timing fields only — drop raw / transcript duplicates."""
    if not asr_res:
        return None
    return {
        "load_s": float(asr_res.get("load_s") or 0.0),
        "prep_s": float(
            asr_res.get("preprocessing_s", asr_res.get("prep_s")) or 0.0
        ),
        "infer_s": float(asr_res.get("inference_s", asr_res.get("infer_s")) or 0.0),
        "total_s": float(asr_res.get("total_s") or 0.0),
    }


def compact_retrieved(docs: List[Dict[str, Any]], *, preview_chars: int = 120) -> List[Dict[str, Any]]:
    """Ship scores + short preview instead of full proclamation passages."""
    out: List[Dict[str, Any]] = []
    for d in docs:
        text = d.get("text_spoken") or d.get("text_clean") or ""
        out.append(
            {
                "id": d.get("id"),
                "article_id": d.get("article_id") or d.get("id"),
                "score": d.get("score"),
                "dense": d.get("dense"),
                "keyword": d.get("keyword"),
                "text_preview": text[:preview_chars],
            }
        )
    return out


def build_stage_timings(marks: Dict[str, float], *, asr_res=None, ret_timing=None, llm_meta=None, audio_chunks=None) -> Dict[str, float]:
    """Independent stage timings for UI waterfall / debugging."""
    audio_chunks = audio_chunks or []
    llm_meta = llm_meta or {}
    ret_timing = ret_timing or {}

    asr_s = float((asr_res or {}).get("total_s") or marks.get("asr_done") or 0.0)
    retrieval_s = float(ret_timing.get("total_ms") or 0.0) / 1000.0
    if not retrieval_s and "retrieval_done" in marks:
        retrieval_s = float(marks["retrieval_done"]) - float(marks.get("asr_done") or 0.0)

    llm_ttft_s = float(llm_meta.get("ttft_s") or marks.get("first_token") or 0.0)
    tts_first_s = float(audio_chunks[0]["time"]) if audio_chunks else 0.0
    first_audio_s = float(marks.get("first_audio_ready") or 0.0)
    total_s = float(marks.get("all_done") or 0.0)

    return {
        "asr_s": round(asr_s, 4),
        "retrieval_s": round(max(0.0, retrieval_s), 4),
        "llm_ttft_s": round(llm_ttft_s, 4),
        "tts_first_s": round(tts_first_s, 4),
        "first_audio_s": round(first_audio_s, 4),
        "total_s": round(total_s, 4),
    }
