#!/usr/bin/env python3
"""
FastAPI Server & WebSocket Streaming Service for Amharic Speech-to-Speech RAG.
Exposes REST endpoints (/api/v1/chat, /api/v1/transcribe, /api/v1/retrieve, /api/v1/synthesize),
WebSocket real-time streaming (/ws/s2s), and an embedded Glassmorphic Voice UI.
"""
import os
import sys
import json
import time
import asyncio
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from server.config import (
    NUM_GPUS,
    MULTI_GPU,
    ASR_DEVICE,
    RETRIEVAL_DEVICE,
    TTS_DEVICE,
    LLM_GPU_ID,
    LLM_PORT,
    TOP_K_RETRIEVAL,
    CHAT_HISTORY_ENABLED,
    IMPROVEMENTS as _IMPROVEMENTS,
)
from server.orchestrator import S2SOrchestrator
from server.chat_history import ChatHistory
from server.ws_lifecycle import (
    WSRequestLifecycle,
    busy_error_payload,
    drive_pipeline_events,
)

# Global orchestrator singleton
orchestrator = S2SOrchestrator()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes models on startup."""
    print("[FastAPI] Initializing Speech-to-Speech Pipeline Models...")
    orchestrator.initialize()
    yield
    print("[FastAPI] Shutting down LLM server...")
    orchestrator.llm.stop()

app = FastAPI(
    title="Amharic Speech-to-Speech RAG API",
    description="Production-grade Amharic voice AI API powered by Hohe ASR, Hybrid E5 RAG, Gemma 4B LLM, and OmniVoice TTS.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for all frontends & external clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------
# Request & Response Schemas
# -------------------------------------------------------------
class ChatRequest(BaseModel):
    text: Optional[str] = Field(None, description="Typed Amharic question (Ge'ez script)")
    audio_b64: Optional[str] = Field(None, description="Base64-encoded WAV/MP3 recording")
    stream: bool = Field(True, description="Whether to stream audio chunks or return joined audio")
    k: int = Field(TOP_K_RETRIEVAL, description="Top-k context passages to retrieve")

class ChatResponse(BaseModel):
    transcript: str
    full_response: str
    audio_b64: Optional[str] = None
    audio_duration_s: float
    total_compute_s: float
    metrics: Dict[str, Any] = {}

class TranscribeRequest(BaseModel):
    audio_b64: str = Field(..., description="Base64-encoded audio waveform")

class TranscribeResponse(BaseModel):
    transcript: str
    metrics: Dict[str, Any]

class RetrieveRequest(BaseModel):
    query: str = Field(..., description="Amharic search query")
    k: int = Field(TOP_K_RETRIEVAL, description="Number of results to retrieve")
    mode: str = Field("hybrid", description="hybrid | dense | keyword")

class RetrieveResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    timing_ms: Dict[str, float]

class SynthesizeRequest(BaseModel):
    text: str = Field(..., description="Amharic text to speak")
    num_step: int = Field(16, description="OmniVoice diffusion steps (8 for fast, 16 for full)")

class SynthesizeResponse(BaseModel):
    text: str
    audio_b64: str
    duration_s: float
    time_s: float
    rtf: float

class HealthResponse(BaseModel):
    status: str
    num_gpus: int
    multi_gpu: bool
    gpu_topology: Dict[str, Any]
    models_ready: Dict[str, bool]
    vram_allocated_gb: float

# -------------------------------------------------------------
# Health & Status Endpoints
# -------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """Returns real-time GPU topology, VRAM allocation, and model readiness statuses."""
    vram_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    return {
        "status": "healthy" if orchestrator.is_initialized else "initializing",
        "num_gpus": NUM_GPUS,
        "multi_gpu": MULTI_GPU,
        "gpu_topology": {
            "asr_device": ASR_DEVICE,
            "retrieval_device": RETRIEVAL_DEVICE,
            "tts_device": TTS_DEVICE,
            "llm_gpu_id": LLM_GPU_ID,
            "llm_port": LLM_PORT
        },
        "models_ready": {
            "asr": orchestrator.asr.is_ready,
            "retrieval": orchestrator.retrieval.is_ready,
            "llm": orchestrator.llm.server_proc is not None,
            "tts": orchestrator.tts.is_ready
        },
        "vram_allocated_gb": round(vram_gb, 2)
    }

# -------------------------------------------------------------
# REST Endpoints
# -------------------------------------------------------------
@app.post("/api/v1/chat", response_model=ChatResponse, tags=["Core Pipeline"])
def chat_endpoint(req: ChatRequest):
    """Executes single-turn Speech/Text RAG query and returns synthesized speech & metrics."""
    events = list(orchestrator.stream_query(
        question=req.text,
        audio_b64=req.audio_b64,
        k=req.k,
        stream_audio=False
    ))

    transcript = ""
    full_text = ""
    full_audio_b64 = None
    duration_s = 0.0
    total_compute_s = 0.0
    metrics = {}

    for ev in events:
        if ev.get("type") == "transcript":
            transcript = ev["text"]
        elif ev.get("type") == "done":
            full_text = ev["full_text"]
            full_audio_b64 = ev.get("full_audio_b64")
            duration_s = ev.get("full_audio_duration_s", 0.0)
            total_compute_s = ev.get("total_compute_s", 0.0)
            metrics = ev.get("metrics", {})

    return {
        "transcript": transcript,
        "full_response": full_text,
        "audio_b64": full_audio_b64,
        "audio_duration_s": duration_s,
        "total_compute_s": total_compute_s,
        "metrics": metrics
    }

@app.post("/api/v1/transcribe", response_model=TranscribeResponse, tags=["Modular Services"])
def transcribe_endpoint(req: TranscribeRequest):
    """Direct ASR transcription of base64 audio."""
    res = orchestrator.asr.transcribe_base64(req.audio_b64)
    return {
        "transcript": res["transcript"],
        "metrics": {
            "load_s": res["load_s"],
            "prep_s": res["preprocessing_s"],
            "infer_s": res["inference_s"],
            "total_s": res["total_s"]
        }
    }

@app.post("/api/v1/retrieve", response_model=RetrieveResponse, tags=["Modular Services"])
def retrieve_endpoint(req: RetrieveRequest):
    """Direct RAG hybrid retrieval over Labour Proclamation chunks."""
    results, timing = orchestrator.retrieval.retrieve(req.query, k=req.k, mode=req.mode)
    return {
        "query": req.query,
        "results": results,
        "timing_ms": timing
    }

@app.post("/api/v1/synthesize", response_model=SynthesizeResponse, tags=["Modular Services"])
def synthesize_endpoint(req: SynthesizeRequest):
    """Direct TTS synthesis into base64 16-bit PCM WAV."""
    res = orchestrator.tts.speak_sentence(req.text, is_first=(req.num_step <= 8))
    if not res:
        raise HTTPException(status_code=400, detail="Failed to synthesize text.")
    return {
        "text": req.text,
        "audio_b64": res["audio_b64"],
        "duration_s": res["duration"],
        "time_s": res["time"],
        "rtf": res["rtf"]
    }

# -------------------------------------------------------------
# WebSocket Real-Time Speech-to-Speech Endpoint
# -------------------------------------------------------------
@app.websocket("/ws/s2s")
async def websocket_s2s(websocket: WebSocket):
    """
    Bidirectional streaming WebSocket endpoint.
    Accepts:
        {"type": "audio", "audio_b64": "...", "request_id": "..."} OR
        {"type": "text", "text": "...", "request_id": "..."} OR
        {"type": "cancel", "request_id": "..."}  # optional request_id
    Emits in real time (each event includes request_id):
        {"type": "accepted", "request_id": "..."}
        {"type": "transcript", ...}
        {"type": "intent", ...}
        {"type": "retrieved", ...}
        {"type": "audio", ...}
        {"type": "done", ...}
        {"type": "cancelled", ...}
        {"type": "error", "code": "busy"|"pipeline_error"|..., ...}

    The sync RAG generator runs in a worker thread. The receive loop stays active
    so cancel / busy rejection work while a pipeline is in flight.
    """
    await websocket.accept()
    print("[WebSocket] Client connected.")

    lifecycle = WSRequestLifecycle()
    pipeline_task: Optional[asyncio.Task] = None
    # Per-connection history: last 2 legal Q/A turns → LLM messages only (not retrieval).
    session_history = ChatHistory(
        enabled=CHAT_HISTORY_ENABLED and bool(_IMPROVEMENTS.get("chat_history", True))
    )

    async def send_json(payload: Dict[str, Any]) -> None:
        await websocket.send_text(json.dumps(payload))

    async def run_one(
        request_id: str,
        question_text: Optional[str],
        audio_b64: Optional[str],
        k: int,
    ) -> None:
        def gen_factory(
            _question=question_text,
            _audio_b64=audio_b64,
            _k=k,
        ):
            return orchestrator.stream_query(
                question=_question,
                audio_b64=_audio_b64,
                k=_k,
                stream_audio=True,
                chat_history=session_history,
            )

        try:
            await drive_pipeline_events(
                request_id=request_id,
                lifecycle=lifecycle,
                gen_factory=gen_factory,
                send_json=send_json,
            )
        except Exception as e:
            print(f"[WebSocket] Pipeline error ({request_id}): {e}")

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
            k = data.get("k", TOP_K_RETRIEVAL)

            # Keep the receive loop free for cancel / busy while this runs.
            pipeline_task = asyncio.create_task(
                run_one(request_id, question_text, audio_b64, k),
                name=f"s2s-{request_id}",
            )

    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected.")
        lifecycle.on_disconnect()
        if pipeline_task is not None and not pipeline_task.done():
            pipeline_task.cancel()
            try:
                await pipeline_task
            except (asyncio.CancelledError, Exception):
                pass
    except Exception as e:
        print(f"[WebSocket] Error: {e}")
        lifecycle.on_disconnect()
        try:
            await send_json({"type": "error", "message": str(e)})
        except Exception:
            pass

# -------------------------------------------------------------
# Embedded Glassmorphic Web UI
# -------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, tags=["Web UI"])
@app.get("/ui", response_class=HTMLResponse, tags=["Web UI"])
def serve_ui():
    """Serves the interactive Amharic Speech-to-Speech Voice Assistant Client."""
    html_content = """<!DOCTYPE html>
<html lang="am">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Amharic Speech-to-Speech RAG | Live API</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Noto+Sans+Ethiopic:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #090d16;
            --card-bg: rgba(22, 30, 49, 0.7);
            --card-border: rgba(255, 255, 255, 0.08);
            --primary: #6366f1;
            --primary-glow: rgba(99, 102, 241, 0.4);
            --accent: #10b981;
            --accent-glow: rgba(16, 185, 129, 0.3);
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --gold: #f59e0b;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', 'Noto Sans Ethiopic', sans-serif;
        }

        body {
            background-color: var(--bg);
            background-image: 
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(16, 185, 129, 0.12) 0px, transparent 50%);
            color: var(--text);
            min-height: 100vh;
            padding: 24px 16px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .container {
            width: 100%;
            max-width: 960px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        /* Header */
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 18px 24px;
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            border: 1px solid var(--card-border);
            border-radius: 20px;
        }

        .logo-group {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .badge-live {
            background: rgba(16, 185, 129, 0.2);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.4);
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 700;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }

        .badge-live::before {
            content: '';
            width: 8px;
            height: 8px;
            background: #10b981;
            border-radius: 50%;
            box-shadow: 0 0 8px #10b981;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.3); opacity: 0.5; }
        }

        .docs-link {
            color: var(--text-muted);
            text-decoration: none;
            font-size: 0.875rem;
            font-weight: 600;
            padding: 8px 14px;
            border-radius: 12px;
            border: 1px solid var(--card-border);
            transition: all 0.2s;
        }

        .docs-link:hover {
            color: #fff;
            background: rgba(255, 255, 255, 0.05);
            border-color: rgba(255, 255, 255, 0.2);
        }

        /* Voice Interaction Center */
        .voice-card {
            background: var(--card-bg);
            backdrop-filter: blur(20px);
            border: 1px solid var(--card-border);
            border-radius: 28px;
            padding: 36px 24px;
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            gap: 24px;
            box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
            position: relative;
            overflow: hidden;
        }

        .mic-button {
            width: 96px;
            height: 96px;
            border-radius: 50%;
            border: none;
            background: linear-gradient(135deg, #6366f1, #4f46e5);
            color: #fff;
            font-size: 2.2rem;
            cursor: pointer;
            box-shadow: 0 10px 25px var(--primary-glow);
            transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
        }

        .mic-button:hover {
            transform: scale(1.08);
            box-shadow: 0 15px 35px var(--primary-glow);
        }

        .mic-button.recording {
            background: linear-gradient(135deg, #ef4444, #dc2626);
            box-shadow: 0 0 40px rgba(239, 68, 68, 0.7);
            animation: record-pulse 1.5s infinite;
        }

        @keyframes record-pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); }
        }

        .status-label {
            font-size: 1rem;
            font-weight: 500;
            color: var(--text-muted);
        }

        /* Text Input & Quick Questions */
        .input-bar {
            width: 100%;
            max-width: 650px;
            display: flex;
            gap: 10px;
            background: rgba(10, 15, 26, 0.6);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 6px 8px;
        }

        .input-bar input {
            flex: 1;
            background: transparent;
            border: none;
            outline: none;
            color: #fff;
            font-size: 0.95rem;
            padding: 8px 12px;
        }

        .input-bar button {
            background: var(--primary);
            color: #fff;
            border: none;
            padding: 8px 18px;
            border-radius: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
        }

        .input-bar button:hover {
            background: #4f46e5;
        }

        .quick-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            justify-content: center;
            max-width: 700px;
        }

        .chip {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--card-border);
            color: var(--text-muted);
            padding: 6px 14px;
            border-radius: 100px;
            font-size: 0.82rem;
            cursor: pointer;
            transition: all 0.2s;
        }

        .chip:hover {
            background: rgba(99, 102, 241, 0.15);
            color: #fff;
            border-color: rgba(99, 102, 241, 0.4);
        }

        /* Conversation & Timeline Display */
        .grid-layout {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        @media (max-width: 768px) {
            .grid-layout {
                grid-template-columns: 1fr;
            }
        }

        .panel {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 20px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 14px;
            backdrop-filter: blur(12px);
        }

        .panel-title {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            font-weight: 700;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .transcript-box {
            min-height: 120px;
            background: rgba(10, 15, 26, 0.4);
            border: 1px solid var(--card-border);
            border-radius: 14px;
            padding: 14px;
            font-size: 0.95rem;
            line-height: 1.6;
            color: #e2e8f0;
        }

        .sentence-card {
            background: rgba(99, 102, 241, 0.1);
            border-left: 3px solid var(--primary);
            padding: 10px 14px;
            border-radius: 8px;
            margin-bottom: 8px;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Latency Metrics Waterfall */
        .metric-bar {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .metric-row {
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            color: var(--text-muted);
        }

        .progress-track {
            width: 100%;
            height: 8px;
            background: rgba(255, 255, 255, 0.06);
            border-radius: 100px;
            overflow: hidden;
            position: relative;
        }

        .progress-fill {
            height: 100%;
            border-radius: 100px;
            transition: width 0.4s ease;
        }

        .fill-asr { background: #38bdf8; }
        .fill-rag { background: #f59e0b; }
        .fill-llm { background: #818cf8; }
        .fill-tts { background: #34d399; }

        .stat-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
        }

        .stat-card {
            background: rgba(10, 15, 26, 0.5);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 10px;
            text-align: center;
        }

        .stat-val {
            font-size: 1.25rem;
            font-weight: 800;
            color: #fff;
        }

        .stat-lbl {
            font-size: 0.72rem;
            color: var(--text-muted);
        }
    </style>
</head>
<body>

<div class="container">
    <header>
        <div class="logo-group">
            <span style="font-size: 1.5rem;">🇪🇹</span>
            <div>
                <h1 style="font-size: 1.15rem; font-weight: 800;">Amharic Voice RAG API</h1>
                <p style="font-size: 0.78rem; color: var(--text-muted);">Ethiopia Labour Proclamation No. 1156/2019</p>
            </div>
        </div>
        <div style="display: flex; gap: 10px; align-items: center;">
            <span class="badge-live" id="statusBadge">API ONLINE</span>
            <a href="/docs" target="_blank" class="docs-link">📖 Swagger UI</a>
        </div>
    </header>

    <div class="voice-card">
        <button id="micBtn" class="mic-button" title="Click to speak">
            🎙️
        </button>
        <div>
            <div id="statusText" class="status-label">Click microphone to speak in Amharic</div>
            <div id="intentTag" style="font-size: 0.8rem; color: #818cf8; margin-top: 4px; font-weight: 600;"></div>
        </div>

        <div class="input-bar">
            <input type="text" id="textInput" placeholder="ወይም ጥያቄዎን እዚህ ይጻፉ... (e.g. የሙከራ ጊዜ ስንት ቀን ነው?)">
            <button id="sendBtn">ጠይቅ</button>
        </div>

        <div class="quick-chips">
            <button class="chip" onclick="askPreset('ሰላም እንደምን ዋልክ')">👋 ሰላም እንደምን ዋልክ</button>
            <button class="chip" onclick="askPreset('የሙከራ ጊዜ ስንት ቀን ነው?')">⏱️ የሙከራ ጊዜ ስንት ቀን ነው?</button>
            <button class="chip" onclick="askPreset('የዓመት ፈቃድ ስንት ቀን ይሰጣል?')">🏖️ የዓመት ፈቃድ ስንት ቀን?</button>
            <button class="chip" onclick="askPreset('የአሜሪካ የሥራ ሕግ እንዴት ነው?')">🚫 የአሜሪካ የሥራ ሕግ (Refusal)</button>
            <button class="chip" onclick="askPreset('በጣም አመሰግናለሁ')">🙏 በጣም አመሰግናለሁ</button>
        </div>
    </div>

    <div class="grid-layout">
        <!-- Spoken Answer Stream -->
        <div class="panel">
            <div class="panel-title">
                <span>💬 Spoken Output (Streaming)</span>
                <span id="playingIndicator" style="display:none; color: #34d399; font-size: 0.75rem;">🔊 Playing...</span>
            </div>
            <div id="transcriptBox" class="transcript-box">
                <div style="color: var(--text-muted); font-style: italic;">Your conversation and synthesized response will appear here...</div>
            </div>
            <div id="retrievedArticles" style="font-size: 0.8rem; color: #94a3b8;"></div>
        </div>

        <!-- Latency Waterfall Telemetry -->
        <div class="panel">
            <div class="panel-title">
                <span>⚡ Latency & VRAM Telemetry</span>
                <span id="ttfsBadge" style="color: #38bdf8; font-weight: 800;">TTFS: --</span>
            </div>

            <div class="stat-grid">
                <div class="stat-card">
                    <div id="statTTFS" class="stat-val">--</div>
                    <div class="stat-lbl">TTFS (Audio 1)</div>
                </div>
                <div class="stat-card">
                    <div id="statTokS" class="stat-val">--</div>
                    <div class="stat-lbl">LLM tok/s</div>
                </div>
                <div class="stat-card">
                    <div id="statCompute" class="stat-val">--</div>
                    <div class="stat-lbl">Total Compute</div>
                </div>
            </div>

            <div class="metric-bar">
                <div class="metric-row"><span>ASR Transcription</span><span id="tASR">-- ms</span></div>
                <div class="progress-track"><div id="pASR" class="progress-fill fill-asr" style="width: 0%"></div></div>
            </div>

            <div class="metric-bar">
                <div class="metric-row"><span>Hybrid RAG Search</span><span id="tRAG">-- ms</span></div>
                <div class="progress-track"><div id="pRAG" class="progress-fill fill-rag" style="width: 0%"></div></div>
            </div>

            <div class="metric-bar">
                <div class="metric-row"><span>LLM Time to First Token</span><span id="tLLM">-- ms</span></div>
                <div class="progress-track"><div id="pLLM" class="progress-fill fill-llm" style="width: 0%"></div></div>
            </div>

            <div class="metric-bar">
                <div class="metric-row"><span>OmniVoice TTS (1st Sentence)</span><span id="tTTS">-- ms</span></div>
                <div class="progress-track"><div id="pTTS" class="progress-fill fill-tts" style="width: 0%"></div></div>
            </div>
        </div>
    </div>
</div>

<script>
    let ws = null;
    let mediaRecorder = null;
    let micStream = null;
    let audioChunks = [];
    let audioQueue = [];
    let isPlayingAudio = false;
    let currentAudio = null;
    let activeRequestId = null;
    let requestInFlight = false;
    let onStopHandled = false;
    let reconnectAttempt = 0;
    let reconnectTimer = null;
    let requestTimeoutId = null;
    let intentionalClose = false;

    const MAX_RECONNECT_ATTEMPTS = 8;
    const RECONNECT_BASE_MS = 1000;
    const RECONNECT_MAX_MS = 15000;
    const REQUEST_TIMEOUT_MS = 120000;

    const micBtn = document.getElementById('micBtn');
    const statusText = document.getElementById('statusText');
    const intentTag = document.getElementById('intentTag');
    const transcriptBox = document.getElementById('transcriptBox');
    const textInput = document.getElementById('textInput');
    const sendBtn = document.getElementById('sendBtn');
    const playingIndicator = document.getElementById('playingIndicator');
    const statusBadge = document.getElementById('statusBadge');

    function newClientRequestId() {
        if (window.crypto && crypto.randomUUID) {
            return crypto.randomUUID().slice(0, 12);
        }
        return 'req-' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
    }

    function setMicEnabled(enabled) {
        micBtn.disabled = !enabled;
        micBtn.style.opacity = enabled ? '1' : '0.55';
        micBtn.style.pointerEvents = enabled ? 'auto' : 'none';
    }

    function clearRequestTimeout() {
        if (requestTimeoutId) {
            clearTimeout(requestTimeoutId);
            requestTimeoutId = null;
        }
    }

    function armRequestTimeout() {
        clearRequestTimeout();
        requestTimeoutId = setTimeout(() => {
            if (!requestInFlight) return;
            requestInFlight = false;
            clearPlayback();
            setMicEnabled(true);
            statusText.innerText = "ጊዜ አልቋል — እንደገና ይሞክሩ (Request timed out)";
        }, REQUEST_TIMEOUT_MS);
    }

    function clearPlayback() {
        audioQueue = [];
        isPlayingAudio = false;
        playingIndicator.style.display = 'none';
        if (currentAudio) {
            try {
                currentAudio.onended = null;
                currentAudio.pause();
                currentAudio.src = '';
            } catch (e) { /* ignore */ }
            currentAudio = null;
        }
    }

    function resetRecorderState() {
        onStopHandled = false;
        audioChunks = [];
        if (mediaRecorder) {
            try {
                mediaRecorder.ondataavailable = null;
                mediaRecorder.onstop = null;
                mediaRecorder.onerror = null;
                if (mediaRecorder.state === 'recording' || mediaRecorder.state === 'paused') {
                    mediaRecorder.stop();
                }
            } catch (e) { /* ignore */ }
            mediaRecorder = null;
        }
        micBtn.classList.remove('recording');
    }

    function micStreamIsLive() {
        return !!(micStream && micStream.getTracks().some(t => t.readyState === 'live'));
    }

    async function ensureMicStream() {
        if (micStreamIsLive()) return micStream;
        releaseMicStream();
        micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        return micStream;
    }

    function releaseMicStream() {
        resetRecorderState();
        if (micStream) {
            micStream.getTracks().forEach(t => {
                try { t.stop(); } catch (e) { /* ignore */ }
            });
            micStream = null;
        }
    }

    function isStaleEvent(ev) {
        if (!ev || !ev.request_id) return false;
        if (!activeRequestId) return false;
        if (ev.type === 'error' && ev.code === 'busy') return false;
        return ev.request_id !== activeRequestId;
    }

    function markRequestFinished(ev) {
        if (!ev) return;
        if (ev.request_id && activeRequestId && ev.request_id !== activeRequestId) {
            return;
        }
        if (ev.type === 'done' || ev.type === 'cancelled' ||
            (ev.type === 'error' && ev.code !== 'busy')) {
            requestInFlight = false;
            clearRequestTimeout();
            setMicEnabled(true);
        }
    }

    function scheduleReconnect() {
        if (reconnectTimer || intentionalClose) return;
        if (reconnectAttempt >= MAX_RECONNECT_ATTEMPTS) {
            statusBadge.innerText = "OFFLINE";
            statusText.innerText = "ግንኙነት አልተሳካም (Reconnect failed)";
            setMicEnabled(false);
            releaseMicStream();
            clearPlayback();
            return;
        }
        const delay = Math.min(
            RECONNECT_MAX_MS,
            RECONNECT_BASE_MS * Math.pow(2, reconnectAttempt)
        );
        reconnectAttempt += 1;
        statusBadge.innerText = "RECONNECTING...";
        reconnectTimer = setTimeout(() => {
            reconnectTimer = null;
            connectWS();
        }, delay);
    }

    function connectWS() {
        if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
            return;
        }
        const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${proto}//${window.location.host}/ws/s2s`;
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            console.log("Connected to Speech-to-Speech WebSocket");
            statusBadge.innerText = "API CONNECTED";
            reconnectAttempt = 0;
            if (reconnectTimer) {
                clearTimeout(reconnectTimer);
                reconnectTimer = null;
            }
            if (!requestInFlight) setMicEnabled(true);
        };

        ws.onmessage = (e) => {
            let ev;
            try {
                ev = JSON.parse(e.data);
            } catch (err) {
                console.error("Invalid WS JSON", err);
                return;
            }
            handleServerEvent(ev);
        };

        ws.onerror = () => {
            statusBadge.innerText = "SOCKET ERROR";
        };

        ws.onclose = () => {
            requestInFlight = false;
            clearRequestTimeout();
            resetRecorderState();
            clearPlayback();
            setMicEnabled(false);
            scheduleReconnect();
        };
    }
    connectWS();

    function handleServerEvent(ev) {
        if (isStaleEvent(ev)) {
            console.log("Ignoring stale event", ev.type, ev.request_id);
            return;
        }

        if (ev.type === "accepted") {
            activeRequestId = ev.request_id;
            requestInFlight = true;
            setMicEnabled(false);
            armRequestTimeout();
            return;
        }

        if (ev.type === "error" && ev.code === "busy") {
            activeRequestId = ev.active_request_id || activeRequestId;
            requestInFlight = true;
            statusText.innerText = "ጥያቄ በሂደት ላይ ነው... (Busy — wait for current answer)";
            markRequestFinished(ev);
            return;
        }

        if (ev.type === "cancelled") {
            statusText.innerText = "ጥያቄ ተሰርዟል (Cancelled)";
            clearPlayback();
            markRequestFinished(ev);
            return;
        }

        if (ev.type === "cancel_ack") {
            return;
        }

        if (ev.type === "transcript") {
            statusText.innerText = `ጥያቄ: "${ev.text}"`;
            transcriptBox.innerHTML = `<div style="color: #94a3b8; margin-bottom: 8px;"><b>👤 እርስዎ:</b> ${ev.text}</div>`;
            if (ev.asr_metrics) {
                document.getElementById('tASR').innerText = `${Math.round(ev.asr_metrics.total_s * 1000)} ms`;
                document.getElementById('pASR').style.width = '100%';
            }
        } else if (ev.type === "intent") {
            intentTag.innerText = `Intent: ${ev.intent} ${ev.is_direct ? '(Zero-Latency Direct)' : '(RAG Pipeline)'}`;
        } else if (ev.type === "retrieved") {
            document.getElementById('tRAG').innerText = `${Math.round(ev.retrieval_timing.total_ms)} ms`;
            document.getElementById('pRAG').style.width = '100%';
            const ids = ev.results.map(r => r.article_id || r.id).join(', ');
            document.getElementById('retrievedArticles').innerText = `Retrieved Articles: ${ids} (Dense Top: ${ev.top_dense.toFixed(3)})`;
        } else if (ev.type === "first_token") {
            document.getElementById('tLLM').innerText = `${Math.round(ev.t * 1000)} ms`;
            document.getElementById('pLLM').style.width = '100%';
        } else if (ev.type === "audio") {
            const card = document.createElement('div');
            card.className = 'sentence-card';
            card.innerHTML = `<b>🤖 ረዳት:</b> ${ev.sentence}`;
            transcriptBox.appendChild(card);

            if (ev.is_first) {
                const ttfs = ev.t.toFixed(2);
                document.getElementById('statTTFS').innerText = `${ttfs}s`;
                document.getElementById('ttfsBadge').innerText = `TTFS: ${ttfs}s`;
                document.getElementById('tTTS').innerText = `${Math.round(ev.t * 1000)} ms`;
                document.getElementById('pTTS').style.width = '100%';
            }

            queueAudio(ev.audio_b64);
        } else if (ev.type === "done") {
            statusText.innerText = "መልስ ተጠናቋል (Ready)";
            if (ev.stage_timings) {
                const st = ev.stage_timings;
                if (st.asr_s) {
                    document.getElementById('tASR').innerText = `${Math.round(st.asr_s * 1000)} ms`;
                    document.getElementById('pASR').style.width = '100%';
                }
                if (st.retrieval_s) {
                    document.getElementById('tRAG').innerText = `${Math.round(st.retrieval_s * 1000)} ms`;
                    document.getElementById('pRAG').style.width = '100%';
                }
                if (st.llm_ttft_s) {
                    document.getElementById('tLLM').innerText = `${Math.round(st.llm_ttft_s * 1000)} ms`;
                    document.getElementById('pLLM').style.width = '100%';
                }
                if (st.tts_first_s || st.first_audio_s) {
                    const ttsMs = Math.round((st.tts_first_s || st.first_audio_s) * 1000);
                    document.getElementById('tTTS').innerText = `${ttsMs} ms`;
                    document.getElementById('pTTS').style.width = '100%';
                }
                if (st.first_audio_s) {
                    document.getElementById('statTTFS').innerText = `${st.first_audio_s.toFixed(2)}s`;
                    document.getElementById('ttfsBadge').innerText = `TTFS: ${st.first_audio_s.toFixed(2)}s`;
                }
            }
            if (ev.metrics) {
                document.getElementById('statCompute').innerText = `${ev.total_compute_s.toFixed(2)}s`;
                if (ev.metrics.tokens_per_sec) {
                    document.getElementById('statTokS').innerText = `${ev.metrics.tokens_per_sec.toFixed(1)}`;
                }
            } else if (ev.total_compute_s != null) {
                document.getElementById('statCompute').innerText = `${ev.total_compute_s.toFixed(2)}s`;
            }
            markRequestFinished(ev);
        } else if (ev.type === "error") {
            const msg = ev.message || "unknown error";
            statusText.innerText = (ev.code === "timeout")
                ? ("ጊዜ አልቋል: " + msg)
                : ("ስህተት: " + msg);
            clearPlayback();
            markRequestFinished(ev);
        }
    }

    function queueAudio(b64) {
        if (!b64) return;
        // Same WAV twice in a row (e.g. IDK cache replay) — keep one.
        if (audioQueue.length && audioQueue[audioQueue.length - 1] === b64) return;
        if (isPlayingAudio && currentAudio && currentAudio.src &&
            currentAudio.src.endsWith(b64)) return;
        audioQueue.push(b64);
        if (!isPlayingAudio) {
            playNextAudio();
        }
    }

    function playNextAudio() {
        if (audioQueue.length === 0) {
            isPlayingAudio = false;
            playingIndicator.style.display = 'none';
            currentAudio = null;
            return;
        }

        isPlayingAudio = true;
        playingIndicator.style.display = 'inline';
        const b64 = audioQueue.shift();
        const snd = new Audio("data:audio/wav;base64," + b64);
        currentAudio = snd;
        snd.onended = () => {
            if (currentAudio === snd) currentAudio = null;
            playNextAudio();
        };
        snd.play().catch(e => {
            console.error("Audio play error:", e);
            if (currentAudio === snd) currentAudio = null;
            playNextAudio();
        });
    }

    // Convert the browser-native MediaRecorder container (usually WebM/Opus)
    // into a real PCM WAV before sending it to the ASR service.
    function encodeWav(audioBuffer) {
        const channels = audioBuffer.numberOfChannels;
        const frames = audioBuffer.length;
        const bytesPerSample = 2;
        const blockAlign = channels * bytesPerSample;
        const buffer = new ArrayBuffer(44 + frames * blockAlign);
        const view = new DataView(buffer);

        function writeString(offset, value) {
            for (let i = 0; i < value.length; i++) {
                view.setUint8(offset + i, value.charCodeAt(i));
            }
        }

        writeString(0, 'RIFF');
        view.setUint32(4, 36 + frames * blockAlign, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, channels, true);
        view.setUint32(24, audioBuffer.sampleRate, true);
        view.setUint32(28, audioBuffer.sampleRate * blockAlign, true);
        view.setUint16(32, blockAlign, true);
        view.setUint16(34, 16, true);
        writeString(36, 'data');
        view.setUint32(40, frames * blockAlign, true);

        const channelData = [];
        for (let channel = 0; channel < channels; channel++) {
            channelData.push(audioBuffer.getChannelData(channel));
        }

        let offset = 44;
        for (let frame = 0; frame < frames; frame++) {
            for (let channel = 0; channel < channels; channel++) {
                const sample = Math.max(-1, Math.min(1, channelData[channel][frame]));
                view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
                offset += bytesPerSample;
            }
        }
        return new Blob([buffer], { type: 'audio/wav' });
    }

    async function mediaBlobToWav(blob) {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        try {
            const encoded = await blob.arrayBuffer();
            const decoded = await audioContext.decodeAudioData(encoded);
            return encodeWav(decoded);
        } finally {
            await audioContext.close();
        }
    }

    function canSendOnSocket() {
        return ws && ws.readyState === WebSocket.OPEN;
    }

    function sendPipelineMessage(payload) {
        if (!canSendOnSocket()) {
            statusText.innerText = "ግንኙነት የለም — እንደገና በመገናኘት ላይ... (Socket not ready)";
            setMicEnabled(true);
            return false;
        }
        if (requestInFlight) {
            statusText.innerText = "ጥያቄ በሂደት ላይ ነው... (Busy — wait for current answer)";
            return false;
        }
        const request_id = payload.request_id || newClientRequestId();
        payload.request_id = request_id;
        activeRequestId = request_id;
        requestInFlight = true;
        clearPlayback();
        setMicEnabled(false);
        armRequestTimeout();
        ws.send(JSON.stringify(payload));
        return true;
    }

    micBtn.onclick = async () => {
        if (mediaRecorder && mediaRecorder.state === "recording") {
            try {
                mediaRecorder.stop();
            } catch (e) {
                console.error(e);
            }
            micBtn.classList.remove('recording');
            statusText.innerText = "ድምፅ በማቀነባበር ላይ... (Processing)";
            return;
        }

        if (!canSendOnSocket()) {
            statusText.innerText = "ግንኙነት የለም — ማይክ አይገኝም (Socket not ready)";
            return;
        }
        if (requestInFlight) {
            statusText.innerText = "ጥያቄ በሂደት ላይ ነው... (Busy)";
            return;
        }

        try {
            const stream = await ensureMicStream();
            resetRecorderState();
            onStopHandled = false;
            audioChunks = [];
            mediaRecorder = new MediaRecorder(stream);

            mediaRecorder.ondataavailable = (e) => {
                if (e.data && e.data.size > 0) audioChunks.push(e.data);
            };

            mediaRecorder.onerror = (e) => {
                console.error("MediaRecorder error:", e);
                statusText.innerText = "መቅረጽ ስህተት (Recorder error)";
                resetRecorderState();
                setMicEnabled(true);
            };

            mediaRecorder.onstop = async () => {
                if (onStopHandled) return;
                onStopHandled = true;
                const chunks = audioChunks.slice();
                const mimeType = (mediaRecorder && mediaRecorder.mimeType) || 'audio/webm';
                mediaRecorder = null;
                audioChunks = [];
                micBtn.classList.remove('recording');

                try {
                    if (!chunks.length) {
                        statusText.innerText = "ምንም ድምፅ አልተቀረጸም (Empty recording)";
                        setMicEnabled(true);
                        return;
                    }
                    const recordedBlob = new Blob(chunks, { type: mimeType });
                    const wavBlob = await mediaBlobToWav(recordedBlob);
                    const reader = new FileReader();
                    reader.readAsDataURL(wavBlob);
                    reader.onloadend = () => {
                        const base64Data = reader.result.split(',')[1];
                        resetTelemetry();
                        const ok = sendPipelineMessage({
                            type: "audio",
                            audio_b64: base64Data
                        });
                        if (!ok) {
                            statusText.innerText = "መላክ አልተቻለም (Send failed)";
                            setMicEnabled(true);
                        }
                    };
                    reader.onerror = () => {
                        statusText.innerText = "ፋይል ማንበብ አልተቻለም (File read failed)";
                        setMicEnabled(true);
                    };
                } catch (err) {
                    console.error("Audio conversion error:", err);
                    statusText.innerText = "ድምፅን መቀየር አልተቻለም (Audio conversion failed)";
                    setMicEnabled(true);
                }
            };

            mediaRecorder.start();
            micBtn.classList.add('recording');
            statusText.innerText = "እያዳመጥኩ ነው... ተናግረው ሲጨርሱ ማይክራፎኑን ይጫኑ (Listening...)";
        } catch (err) {
            releaseMicStream();
            setMicEnabled(true);
            alert("Microphone access error: " + err);
        }
    };

    function resetTelemetry() {
        clearPlayback();
        document.getElementById('statTTFS').innerText = '--';
        document.getElementById('statTokS').innerText = '--';
        document.getElementById('statCompute').innerText = '--';
        document.getElementById('tASR').innerText = '-- ms';
        document.getElementById('tRAG').innerText = '-- ms';
        document.getElementById('tLLM').innerText = '-- ms';
        document.getElementById('tTTS').innerText = '-- ms';
        document.querySelectorAll('.progress-fill').forEach(f => f.style.width = '0%');
    }

    function sendTextQuery(q) {
        if (!q.trim()) return;
        resetTelemetry();
        statusText.innerText = `ጥያቄ: "${q}"`;
        const ok = sendPipelineMessage({ type: "text", text: q });
        if (!ok) setMicEnabled(canSendOnSocket() && !requestInFlight);
    }

    sendBtn.onclick = () => {
        sendTextQuery(textInput.value);
        textInput.value = '';
    };

    textInput.onkeydown = (e) => {
        if (e.key === 'Enter') {
            sendTextQuery(textInput.value);
            textInput.value = '';
        }
    };

    window.askPreset = function(text) {
        sendTextQuery(text);
    };

    // Release mic if the tab is hidden for a long time or the page unloads.
    window.addEventListener('pagehide', () => {
        intentionalClose = true;
        releaseMicStream();
        clearPlayback();
        clearRequestTimeout();
        if (reconnectTimer) clearTimeout(reconnectTimer);
    });
</script>

</body>
</html>
"""
    return HTMLResponse(content=html_content)
