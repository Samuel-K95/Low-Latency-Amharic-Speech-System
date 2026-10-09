#!/usr/bin/env python3
"""
Server Configuration for Amharic Speech-to-Speech RAG API.
Supports dynamic single-GPU (Colab) and dual-GPU (Kaggle) topologies.
"""
import os
import sys
from pathlib import Path
import torch

# Base directories resolution
SERVER_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SERVER_DIR.parent

# Detect execution environment
IS_KAGGLE = os.path.exists("/kaggle")
IS_COLAB = "google.colab" in sys.modules or os.path.exists("/content")

# Hardware & Multi-GPU configuration
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
MULTI_GPU = NUM_GPUS >= 2

# Device allocation topology
# Kaggle Dual GPU (2x T4): GPU 0 runs ASR, Dense Retrieval & TTS; GPU 1 runs dedicated llama-server
# Colab Single GPU (1x T4): All models fit comfortably within ~10.1 GB VRAM on GPU 0
PRIMARY_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
ASR_DEVICE = PRIMARY_DEVICE
RETRIEVAL_DEVICE = PRIMARY_DEVICE
TTS_DEVICE = PRIMARY_DEVICE

LLM_GPU_ID = 1 if MULTI_GPU else 0
LLM_PORT = int(os.environ.get("LLM_PORT", 8081))
LLM_HOST = "127.0.0.1"
LLM_SERVER_URL = f"http://{LLM_HOST}:{LLM_PORT}"

# Project Paths
DATA_DIR = PROJECT_DIR / "data"
INDEX_DIR = PROJECT_DIR / "index"
BIN_DIR = PROJECT_DIR / "bin"
SYSTEM_PROMPT_PATH = PROJECT_DIR / "system_prompt.txt"

# Working directory for logs & temp files
if IS_KAGGLE:
    WORKING_DIR = Path("/kaggle/working")
elif IS_COLAB:
    WORKING_DIR = Path("/content")
else:
    WORKING_DIR = PROJECT_DIR / "scratch"
WORKING_DIR.mkdir(parents=True, exist_ok=True)

SERVER_LOG_PATH = WORKING_DIR / "llama_server.log"
GGUF_DIR = WORKING_DIR / "gguf"
GGUF_DIR.mkdir(parents=True, exist_ok=True)

# Exact, production-verified model identifiers
ASR_MODEL_ID = os.environ.get("ASR_MODEL_ID", "snapwre/hohe-asr-amharic")
EMBEDDING_MODEL_ID = os.environ.get("EMBEDDING_MODEL_ID", "intfloat/multilingual-e5-base")
TTS_MODEL_ID = os.environ.get("TTS_MODEL_ID", "african-low-resource/omnivoice-amharic")

# GGUF Repositories and Quantizations
LLAMA_REPOS = [
    "bartowski/google_gemma-4-E4B-it-GGUF",
    "unsloth/gemma-4-E4B-it-GGUF",
    "ggml-org/gemma-4-E4B-it-GGUF"
]
LLAMA_QUANT = "Q4_K_M"
LLAMA_QUANT_FALLBACKS = ["Q4_0", "Q8_0"]
LLAMA_CTX = 4096

# Audio Settings
ASR_SAMPLE_RATE = 16000
TTS_SAMPLE_RATE = 24000
TTS_STEPS_FIRST = 8      # Fast 8-step synthesis for TTFS < 1.0s
TTS_STEPS_FULL = 16      # High-fidelity 16-step synthesis for subsequent sentences
TTS_GUIDANCE = 2.0
TTS_SEED = 42
TTS_REFINE_FINAL = False
CHUNK_GAP_S = 0.12

# Fixed male voice lock (OmniVoice voice cloning).
# Without this, generate() auto-picks a new voice each call and the speaker drifts between runs.
TTS_VOICE_LOCK = os.environ.get("TTS_VOICE_LOCK", "1") not in ("0", "false", "False")
TTS_VOICE_REF_SEED = int(os.environ.get("TTS_VOICE_REF_SEED", "42"))
TTS_VOICE_REF_TEXT = os.environ.get(
    "TTS_VOICE_REF_TEXT",
    "ሰላም፣ እኔ ረዳትዎ ነኝ፣ ስለ ኢትዮጵያ የሠራተኛ ሕግ ጥያቄዎችን እመልሳለሁ።",
)
TTS_VOICE_REF_PATH = WORKING_DIR / "voice_ref_male.wav"

# Retrieval & Intent Parameters
ALPHA_HYBRID = 0.6       # 60% Dense E5-Base + 40% Sparse Character TF-IDF
TOP_K_RETRIEVAL = 3
MIN_DENSE_THRESHOLD = 0.81

# LLM Generation Parameters
# 200 matches the Kaggle notebook; enough for 1-2 Amharic sentences without bloating TTFS.
# Override with LLM_MAX_TOKENS=256 if you want slightly longer answers.
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "200"))
LLM_TEMPERATURE = 0.0    # 0.0 for deterministic, reproducible answers
LLM_REPEAT_PENALTY = 1.05

# Multi-turn chat history (LLM messages only — never fed into retrieval).
# 2 turns = last two (user, assistant) pairs. Cap keeps prefill tiny vs RAG context.
CHAT_HISTORY_ENABLED = os.environ.get("CHAT_HISTORY_ENABLED", "1") not in ("0", "false", "False")
CHAT_HISTORY_MAX_TURNS = int(os.environ.get("CHAT_HISTORY_MAX_TURNS", "2"))
CHAT_HISTORY_MAX_CHARS = int(os.environ.get("CHAT_HISTORY_MAX_CHARS", "280"))

# Response Strings
IDK_TEXT = "ይቅርታ፣ ለዚህ ጥያቄ መልስ የሚሆን መረጃ የለኝም።"
FALLBACK_REPLY = "ሰላም! እንዴት ልረዳህ እችላለሁ?"

# Optimization switches
IMPROVEMENTS = {
    "early_first_words": 4,     # Cut first phrase after 4 complete words on comma boundary
    "router": True,             # Zero-latency conversational intent bypass
    "answer_cache": True,       # Cosine cache for identical/near-identical queries
    "numbers_to_words": True,   # Convert numbers to spoken Amharic words for TTS
    "cache_min": 0.95,          # Minimum similarity to trigger semantic answer cache hit
    "chat_history": True,       # Last-N turns in LLM messages only (see CHAT_HISTORY_*)
}

# Phase 5 latency / payload controls (architecture unchanged)
PIPELINE_TIMEOUT_S = float(os.environ.get("PIPELINE_TIMEOUT_S", "90"))
ANSWER_CACHE_MAX = int(os.environ.get("ANSWER_CACHE_MAX", "64"))
ANSWER_CACHE_WARM_N = int(os.environ.get("ANSWER_CACHE_WARM_N", "12"))
# When True, WebSocket streaming omits redundant full joined WAV (sentences already streamed).
WS_OMIT_FULL_AUDIO = os.environ.get("WS_OMIT_FULL_AUDIO", "1") not in ("0", "false", "False")
LLM_HEALTH_TTL_S = float(os.environ.get("LLM_HEALTH_TTL_S", "30"))
