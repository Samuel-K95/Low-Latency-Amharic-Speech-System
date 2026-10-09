# Final Optimized System (Week 7)

Latency-optimized Amharic speech-to-speech RAG system built from the Week 6 research baseline.

These three pieces work together as one system: the modular package, the pipeline notebook, and the API server notebook.

## Contents

### `amharic_rag/`

Modular production package for the Amharic S2S RAG stack:

- **`src/`** — cleaning, chunking, embeddings, indexing, retrieval, prompts, and generation helpers
- **`server/`** — FastAPI app with ASR, LLM, TTS, retrieval, orchestration, and WebSocket streaming
- **`data/` / `index/`** — Labour Proclamation corpus and FAISS/index artifacts
- **`tests/`** — pytest coverage for pipeline, streaming, latency, and regression checks

### `amharic-speech-to-speech-ragv2 (1).ipynb`

End-to-end Colab/notebook pipeline (ASR → RAG → LLM → TTS) that uses `amharic_rag/`:

- **ASR**: Hohe Amharic ASR
- **LLM**: Gemma via `llama.cpp`
- **TTS**: OmniVoice
- RAG retrieval over the indexed legal corpus with an interactive HTML client

### `amharic-speech-to-speech-rag-api-server (13).ipynb`

Deployment notebook that launches the FastAPI server from `amharic_rag/server/`:

- Multi-GPU service wiring (ASR / LLM / TTS / retrieval)
- WebSocket streaming and optional Cloudflare tunnel for public access
- Runs validation tests before exposing the live endpoint

## How they relate

1. `amharic_rag/` is the shared implementation.
2. `ragv2` demonstrates and exercises the full interactive pipeline.
3. `rag-api-server` packages that same stack as a deployable real-time API.

Research baselines and bottleneck analysis live on the `Research-experiments` branch (Week 6).
