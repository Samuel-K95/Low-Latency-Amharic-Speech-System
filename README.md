# Low-Latency Amharic Speech System

Week 6–7 group assignment: reduce end-to-end latency of an Amharic speech-to-speech conversational system, measured primarily as **Time to First Audio** (end of user speech → first valid response audio).

This `main` branch is the index for the project. Implementation work lives on the feature branches below.

## Branches

| Branch | Week | Role |
|--------|------|------|
| [`Research-experiments`](https://github.com/Samuel-K95/Low-Latency-Amharic-Speech-System/tree/Research-experiments) | Week 6 | Baseline system, component benchmarks, bottleneck identification |
| [`final-optimized-system`](https://github.com/Samuel-K95/Low-Latency-Amharic-Speech-System/tree/final-optimized-system) | Week 7 | Optimized S2S RAG pipeline, modular package, and API server |

### Research-experiments (Week 6)

- `Amharic_S2S_Benchmarks (1).ipynb` — ASR / LLM / TTS latency and quality benchmarks
- `hohe_asr_tiny_aya_llm_tts_omnivoice_cleaned (2).ipynb` — interactive baseline (Hohe + Tiny Aya + OmniVoice)
- `ehio_asr_tiny_aya_earth_llm_omnivoice_TTS.ipynb` — Ethio-ASR variant with Tiny Aya + OmniVoice
- `ethio_asr_llm_omnivoice_tts.ipynb` — Ethio-ASR + Amharic-LLAMA + OmniVoice alternate stack
- [`PERFORMANCE_BENCHMARKS.md`](https://github.com/Samuel-K95/Low-Latency-Amharic-Speech-System/blob/Research-experiments/PERFORMANCE_BENCHMARKS.md) — Week 6 cascade results (hohe + Tiny Aya + OmniVoice)

### final-optimized-system (Week 7)

- `amharic_rag/` — modular RAG + FastAPI server package
- `amharic-speech-to-speech-ragv2.ipynb` — end-to-end optimized pipeline notebook
- `amharic-speech-to-speech-rag-api-server.ipynb` — API server launch / deployment notebook
- [`PERFORMANCE_BENCHMARKS.md`](https://github.com/Samuel-K95/Low-Latency-Amharic-Speech-System/blob/final-optimized-system/PERFORMANCE_BENCHMARKS.md) — sequential vs optimized RAG latency and quality

## Performance comparison

See [`PERFORMANCE_COMPARISON.md`](PERFORMANCE_COMPARISON.md) for a side-by-side view of the research baseline vs the final optimized system (TTFA, bottlenecks, and accuracy).

**Headline:** research E2E TTFA **3.78 s** → final optimized **2.04 s**, with grounded RAG accuracy (**97%** top-3 retrieval, **93%** correct refusals).

## Objective

Minimize conversational latency without unacceptable loss of recognition quality, response relevance, or speech naturalness. Architecture and model choices are driven by measured bottlenecks from Week 6, then optimized and packaged in Week 7.
