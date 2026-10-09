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
- `hohe_asr_tiny_aya_llm_tts_omnivoice_cleaned (2).ipynb` — interactive baseline S2S prototype

### final-optimized-system (Week 7)

- `amharic_rag/` — modular RAG + FastAPI server package
- `amharic-speech-to-speech-ragv2 (1).ipynb` — end-to-end optimized pipeline notebook
- `amharic-speech-to-speech-rag-api-server (13).ipynb` — API server launch / deployment notebook

## Objective

Minimize conversational latency without unacceptable loss of recognition quality, response relevance, or speech naturalness. Architecture and model choices are driven by measured bottlenecks from Week 6, then optimized and packaged in Week 7.
