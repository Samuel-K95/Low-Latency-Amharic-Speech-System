# Research Experiments (Week 6)

Baseline research, benchmarking, and bottleneck identification for the low-latency Amharic speech-to-speech system.

## Contents

### `Amharic_S2S_Benchmarks (1).ipynb`

End-to-end component benchmarks for an Amharic ASR → LLM → TTS cascade:

- Measures ASR, LLM, and TTS latency and quality independently
- Uses FLEURS (`am_et`) for ASR and Belebele (`amh_Ethi`) for comprehension
- Tracks time-to-first-audio (TTFA), CER, and pipeline latency estimates
- Establishes the Week 6 baseline and identifies major delay sources

### `hohe_asr_tiny_aya_llm_tts_omnivoice_cleaned (2).ipynb`

Interactive prototype of a full speech-to-speech pipeline:

- **ASR**: Hohe Amharic ASR (`hohe-asr-amharic`)
- **LLM**: Cohere Labs Tiny Aya Earth
- **TTS**: OmniVoice
- Browser microphone UI with instrumented timing

Together, these notebooks provide the measured baseline and early pipeline used to decide what to optimize in Week 7 (`final-optimized-system`).
