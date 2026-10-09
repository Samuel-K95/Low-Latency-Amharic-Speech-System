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

Interactive baseline prototype (selected stack):

- **ASR**: Hohe Amharic ASR (`hohe-asr-amharic`)
- **LLM**: Cohere Labs Tiny Aya Earth
- **TTS**: OmniVoice
- Browser microphone UI with instrumented timing

### `ehio_asr_tiny_aya_earth_llm_omnivoice_TTS.ipynb`

Executed Ethio-ASR baseline stack (same LLM/TTS as the hohe prototype):

- **ASR**: Ethio-ASR (`badrex/Ethio-ASR-amharic`) — ASR alternative to hohe
- **LLM**: Tiny Aya Earth
- **TTS**: OmniVoice
- Live conversational turns used to compare Ethio-ASR quality/behavior against the hohe baseline

### `ethio_asr_llm_omnivoice_tts.ipynb`

Earlier Ethio-ASR baseline with an Amharic-specialized LLM:

- **ASR**: Ethio-ASR
- **LLM**: EthioNLP Amharic-LLAMA (+ LoRA)
- **TTS**: OmniVoice
- Documents an alternate LLM path evaluated before settling on Tiny Aya

Together, these notebooks cover measured cascade benchmarks plus the main interactive stack experiments that informed Week 7 model choices (`final-optimized-system`).
