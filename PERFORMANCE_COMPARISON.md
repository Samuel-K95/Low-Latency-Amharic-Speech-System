# Performance Comparison: Research Baseline vs Final Optimized System

Side-by-side view of the two branch benchmark reports:

| Branch | Document |
|--------|----------|
| [`Research-experiments`](https://github.com/Samuel-K95/Low-Latency-Amharic-Speech-System/tree/Research-experiments) | [`PERFORMANCE_BENCHMARKS.md`](https://github.com/Samuel-K95/Low-Latency-Amharic-Speech-System/blob/Research-experiments/PERFORMANCE_BENCHMARKS.md) (Week 6) |
| [`final-optimized-system`](https://github.com/Samuel-K95/Low-Latency-Amharic-Speech-System/tree/final-optimized-system) | [`PERFORMANCE_BENCHMARKS.md`](https://github.com/Samuel-K95/Low-Latency-Amharic-Speech-System/blob/final-optimized-system/PERFORMANCE_BENCHMARKS.md) (Week 7) |

**Primary metric:** Time to First Audio (TTFA) — end of user speech → first valid response audio.

---

## Snapshot

| | Research-experiments (Week 6) | final-optimized-system (Week 7) |
|--|-------------------------------|----------------------------------|
| **Role** | Model selection + cascade baseline | Production S2S **RAG** + latency packaging |
| **Stack** | hohe ASR → **Tiny Aya Earth** → OmniVoice 16 | hohe ASR → **retrieval** → **Gemma 4 E4B** (4-bit) → OmniVoice |
| **Hardware** | 1× Tesla T4 | **2× Tesla T4** |
| **Task** | Open cascade (FLEURS / Belebele / TTS suite) | Labour Proclamation 1156/2019 Q&A (43 spoken legal questions) |
| **Headline TTFA** | **3.78 s** median (measured E2E, +400 ms VAD) | Sequential **4.17 s** → optimized **2.04 s** (−51%) |
| **p90 TTFA** | 5.25 s | Sequential 7.42 s → optimized **3.31 s** |

---

## Latency: how the optimized branch is faster

### Cross-branch TTFA

| Configuration | Median TTFA | vs research baseline (3.78 s) |
|---------------|-------------|-------------------------------|
| Research measured E2E (hohe + Tiny Aya + Omni 16) | 3.78 s | — |
| Final sequential RAG (B0) | 4.17 s | Slightly slower (full answer + retrieval + longer TTS wait) |
| **Final optimized RAG (B4)** | **2.04 s** | **~46% faster** than research E2E (~1.85× sooner) |

Week 6 established that **LLM then TTS** dominate wait time. Week 7 kept hohe + OmniVoice, upgraded the LLM path to Gemma via `llama.cpp`, added grounded retrieval, and cut TTFA by changing **how stages run together**.

### What drove the latency cut (final branch)

| Step | Change | Median TTFA |
|------|--------|-------------|
| B0 | Sequential (full LLM + full TTS) | 4.17 s |
| B1 | Prompt cache | 3.80 s |
| B2 | Stream sentences to TTS | 3.46 s |
| B3 | Short first phrase (8 TTS steps) | 2.46 s |
| **B4** | **Gemma on second GPU + router + answer cache** | **2.04 s** |

Additional fast paths on the optimized system:

| Scenario | Sequential (B0) | Optimized (B4) |
|----------|-----------------|----------------|
| New legal question | 4.17 s | **2.04 s** |
| Small talk | 5.34 s | **0.07 s** |
| Same question again | 4.16 s | **0.09 s** |

### Bottleneck shift

| | Research (measured E2E share) | Final optimized (B4 stage means to first audio) |
|--|-------------------------------|--------------------------------------------------|
| ASR | ~3% (~116 ms) | Small (~0.1 s) |
| Retrieval | — (no RAG) | Small (~0.04 s) |
| LLM | **~52%** (~1984 ms first clause) | Prompt + first phrase ~**0.73 + 0.74 s** |
| TTS | **~31%** (~1167 ms, Omni 16) | First phrase ~**0.62 s** (8 steps) |
| VAD / other | ~11% (400 ms) | Endpointing excluded from final clock |

Research bottleneck: isolated **Tiny Aya first clause + full first OmniVoice clause**.  
Optimized system: **overlap** LLM/TTS, shorter first TTS chunk, dual-GPU, and skip full generation for small talk / repeats.

---

## Accuracy: how the optimized branch is stronger

The suites are not identical (open cascade vs legal RAG), but the final system is **substantially more accurate for the product task** — grounded answers, correct refusals, and solid real-speaker ASR — while getting faster.

### Answer quality (final legal RAG)

| Metric | Sequential B0 | Optimized B4 | Notes |
|--------|---------------|--------------|--------|
| Right passage in top 3 | **97%** | **97%** | Retrieval holds under all latency steps |
| “I don’t know” used correctly | 91% | **93%** | Slightly better refusals when optimized |
| Word overlap F1 vs reference | 37% | **38%** | Stable despite streaming / short phrase |
| Letters garbled when spoken | 1.1% | 2.7% | Small clarity trade for −51% TTFA |

Research Week 6 had **no retrieval grounding** — Tiny Aya answered from model knowledge (Belebele MCQ **63.5% ± 6.7%**). The final stack answers from **Labour Proclamation** context with **97%** top-3 hit rate and disciplined IDK behavior — a large jump in task accuracy for the legal assistant.

### Speech recognition

| | Research (hohe on FLEURS) | Final (hohe on WAXAL humans) |
|--|---------------------------|------------------------------|
| Set | 400 FLEURS variants | 100 real speakers, 19 talkers |
| CER | 0.116 overall; **0.060** clean | **9.1%** overall (half of clips &lt; 6.3% CER) |
| WER | 0.277 overall; **0.165** clean | **21.8%** |
| Throughput | RTF ~0.013 (T4) | ~**40×** real time (28.3 min → 42.5 s) |

Same ASR family (**hohe**); final also reports clean TTS-spoken legal questions at CER **0.7%** / WER **3.6%**.

### TTS intelligibility

| | Research (Omni 16 round-trip via hohe) | Final (spoken answers, ASR listen-back) |
|--|----------------------------------------|----------------------------------------|
| Metric | Round-trip CER **0.065** / WER **0.150** | Garbled letters **2.7%** at B4 (vs 1.1% sequential) |
| Tradeoff | Omni beat MMS on quality | Short first phrase (8 steps) buys latency; slight clarity cost |

---

## Stack evolution

| Stage | Research selection | Final optimized |
|-------|--------------------|-----------------|
| ASR | hohe-asr-amharic | hohe-asr-amharic (kept) |
| Knowledge | None (LLM only) | **Hybrid RAG** over proclamation corpus |
| LLM | Tiny Aya Earth (fp16) | **Gemma 4 E4B 4-bit** via `llama.cpp` |
| TTS | OmniVoice 16 | OmniVoice (8-step first phrase, then continue) |
| Packaging | Benchmark + Colab prototype | Modular `amharic_rag/` + API server notebook |

Week 6 locked **hohe + OmniVoice** and showed LLM/TTS dominate TTFA. Week 7 kept those choices, added RAG for accuracy, and applied prompt cache, streaming, short first audio, dual-GPU, router, and answer cache for latency.

---

## Bottom line

1. **Latency:** Research E2E **3.78 s** → final optimized **2.04 s** (~**46%** lower TTFA). Inside the final branch alone: **4.17 → 2.04 s (−51%)**.
2. **Accuracy:** Final system is a **grounded** legal assistant (**97%** top-3 retrieval, **93%** correct IDK) vs research’s ungrounded cascade (Belebele **~63.5%**). Answer relevance for the target domain improves substantially while quality holds across optimization steps.
3. **Fast paths:** Optimized small talk / repeats reach **~0.07–0.09 s**, far below anything in the research cascade suite.
4. **Cost of speed:** A modest rise in spoken garbled letters (**1.1% → 2.7%**) for the largest TTS latency win — accepted in the final ablations.

Full numbers: see each branch’s `PERFORMANCE_BENCHMARKS.md`.
