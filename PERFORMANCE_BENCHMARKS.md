# Performance Benchmarks: Sequential Baseline vs Optimized Pipeline

Latency and quality results for the Amharic speech-to-speech RAG system on `final-optimized-system`.

**Primary metric:** Time to First Audio (TTFA) — time from end of user speech to the first valid response audio.

**Source:** Kaggle run, 7 Oct 2026 (shared notebook + benchmark cell). Hardware: **2× Tesla T4**. Test set: **43 spoken legal questions** per configuration step. ASR quality also evaluated on **100 WAXAL** human recordings.

Related notebooks:

- [amharic-speech-to-speech-ragv2](https://www.kaggle.com/code/tsegayeshewamare/amharic-speech-to-speech-ragv2)
- [amharic-speech-to-speech-rag-api-server](https://www.kaggle.com/code/samuelkeno/amharic-speech-to-speech-rag-api-server)

---

## Pipeline under test

Same models in every configuration (only scheduling and packaging change):

| Stage | Model / component |
|-------|-------------------|
| ASR | `hohe-asr-amharic` |
| Retrieval | Hybrid retrieval over Labour Proclamation 1156/2019 |
| LLM | Gemma 4 E4B (4-bit, `llama.cpp`) |
| TTS | OmniVoice Amharic |

**Objective:** Cut TTFA by changing how stages run together, not which models run, without dropping answer quality.

**Timing notes:** Clock starts when the recording ends (endpointing / silence wait excluded). Sequential baseline means mean ~4.65 s before audio when Gemma and TTS each wait for a full answer; the stepwise medians below use the 43-question spoken suite.

---

## Headline result

| Configuration | Median TTFA | p90 (slowest 10%) |
|---------------|-------------|-------------------|
| **B0 — Sequential** | **4.17 s** | 7.42 s |
| **B4 — Optimized** | **2.04 s** | 3.31 s |
| **Improvement** | **−51%** (4.17 → 2.04 s) | 7.42 → 3.31 s |

---

## 1. Sequential baseline (B0) — where the wait goes

In B0, Gemma writes the whole answer, then TTS synthesizes the whole answer. Mean breakdown over 43 questions (everything on GPU 0, no prompt cache):

| Stage | Mean time (s) | Notes |
|-------|---------------|--------|
| ASR | (small vs total) | Spoken question → text |
| Retrieval | (small vs total) | Top passages for the prompt |
| Gemma reads prompt | **1.11** | System prompt + 3 passages (~1350 tokens) |
| Gemma writes full answer | **1.39** | Typically ~66 tokens |
| TTS reads full answer | **2.03** | OmniVoice, ~20 words / ~15.6 s speech, 16 steps |
| **Total (mean)** | **~4.65** | Gemma ≈ 54% of the wait before TTS can start |

---

## 2. Optimization steps (cumulative)

Each step keeps the previous ones. Models stay fixed.

| Step | Change | Measured effect |
|------|--------|-----------------|
| **B1** | Prompt cache — `llama-server` keeps the system prompt; only new passages are re-read | Gemma prompt read **1.11 → 0.75 s**; median TTFA **4.17 → 3.80 s** |
| **B2** | Stream finished sentences to TTS while Gemma continues; playback after sentence 1 | Median TTFA **3.80 → 3.46 s** |
| **B3** | Short first phrase — first chunk may end at ፣; **8 TTS steps** instead of 16 | Median TTFA **3.46 → 2.46 s** (largest single drop) |
| **B4** | Gemma on second T4; ASR / TTS / retrieval stay on GPU 0 | Median TTFA **2.46 → 2.04 s**; GPU util **86%/0% → 43%/49%** |
| **B4** | Router for small talk — greetings / thanks / “who are you” → pre-made audio | Small talk **5.34 → 0.07 s** |
| **B4** | Answer cache — exact or ≥0.98-similar question replays stored answer + audio | Repeat questions **4.16 → 0.09 s** |
| **B4** | Numbers → Amharic words before TTS | Supports speech clarity / garbled-letter checks |

### Median TTFA by step (43 spoken questions)

| Step | Description | Median TTFA (s) |
|------|-------------|-----------------|
| B0 | Sequential | 4.17 |
| B1 | + prompt cache | 3.80 |
| B2 | + stream sentences to TTS | 3.46 |
| B3 | + short first phrase (8 TTS steps) | 2.46 |
| B4 | + Gemma on own GPU (optimized) | **2.04** |

---

## 3. Stage-level TTFA (mean seconds)

Approximate stage contributions to TTFA as optimizations stack (prompt-read / write / TTS shrink once only the first phrase is required):

| Configuration | Prompt read | Write (to first audio) | TTS (to first audio) |
|---------------|-------------|------------------------|----------------------|
| B0 sequential | 1.11 | 1.39 | 2.03 |
| B1 + prompt cache | 0.75 | 1.36 | 1.96 |
| B2 + streaming | 0.75 | 0.98 | 1.57 |
| B3 + short first phrase | 0.75 | 0.80 | 0.76 |
| B4 + own GPU | 0.73 | 0.74 | 0.62 |

**Notes:**

- Streaming alone (B1 → B2) is a modest gain: many answers are one sentence, so TTS often still waits for the full reply.
- Short first phrase (B2 → B3) cuts TTS wait roughly in half (**1.57 → 0.76 s**) — largest stage win.
- Own GPU (B3 → B4): first-phrase Gemma **1.55 → 1.48 s**, TTS **0.76 → 0.62 s** (mean stack ~**2.43 → 2.23 s**).

---

## 4. Timeline example (one legal question)

Question: *“For urgent work, how many hours of overtime a day are allowed at most?”*

| | B0 sequential | B4 optimized |
|--|---------------|--------------|
| ASR | 0.09 s | 0.11 s |
| Retrieval | 0.02 s | 0.04 s |
| LLM | 2.32 s (full answer) | 1.46 s (first phrase; rest continues) |
| TTS to first audio | 1.74 s (all 17 words) | first phrase (~11 words); later sentences during playback |
| **TTFA** | **4.17 s** | **2.18 s** (~1.9× sooner) |

After B4, the first phrase plays ~9.1 s while remaining sentences synthesize in the background.

---

## 5. Small talk and repeated questions

| Scenario | B0 median TTFA | B4 median TTFA |
|----------|----------------|----------------|
| New legal question | 4.17 s | 2.04 s |
| Small talk (5 lines) | 5.34 s | **0.07 s** |
| Same question again (43) | 4.16 s | **0.09 s** |

- Router handled **4/5** small-talk lines with pre-made audio (remaining time is mostly ASR). “What is this law about?” correctly went to Gemma.
- Answer cache hit **43/43** repeats (exact match or 0.98 similarity), same answer and audio.
- Three very short lines (`ሰላም`, `ቻው`, `ማን ነህ?`) returned empty ASR when spoken from TTS and were excluded; validate with real voices.

---

## 6. Accuracy and speech quality (same 43 spoken questions)

| Step | Median TTFA | Right passage in top 3 | “I don’t know” used correctly | Word overlap F1 vs reference | Letters garbled when spoken |
|------|-------------|------------------------|-------------------------------|------------------------------|-----------------------------|
| B0 Sequential | 4.17 s | 97% | 91% | 37% | 1.1% |
| B1 + prompt cache | 3.80 s | 97% | 93% | 38% | 1.1% |
| B2 + stream to TTS | 3.46 s | 97% | 93% | 39% | 0.8% |
| B3 + short first phrase | 2.46 s | 97% | 93% | 38% | 3.1% |
| B4 + own GPU (optimized) | **2.04 s** | **97%** | **93%** | **38%** | **2.7%** |

**How measured**

- **Top-3:** reference passage among the 3 passages given to Gemma.
- **F1:** word overlap with the reference answer.
- **Garbled:** ASR listens to the spoken answer (numbers as words).

**Takeaways**

- Retrieval and refusal quality hold across steps (top-3 stays **97%**; IDK **91% → 93%**).
- Garbled letters rise mainly from the 8-step first phrase (**0.8% → 3.1%** at B3); B4 settles at **~2.7%** (~1 letter in 36).
- Exact wording can vary run-to-run on GPU `llama.cpp` (24–33/43 identical between neighbouring steps) while F1 stays **37–39%**.

---

## 7. ASR on real speakers (WAXAL)

Public set: `google/WaxalNLP` `amh_asr` test split — **100** recordings, **19** speakers, **28.3** minutes. Errors after folding homophone letters and writing numbers as words. Model: notebook `hohe-asr-amharic` on 2× T4.

| Cohort | CER (%) | WER (%) |
|--------|---------|---------|
| All (100) | **9.1** | **21.8** |
| Female (65) | 8.1 | 20.9 |
| Male (35) | 11.1 | 23.7 |

- Median CER under **6.3%** for half of recordings.
- Throughput: **28.3 min** of speech transcribed in **42.5 s** (~**40×** real time).
- On the **43** clean TTS-spoken benchmark questions: CER **0.7%**, WER **3.6%** (much easier than real speakers).

---

## 8. Ablations: kept vs dropped

| Change | Before → After | Verdict |
|--------|----------------|---------|
| Prompt cache | 4.17 → 3.80 s | **Kept** |
| Stream sentences to TTS | 3.80 → 3.46 s | **Kept** |
| Short first phrase, 8 TTS steps | 3.46 s / 0.8% garbled → 2.46 s / 3.1% garbled | **Kept** (biggest gain; slight clarity cost) |
| Gemma on own GPU | 2.46 → 2.04 s | **Kept** (needs 2 GPUs) |
| Router for small talk | 5.34 → 0.07 s (4/5 via router) | **Kept** |
| Answer cache (≥0.98 similar) | 4.16 → 0.09 s (43/43) | **Kept** |
| Pad recordings to whole seconds (ASR) | 0.36 → 0.09 s length; transcript unchanged | **Next** (not in pipeline yet) |
| Speak first 6 words before phrase ends | 2.04 s / 2.3% → 1.55 s / 6.5% garbled | **Dropped** (~3× more garbled) |
| `llama.cpp --swa-full` cache | 2.04 s; −0.08 s but 17/43 answers reworded | **Dropped** |
| 2 passages instead of 3 | top-3 95% → 87% | **Dropped** |
| 4 TTS steps instead of 8 for first phrase | ~−0.25 s; **28%** letters garbled | **Dropped** |

---

## Summary

1. **TTFA:** median **4.17 s → 2.04 s (−51%)** with the same ASR / LLM / TTS stack.
2. **Largest win:** short first phrase + fewer TTS steps for the opening chunk.
3. **Quality:** retrieval top-3 and refusal behavior hold; speech clarity trades a small increase in garbled letters for the latency gain.
4. **Fast paths:** small talk and repeats drop to **~0.07–0.09 s** via router and answer cache.
5. **ASR (WAXAL):** CER **9.1%**, WER **21.8%**, ~**40×** real-time on 2× T4.
