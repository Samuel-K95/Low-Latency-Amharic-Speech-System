# Performance Benchmarks: Week 6 Baseline Pipeline

Latency and quality analysis for the selected Amharic speech-to-speech baseline:

**hohe-asr-amharic → Tiny Aya Earth → OmniVoice Amharic**

**Primary metric:** Time to First Audio (TTFA) — end of user speech → first valid response audio.

**Sources**

- `Amharic_S2S_Benchmarks (1).ipynb` — component and end-to-end measurements on **Tesla T4**
- Latency optimization experiments (laptop orchestration) summarized from the Week 6 latency deck

Related interactive prototype: `hohe_asr_tiny_aya_llm_tts_omnivoice_cleaned (2).ipynb`

---

## Selected stack

| Stage | Model | Why selected |
|-------|--------|--------------|
| **ASR** | `snapwre/hohe-asr-amharic` | Strong Amharic WER/CER vs waxal-MMS; CTC (no 30 s pad); competitive with Ethio-ASR on clean/noisy/phone |
| **LLM** | Cohere Labs **Tiny Aya Earth** (3.35B, fp16) | Finite logits, Amharic replies, reliable first-clause stops; Belebele ~63.5%; Gemma-3-4B fp16 failed on T4 |
| **TTS** | **OmniVoice** Amharic (default **16** steps, guidance 2.0) | Best round-trip intelligibility vs MMS-TTS; MMS kept as speed alternative |

**Objective:** Establish a measured Week 6 baseline and identify bottlenecks before Week 7 optimization (orchestration and packaging, not only model swaps).

---

## Methodology (T4 notebook)

| Item | Setting |
|------|---------|
| GPU | Tesla T4 |
| ASR / LLM / OmniVoice dtype | fp16 (MMS-TTS fp32) |
| Warmup | 3 untimed calls per model |
| VAD hangover (composed + E2E) | **400 ms** |
| LLM `max_new_tokens` | 80; stop at first clause for first-audio metrics |
| OmniVoice configs tested | (8, 2.0), **(16, 2.0)**, (32, 2.0) |

**Datasets**

- **ASR:** FLEURS `am_et` — 400 clips (100 clean + noisy SNR10 + phone 8 kHz + fast 1.25×); ~70.3 min audio
- **LLM:** 12 team prompts; Belebele `amh_Ethi` 200 MCQ
- **TTS:** 52 sentences (FLEURS + starter categories)
- **E2E measured:** n=20 `fleurs_clean` clips; all three models resident; LLM first clause → OmniVoice 16-step on that clause

---

## Headline result (selected stack)

| Metric | Value | Conditions |
|--------|-------|------------|
| **Measured E2E TTFA (median)** | **3.778 s** | hohe + Tiny Aya + OmniVoice 16; +400 ms VAD; n=20; T4 |
| TTFA p90 | 5.254 s | same |
| TTFA mean | 3.861 s | same |
| Composed TTFA (p50 estimate) | 2.603 s | VAD 400 + ASR p50 + LLM first-clause p50 + TTS first-clause p50 |

This **~3.78 s** figure is the Week 6 team baseline on T4 for the selected stack.

### Measured E2E stage times (median / p90)

| Stage | Median (ms) | p90 (ms) | Share of median TTFA (~3778 ms) |
|-------|-------------|----------|----------------------------------|
| ASR | 116 | 187 | ~3% |
| LLM (first clause) | **1984** | 3014 | **~52%** |
| TTS (first clause, Omni 16) | **1167** | 1727 | **~31%** |
| VAD hangover | 400 | — | ~11% |
| TTFA (no VAD) | 3378 | 4854 | — |
| **TTFA (with VAD)** | **3778** | **5254** | — |

**Bottleneck:** LLM first clause, then TTS. ASR is a small fraction of TTFA on T4.

Peak residency during E2E: ~9.5 GB VRAM (peak ~9.8 GB); GPU util mean ~81% / max 100%.

---

## 1. ASR benchmarks

All-condition summary (n=400 FLEURS variants):

| Model | Params (M) | WER | CER | Latency p50 (ms) | Latency p90 (ms) | RTF p50 |
|-------|------------|-----|-----|------------------|------------------|---------|
| **hohe-asr-amharic** | 606 | **0.277** | 0.116 | **161** | 464 | 0.013 |
| Ethio-ASR-amharic | 606 | 0.266 | **0.102** | 127 | 213 | 0.012 |
| waxal-mms-300m-amh | 316 | 0.485 | 0.230 | 64 | 83 | 0.006 |

### Per condition (WER / CER / lat p50)

| Condition | hohe | Ethio | waxal |
|-----------|------|-------|-------|
| fleurs_clean | **0.165** / 0.060 / 403 ms | 0.172 / **0.051** / 132 ms | 0.275 / 0.086 / 69 ms |
| noisy SNR10 | **0.236** / 0.085 / 118 ms | 0.242 / **0.079** / 134 ms | 0.478 / 0.185 / 63 ms |
| phone 8 kHz | **0.169** / 0.061 / 125 ms | 0.177 / **0.050** / 135 ms | 0.302 / 0.095 / 62 ms |
| fast 1.25× | 0.536 / 0.259 / 399 ms | **0.475** / **0.226** / 105 ms | 0.883 / 0.554 / 65 ms |

**Selection note:** Ethio is slightly faster and better on fast speech; waxal is fastest but much weaker accuracy. **hohe** was chosen as the quality CTC pick for the conversational baseline (clean/noisy/phone WER ties or beats Ethio; far better than waxal).

---

## 2. LLM benchmarks

| Metric | **Tiny Aya Earth** | gemma-3-4b-it (fp16 T4) |
|--------|--------------------|-------------------------|
| Params | 3.35B | 4.3B |
| Logits finite | **Yes** | **No** (invalidates latency/quality) |
| TTFT p50 / p90 | 237 / 253 ms | 89 / 105 ms |
| **First-clause p50 / p90** | **1042 / 1390 ms** | 5735 / 6151 ms |
| Decode tok/s p50 | 21.1 | 14.0 |
| Clause found rate | **1.0** | **0.0** |
| Amharic script ratio | **0.96** | 0.0 |
| Belebele accuracy (n=200) | **63.5% ± 6.7%** | unscored (200 non-finite) |
| Load / VRAM peak | ~354 s / ~6.5 GB | ~478 s / ~8.2 GB |

**Selection note:** Tiny Aya is the only evaluated LLM that produced usable Amharic first clauses and Belebele scores on this T4 setup. Gemma-3-4B fp16 failed completely in this notebook run.

---

## 3. TTS benchmarks (52 sentences)

| Config | First-clause p50 / p90 (ms) | Full p50 (ms) | RTF (full / 1st clause) | Round-trip WER / CER (via hohe) |
|--------|----------------------------|---------------|-------------------------|----------------------------------|
| OmniVoice **s8** g2 | **557 / 880** | 706 | 0.069 / 0.080 | 0.169 / 0.071 |
| OmniVoice **s16** g2 (E2E default) | **1001 / 1543** | 1297 | 0.127 / 0.141 | **0.150 / 0.065** |
| OmniVoice s32 g2 | 1854 / 2975 | 2454 | 0.239 / 0.266 | **0.140 / 0.056** |
| MMS-TTS-amh | **147 / 229** | 195 | 0.023 / 0.024 | 0.261 / 0.092 |

**Selection note:** OmniVoice was selected under a **quality constraint** (better round-trip CER/WER than MMS). MMS remains the speed path (~147 ms first clause) but loses intelligibility, especially on numbers/dates in laptop checks.

---

## 4. Composed TTFA (serial estimate)

Formula: `400 ms VAD + ASR lat_p50 + LLM first_clause_p50 + TTS first_clause_p50`

| ASR | LLM | TTS | Composed TTFA p50 | TTFA p90 (upper) |
|-----|-----|-----|-------------------|------------------|
| hohe | Tiny Aya | Omni **s16** | **2603 ms** | 3797 ms |
| hohe | Tiny Aya | Omni s8 | 2159 ms | 3134 ms |
| hohe | Tiny Aya | MMS | 1749 ms | 2483 ms |
| hohe | Tiny Aya | Omni s32 | 3456 ms | 5229 ms |

Fastest composed combo in the matrix (waxal + Tiny Aya + MMS ≈ **1653 ms**) sacrifices ASR accuracy. **Selected default** remains hohe + Tiny Aya + Omni s16.

Measured E2E (**3778 ms** median) is higher than the composed estimate because live LLM/TTS clause times under full residency exceed the isolated component medians.

---

## 5. Latency optimization findings (CPU laptop suite)

Separate smoke tests on **Intel i5-6300U (2 cores, no GPU)** with 5 spoken turns, exploring orchestration on related Amharic stacks (Whisper/MMS CTC + small Llama + MMS-TTS). These inform *how* to cut wait time around the T4 baseline.

### End-to-end orchestration gains

| Configuration | Median TTFA |
|---------------|-------------|
| BEFORE — serial (full reply before TTS) | **25.2 s** |
| AFTER — stream ~5-word chunks to TTS | **9.2 s** (~2.7×) |
| AFTER + CTC ASR swap | **5.9 s** (−77% from 25.2) |

Example single-turn timeline: **25.2 s → 6.6 s** (~3.8×) with overlapped LLM/TTS and shorter first TTS chunk.

**Projected** with the same class of orchestration on T4: roughly **1.1–2.0 s** TTFA (plan target vs the **3.78 s** measured baseline).

### Amharic interaction / model findings

| Finding | Number |
|---------|--------|
| Pauses per utterance | 7.1 |
| Mid-sentence pause p90 | 480 ms |
| Cut-off rate @ 500 ms silence | 0.53 / utterance |
| Cut-off rate @ 700 ms silence | 0.27 / utterance |
| Smart Turn v3.2 false “finished” | **84%** |
| Tokens/word: Whisper & Llama 3.2 vs Amharic-extended Llama/Gemma | **12.2 vs 1.7** |
| Whisper encoder on ~5 s audio (pads to 30 s) | ~3.8 s |
| CTC vs Whisper CER (test clips) | 12.7% vs 13.0% |
| MMS-TTS date/number round-trip CER (laptop) | ~46% |
| GPU/CPU contention cost on first TTS chunk | ~0.9 s |

### Experiment table (kept / dropped)

| Experiment | Before → After | Verdict |
|------------|----------------|---------|
| Prompt cache (llama.cpp) | 1.50 → **0.42 s** TTFT | Kept |
| Whisper-S CT2 int8 vs PyTorch fp32 | 37.9 → **14.6 s**/clip | Faster; CTC still preferred |
| int8 CTC ASR | 10.8 s @ WER 31.4% → 15.6 s @ WER 33.7% | Dropped |
| Dataset.ET 5-gram on MMS/Ethio | WER 31.4%/25.9% → **26.4%/23.0%** (+0.1–0.2 s) | Kept |
| Transcribe during pauses | 2.4 → **0.6–1.6 s** ASR after speech | Kept |
| Pause LLM during first TTS chunk | 4.4 → **2.1 s** first chunk | Kept |
| Gemma 4 E2B: thinking off + sliding-window cache | ~12 → **3.6 s** to first chunk; answers 6/10 vs 1/10 | Kept |
| Six-word slicing (dev) | 5.9 → **2.6 s** first audio | Kept |
| Llama-1B vs Gemma4 to 5 words | **1.3 s vs 3.6 s** | Routing option |

**Takeaway:** On constrained hardware, **orchestration** (streaming TTS, CTC, pause handling) moved TTFA more than swapping to larger models. Next hard lever called out: Amharic end-of-turn detection (84% false complete with Smart Turn).

---

## 6. Analysis summary

1. **Baseline stack locked:** hohe ASR + Tiny Aya Earth + OmniVoice (16-step) for quality-aware S2S on T4.
2. **Measured TTFA:** median **3.78 s** (p90 **5.25 s**) with 400 ms VAD — LLM (~52%) and TTS (~31%) dominate.
3. **Tradeoffs:** Ethio/waxal ASR and MMS TTS can cut latency but lose accuracy or intelligibility; Gemma-3-4B fp16 was not viable in the notebook run.
4. **Optimization direction:** Keep the selected models; apply streaming first-audio, prompt/answer caching, dual-GPU packing, and better endpointing — laptop suite shows **25.2 → 5.9 s** from orchestration alone, with a T4 target band near **1–2 s**.

Week 7 packaging and further TTFA cuts for the RAG/API system live on the `final-optimized-system` branch.
