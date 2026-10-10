# Performance Benchmarks: Week 6 Baseline Pipeline

Latency and quality results from the Amharic speech-to-speech cascade benchmark notebook.

**Selected stack:** hohe-asr-amharic → Tiny Aya Earth → OmniVoice Amharic (16 steps)

**Primary metric:** Time to First Audio (TTFA) — end of user speech → first valid response audio.

**Source:** `Amharic_S2S_Benchmarks (1).ipynb` only (Tesla T4 Colab run)

---

## Selected stack

| Stage | Model | Why selected (from this notebook) |
|-------|--------|-----------------------------------|
| **ASR** | `snapwre/hohe-asr-amharic` | Strong WER/CER vs waxal-MMS on FLEURS; competitive with Ethio-ASR on clean/noisy/phone |
| **LLM** | Cohere Labs **Tiny Aya Earth** (3.35B, fp16) | Finite logits, Amharic replies, first-clause detection; Belebele ~63.5%; Gemma-3-4B fp16 failed on T4 |
| **TTS** | **OmniVoice** Amharic (**16** steps, guidance 2.0) | Best round-trip CER/WER vs MMS-TTS; MMS is faster but less intelligible |

---

## Methodology

| Item | Setting |
|------|---------|
| GPU | Tesla T4 |
| ASR / LLM / OmniVoice dtype | fp16 (MMS-TTS fp32) |
| Warmup | 3 untimed calls per model |
| VAD hangover (composed + measured E2E) | **400 ms** |
| LLM `max_new_tokens` | 80; stop at first clause for first-audio metrics |
| OmniVoice configs | (8, 2.0), **(16, 2.0)**, (32, 2.0) |

**Datasets**

- **ASR:** FLEURS `am_et` — 400 clips (100 clean + noisy SNR10 + phone 8 kHz + fast 1.25×); ~70.3 min audio
- **LLM:** 12 team prompts; Belebele `amh_Ethi` 200 MCQ
- **TTS:** 52 sentences (FLEURS + starter categories)
- **Measured E2E:** n=20 `fleurs_clean` clips; all three models resident; LLM first clause → OmniVoice 16-step on that clause

---

## Headline result (selected stack)

| Metric | Value | Conditions |
|--------|-------|------------|
| **Measured E2E TTFA (median)** | **3.778 s** | hohe + Tiny Aya + OmniVoice 16; +400 ms VAD; n=20 |
| TTFA p90 | 5.254 s | same |
| TTFA mean | 3.861 s | same |
| Composed TTFA (p50 estimate) | 2.603 s | VAD 400 + ASR p50 + LLM first-clause p50 + TTS first-clause p50 |

### Measured E2E stage times

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

## 1. ASR (Category 1)

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

Ethio is slightly faster and better on fast speech; waxal is fastest but much weaker accuracy. **hohe** was selected as the quality CTC pick for the conversational baseline.

---

## 2. LLM (Category 2)

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

Tiny Aya is the only evaluated LLM that produced usable Amharic first clauses and Belebele scores in this run. Gemma-3-4B fp16 failed completely.

---

## 3. TTS (Category 3)

52 sentences; round-trip intelligibility via hohe ASR:

| Config | First-clause p50 / p90 (ms) | Full p50 (ms) | RTF (full / 1st clause) | Round-trip WER / CER |
|--------|----------------------------|---------------|-------------------------|----------------------|
| OmniVoice **s8** g2 | **557 / 880** | 706 | 0.069 / 0.080 | 0.169 / 0.071 |
| OmniVoice **s16** g2 (E2E default) | **1001 / 1543** | 1297 | 0.127 / 0.141 | **0.150 / 0.065** |
| OmniVoice s32 g2 | 1854 / 2975 | 2454 | 0.239 / 0.266 | **0.140 / 0.056** |
| MMS-TTS-amh | **147 / 229** | 195 | 0.023 / 0.024 | 0.261 / 0.092 |

OmniVoice was selected for **quality** (better round-trip CER/WER). MMS is the speed alternative (~147 ms first clause).

---

## 4. Composed vs measured TTFA (Category 8)

**Composed** estimate: `400 ms VAD + ASR lat_p50 + LLM first_clause_p50 + TTS first_clause_p50`

| ASR | LLM | TTS | Composed TTFA p50 | TTFA p90 (upper) |
|-----|-----|-----|-------------------|------------------|
| hohe | Tiny Aya | Omni **s16** | **2603 ms** | 3797 ms |
| hohe | Tiny Aya | Omni s8 | 2159 ms | 3134 ms |
| hohe | Tiny Aya | MMS | 1749 ms | 2483 ms |
| hohe | Tiny Aya | Omni s32 | 3456 ms | 5229 ms |

Fastest composed combo in the matrix (waxal + Tiny Aya + MMS ≈ **1653 ms**) sacrifices ASR accuracy.

**Measured E2E** for the selected stack (hohe + Tiny Aya + Omni s16, n=20) is higher than the composed estimate (**3778 ms** median) because live LLM/TTS clause times under full residency exceed the isolated component medians.

---

## Summary

1. **Baseline stack:** hohe ASR + Tiny Aya Earth + OmniVoice (16-step).
2. **Measured TTFA:** median **3.78 s** (p90 **5.25 s**) with 400 ms VAD.
3. **Bottlenecks:** LLM (~52%) then TTS (~31%); ASR is small on T4.
4. **Tradeoffs:** Ethio/waxal ASR and MMS TTS can cut latency but lose accuracy or intelligibility; Gemma-3-4B fp16 was not viable in this notebook run.
