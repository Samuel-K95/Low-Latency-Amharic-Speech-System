# Latency research: low-latency Amharic speech-to-speech (Week 6)

The research phase of the project: how speech-to-speech systems work, which open Amharic models exist, where the time goes in
each stage, and which latency methods pay off for Amharic. Everything was measured on a laptop CPU (Intel i5-6300U, 2 cores, no
GPU) between 25 and 30 September 2026. The optimized GPU system is not in this folder.

**Report:** [`Week6_Low_Latency_Amharic_S2S_Report.pdf`](Week6_Low_Latency_Amharic_S2S_Report.pdf) (32 pages): how S2S systems work,
Amharic models (September 2026), CPU measurements of every stage, the frozen baseline, bottleneck analysis, planned experiments
with decision rules, the evaluation protocol and a plan.

## Headline result (laptop CPU, measured)

| Configuration | Time to first audio, median | p90 |
|---|---|---|
| B0 baseline: sequential, Whisper-small → Amharic Llama-3.2-1B → MMS-TTS | 25.2 s | 46.3 s |
| + streaming LLM → TTS, first chunk ≤ 5 words | 9.2 s | 10.6 s |
| + MMS-300M CTC ASR instead of Whisper-small (same CER) | **5.9 s** | **6.8 s** |

Same kinds of models throughout; only how the stages run together and the ASR family change.

## What was measured, and where

| Topic | Script (`smoke/`) | Raw results (`smoke/results/`) |
|---|---|---|
| ASR speed and accuracy on 30 WAXAL test clips (Whisper tiny/small, PyTorch vs CTranslate2 int8; MMS-300M CTC fp32/int8) | `asr_bench.py`, `fetch_waxal.py` | `asr_*.json` |
| Whisper encoder vs decoder time (30 s padding) | `whisper_split.py` | `whisper_encoder_decoder_split.json` |
| End-of-speech detection: pauses in Amharic speech, silence rules, Smart Turn | `endpoint_bench.py` | `endpointing.json` |
| Tokens per Amharic word for each model's tokenizer | `tokenizer_fertility.py` | `tokenizer_fertility.json` |
| LLM time to first token / first clause, prompt cache (llama.cpp) | `llm_bench.py` | `llama_bench_*.json`, `llm_*.json` |
| TTS speed: MMS-TTS on CPU per test sentence (romanization, synthesis, RTF; cold vs warm; 2 vs 4 threads) | `tts_mms_bench.py` | `tts_mms.json` |
| TTS intelligibility: synthesize, transcribe back, compare (finds the dropped digits) | `tts_roundtrip.py` | `tts_roundtrip_mms.json` |
| End-to-end turns: sequential vs streaming vs CTC ASR | `e2e_baseline.py` | `e2e_cpu_*.json` |
| CPU / RAM use | `resources.py` | `resources.json` |
| All of the above in one file | `summarize.py` | `summary.json` |

Shared helpers: `sentences_am.py` (test sentences), `am_text.py` (Amharic text normalization for scoring), `am_numbers.py`
(numbers to Amharic words before TTS).

**Experiments after the report (27-28 September), not in the PDF:**

| Experiment | Script | Results |
|---|---|---|
| Ethio-ASR-amharic, and an Amharic 5-gram language model on the CTC ASR | `asr_bench.py`, `download_ethio.py` | `asr_ethio-asr-amharic_*.json`, `asr_mms300m-waxal_ctc-kenlm.json` |
| Gemma 4 E2B vs Amharic Llama-3.2-1B (thinking off, full sliding-window cache) | `llm_compare.py` | `llm_compare_*.json` |

## Live demo (laptop CPU)

```bash
./talk.sh            # speak Amharic, pause about 0.7 s to finish your turn; Ctrl+C quits
./talk.sh --text     # type Amharic instead of speaking
./talk.sh --help     # all options: --asr ctc|hohe|whisper-small|whisper-tiny, --llm llama1b|gemma|tinyaya, --tts mms|omnivoice
```

Mic → Silero VAD → MMS-300M CTC ASR (starts on pauses while you speak) → Amharic Llama-3.2-1B (llama.cpp, cached prompt, paused
while the first chunk is synthesized) → number normalizer → MMS-TTS (first chunk ≤ 5 words) → speakers. Each turn prints its
latency breakdown and is logged to `smoke/results/live_turns.jsonl`. `./talk_team.sh` runs the models chosen for the final
pipeline (hohe ASR + Tiny Aya + OmniVoice 8-step), which are much slower on a CPU.

`smoke/audio/user_turns/` holds the recorded test questions; `smoke/audio/tts_mms/` the MMS-TTS outputs of the TTS tests.

## Reproduce

Large or third-party files are not in the repository; these commands recreate them.

```bash
python3 -m venv .venv && .venv/bin/pip install -r smoke/requirements.txt
cd smoke
../.venv/bin/python fetch_waxal.py          # 30 WAXAL Amharic test clips -> audio/waxal_amh_test/
../.venv/bin/python download_models.py      # MMS-TTS, Whisper-WAXAL tiny/small, Smart Turn, Amharic Llama-3.2-1B GGUF
../.venv/bin/ct2-transformers-converter --model <whisper-small-waxal-amh snapshot> \
    --output_dir models/whisper-small-waxal-amh-ct2-int8 --quantization int8 \
    --copy_files tokenizer.json preprocessor_config.json
```

* **llama.cpp:** the prebuilt CPU release `b11179` (https://github.com/ggml-org/llama.cpp/releases/tag/b11179), unpacked to
  `smoke/bin/llama-b11179/`.
* **Silero VAD:** `silero_vad.onnx` (https://github.com/snakers4/silero-vad) in `smoke/models/vad/`.
* **Tokenizer test corpus:** any Amharic text file, one sentence per line: `AMHARIC_CORPUS=/path/to/corpus.txt python tokenizer_fertility.py`.

The full list of commands for every number in the report is in its Appendix A.

## Rebuild the PDF

```bash
report/build.sh        # LuaLaTeX x2; needs luaotfload, lualibs, lua-uni-algos and Noto Serif/Sans Ethiopic
```
