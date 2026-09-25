"""E-ASR-1: offline Amharic ASR on laptop CPU (accuracy + latency).

Backends: hf-whisper (transformers fp32), ct2-whisper (faster-whisper int8),
hf-ctc (wav2vec2/w2v-BERT CTC, optionally dynamic-int8). Latency is measured
per utterance = what the user waits after end-of-speech in a non-streaming
pipeline.
"""
import argparse, json, time, pathlib, statistics
import numpy as np, soundfile as sf, torch, jiwer
from am_text import normalize

ap = argparse.ArgumentParser()
ap.add_argument("--backend", required=True, choices=["hf-whisper", "ct2-whisper", "hf-ctc"])
ap.add_argument("--model", required=True)
ap.add_argument("--label", required=True)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--int8", action="store_true", help="hf-ctc: torch dynamic int8 on Linear layers")
ap.add_argument("--beam", type=int, default=1)
ap.add_argument("--limit", type=int, default=30)
ap.add_argument("--lm", help="hf-ctc: KenLM n-gram for beam-search decoding (pyctcdecode)")
ap.add_argument("--alpha", type=float, default=0.3); ap.add_argument("--beta", type=float, default=1.5)
ap.add_argument("--lm-beam", type=int, default=100)
args = ap.parse_args()
torch.set_num_threads(args.threads)

AUD = pathlib.Path("audio/waxal_amh_test")
items = [json.loads(l) for l in open(AUD / "manifest.jsonl", encoding="utf-8")][: args.limit]

t0 = time.perf_counter()
if args.backend == "hf-whisper":
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    proc = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model).eval()
    def transcribe(wav):
        feats = proc(wav, sampling_rate=16000, return_tensors="pt").input_features
        with torch.inference_mode():
            ids = model.generate(feats, num_beams=args.beam, max_new_tokens=440, language="am", task="transcribe")
        return proc.batch_decode(ids, skip_special_tokens=True)[0], ids.shape[-1]
elif args.backend == "ct2-whisper":
    # faster-whisper caps max_length at 448, and CTranslate2's Whisper generate()
    # only emits max_length/2 tokens -> Amharic transcripts get cut at ~224 tokens
    # (~18 words). Call the CT2 model directly with max_length=880 (=440 tokens).
    import ctranslate2
    from faster_whisper import WhisperModel
    model = WhisperModel(args.model, device="cpu", compute_type="int8", cpu_threads=args.threads)
    PROMPT = model.hf_tokenizer.encode("<|startoftranscript|><|am|><|transcribe|><|notimestamps|>",
                                       add_special_tokens=False).ids
    def transcribe(wav):
        feats = model.feature_extractor(wav)[:, :3000]
        feats = np.pad(feats, ((0, 0), (0, 3000 - feats.shape[1])))
        enc = model.model.encode(ctranslate2.StorageView.from_array(np.ascontiguousarray(feats[None])), to_cpu=False)
        ids = model.model.generate(enc, [PROMPT], max_length=880, beam_size=args.beam)[0].sequences_ids[0]
        return model.hf_tokenizer.decode(ids), len(ids)
else:
    from transformers import AutoModelForCTC, AutoProcessor
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForCTC.from_pretrained(args.model, dtype=torch.float32).eval()   # fp16 checkpoints are slow on CPU
    if args.int8:
        model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    lm_time = []
    if args.lm:                                   # Dataset.ET setup: pruned 5-gram, alpha 0.3, beta 1.5
        from pyctcdecode import build_ctcdecoder
        # one label per model output position (MMS vocab files can repeat an id, so never sort the vocab dict)
        toks = proc.tokenizer.convert_ids_to_tokens(list(range(model.config.vocab_size)))
        labels = ["" if t == proc.tokenizer.pad_token else " " if t == "|" else (t or "⁇") for t in toks]
        decoder = build_ctcdecoder(labels, args.lm, alpha=args.alpha, beta=args.beta)
    def transcribe(wav):
        inputs = proc(wav, sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            logits = model(**inputs).logits
        if args.lm:
            t = time.perf_counter()
            text = decoder.decode(torch.log_softmax(logits[0], -1).numpy(), beam_width=args.lm_beam)
            lm_time.append(time.perf_counter() - t)
            return text, 0
        return proc.batch_decode(logits.argmax(-1))[0], 0
load_s = time.perf_counter() - t0
params = sum(p.numel() for p in model.parameters()) / 1e6 if hasattr(model, "parameters") else None

transcribe(np.zeros(16000, dtype=np.float32))          # warm-up
refs, hyps, lat, rtf, ntok = [], [], [], [], []
for it in items:
    wav, sr = sf.read(AUD / it["file"], dtype="float32")
    t = time.perf_counter(); hyp, n = transcribe(wav); dt = time.perf_counter() - t
    refs.append(normalize(it["text"])); hyps.append(normalize(hyp) or "<empty>")
    lat.append(dt); rtf.append(dt / it["duration_s"]); ntok.append(n)
    print(f"{it['file']} dur={it['duration_s']:5.1f}s lat={dt:6.2f}s RTF={dt/it['duration_s']:.2f} tok={n:>3} | {hyp[:60]}", flush=True)

res = {"label": args.label, "backend": args.backend, "model": args.model, "threads": args.threads,
       "int8": args.int8, "beam": args.beam, "n_utts": len(items), "params_M": params and round(params, 1),
       "load_s": round(load_s, 1),
       "wer": round(jiwer.wer(refs, hyps) * 100, 1), "cer": round(jiwer.cer(refs, hyps) * 100, 1),
       "latency_s_median": round(statistics.median(lat), 2), "latency_s_p90": round(float(np.percentile(lat, 90)), 2),
       "rtf_median": round(statistics.median(rtf), 3), "audio_s_median": round(statistics.median(i["duration_s"] for i in items), 1),
       "decoder_tokens_per_s_audio": round(sum(ntok) / sum(i["duration_s"] for i in items), 1) if any(ntok) else None,
       "lm": args.lm and pathlib.Path(args.lm).name, "lm_decode_s_median": round(statistics.median(lm_time), 3) if args.backend == "hf-ctc" and lm_time else None,
       "examples": [{"ref": r, "hyp": h} for r, h in list(zip(refs, hyps))[:3]]}
pathlib.Path(f"results/asr_{args.label}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
print(json.dumps({k: v for k, v in res.items() if k != "examples"}, indent=1))
