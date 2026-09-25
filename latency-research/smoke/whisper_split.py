"""E-ASR-2: split Whisper latency into encoder (fixed 30 s window) vs decoder (per token) on short turns."""
import json, time, pathlib, statistics, numpy as np, soundfile as sf, ctranslate2
from faster_whisper import WhisperModel
rows = []
for name, path in [("tiny", "models/whisper-tiny-waxal-amh-ct2-int8"), ("small", "models/whisper-small-waxal-amh-ct2-int8")]:
    m = WhisperModel(path, device="cpu", compute_type="int8", cpu_threads=4)
    prompt = m.hf_tokenizer.encode("<|startoftranscript|><|am|><|transcribe|><|notimestamps|>", add_special_tokens=False).ids
    for f in sorted(pathlib.Path("audio/user_turns").glob("q*.wav")):
        wav, _ = sf.read(f, dtype="float32")
        feats = m.feature_extractor(wav)[:, :3000]; feats = np.pad(feats, ((0, 0), (0, 3000 - feats.shape[1])))
        sv = ctranslate2.StorageView.from_array(np.ascontiguousarray(feats[None]))
        m.model.encode(sv, to_cpu=False)                                  # warm
        t = time.perf_counter(); enc = m.model.encode(sv, to_cpu=False); t_enc = time.perf_counter() - t
        t = time.perf_counter(); ids = m.model.generate(enc, [prompt], max_length=880, beam_size=1)[0].sequences_ids[0]; t_dec = time.perf_counter() - t
        rows.append({"model": name, "file": f.name, "audio_s": round(len(wav) / 16000, 2), "encoder_s": round(t_enc, 2),
                     "decoder_s": round(t_dec, 2), "tokens": len(ids), "ms_per_token": round(t_dec / max(1, len(ids)) * 1000, 1)})
        print(rows[-1], flush=True)
summ = {n: {k: round(statistics.median(r[k] for r in rows if r["model"] == n), 2) for k in ("audio_s", "encoder_s", "decoder_s", "tokens", "ms_per_token")} for n in ("tiny", "small")}
print(json.dumps(summ, indent=1))
pathlib.Path("results/whisper_encoder_decoder_split.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=2))
