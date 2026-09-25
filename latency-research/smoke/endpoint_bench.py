"""E-EP-1: endpointing on Amharic speech (Silero VAD + Smart Turn v3.2) on laptop CPU.

1. Silero VAD per-frame cost (32 ms frames).
2. Pause statistics inside Amharic utterances -> how often a pure-silence
   endpointer with hangover H would cut the user off mid-turn.
3. Smart Turn v3.2 (not trained on Amharic): P(complete) at real mid-turn pauses
   vs at the true end of turn, plus its inference cost.
"""
import json, time, pathlib, statistics
import numpy as np, soundfile as sf, onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import WhisperFeatureExtractor

SR, FRAME, CTX = 16000, 512, 64
AUD = pathlib.Path("audio/waxal_amh_test")
items = [json.loads(l) for l in open(AUD / "manifest.jsonl", encoding="utf-8")]

def session(path):
    so = ort.SessionOptions(); so.inter_op_num_threads = 1; so.intra_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])

vad = session("models/vad/silero_vad.onnx")
def vad_probs(wav):
    state = np.zeros((2, 1, 128), np.float32); ctx = np.zeros(CTX, np.float32); probs, cost = [], []
    for i in range(0, len(wav) - FRAME + 1, FRAME):
        x = np.concatenate([ctx, wav[i:i + FRAME]])[None].astype(np.float32)
        t = time.perf_counter()
        out, state = vad.run(None, {"input": x, "state": state, "sr": np.array(SR, np.int64)})
        cost.append(time.perf_counter() - t)
        probs.append(float(out[0][0])); ctx = wav[i + FRAME - CTX:i + FRAME]
    return np.array(probs), cost

st = session(hf_hub_download("pipecat-ai/smart-turn-v3", "smart-turn-v3.2-cpu.onnx"))
fe = WhisperFeatureExtractor(chunk_length=8)
def smart_turn(audio):
    audio = audio[-8 * SR:]
    feats = fe(audio, sampling_rate=SR, return_tensors="np", padding="max_length",
               max_length=8 * SR, truncation=True, do_normalize=True).input_features.astype(np.float32)
    t = time.perf_counter(); p = float(st.run(None, {"input_features": feats})[0].ravel()[0])
    return p, time.perf_counter() - t

frame_ms = FRAME / SR * 1000
all_vad_cost, pauses_ms, st_cost, p_end, p_mid = [], [], [], [], []
for it in items:
    wav, _ = sf.read(AUD / it["file"], dtype="float32")
    probs, cost = vad_probs(wav); all_vad_cost += cost
    speech = probs > 0.5
    idx = np.flatnonzero(speech)
    if len(idx) == 0: continue
    first, last = idx[0], idx[-1]
    # internal silences between first and last speech frame
    run = 0; mids = []
    for k in range(first, last + 1):
        if not speech[k]: run += 1
        else:
            if run: pauses_ms.append(run * frame_ms); mids.append((k - run, run))
            run = 0
    # smart turn at true end (200 ms after last speech) ...
    end_sample = min(len(wav), int((last + 1) * FRAME + 0.2 * SR))
    p, c = smart_turn(wav[:end_sample]); p_end.append(p); st_cost.append(c)
    # ... and at every mid-turn pause >= 200 ms (where a silence endpointer would ask)
    for start, length in mids:
        if length * frame_ms >= 200:
            cut = int((start * FRAME) + 0.2 * SR)
            p, c = smart_turn(wav[:cut]); p_mid.append(p); st_cost.append(c)

pauses = np.array(pauses_ms)
res = {
    "n_utts": len(items), "vad_frame_ms": frame_ms,
    "silero_ms_per_frame_median": round(statistics.median(all_vad_cost) * 1000, 3),
    "silero_realtime_share_pct": round(statistics.mean(all_vad_cost) * 1000 / frame_ms * 100, 2),
    "internal_pauses_per_utt": round(len(pauses) / len(items), 1),
    "pause_ms_p50_p90_p99": [round(float(np.percentile(pauses, q))) for q in (50, 90, 99)],
    "false_endpoints_per_utt_by_hangover_ms": {h: round(float((pauses >= h).sum()) / len(items), 2)
                                               for h in (200, 300, 500, 700, 1000)},
    "utts_with_any_false_endpoint_by_hangover_ms": {h: None for h in ()},
    "smart_turn_ms_median": round(statistics.median(st_cost) * 1000, 1),
    "smart_turn_p_complete_at_end_median": round(statistics.median(p_end), 3),
    "smart_turn_p_complete_at_midpause_median": round(statistics.median(p_mid), 3) if p_mid else None,
    "smart_turn_end_detected_rate@0.5": round(float(np.mean(np.array(p_end) > 0.5)), 2),
    "smart_turn_midpause_false_complete_rate@0.5": round(float(np.mean(np.array(p_mid) > 0.5)), 2) if p_mid else None,
    "n_midpause_probes": len(p_mid),
}
res.pop("utts_with_any_false_endpoint_by_hangover_ms")
pathlib.Path("results/endpointing.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=1))
