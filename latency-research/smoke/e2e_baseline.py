"""E2E-0: first end-to-end Amharic speech-to-speech baseline on the laptop CPU.

TTFA (End-to-End Time to First Audio) = time from the user's last speech frame to
the moment the first playable response audio buffer exists.

  A  sequential : endpoint -> ASR(full) -> LLM(full reply) -> TTS(full reply)
  B  stream     : endpoint -> ASR(full) -> LLM stream -> TTS(first clause) while LLM continues
  C  stream+cap : like B but the first chunk is cut at a clause mark OR 5 words, whichever first

Every stage is timed separately so the bottleneck is visible.
"""
import argparse, json, os, pathlib, re, signal, statistics, subprocess, threading, time, queue, urllib.request
import numpy as np, soundfile as sf, torch, onnxruntime as ort, ctranslate2, uroman as ur
from faster_whisper import WhisperModel
from transformers import VitsModel, AutoTokenizer
from sentences_am import USER_QUESTIONS

ap = argparse.ArgumentParser()
ap.add_argument("--asr", required=True); ap.add_argument("--asr_backend", default="ct2-whisper", choices=["ct2-whisper", "ctc"]); ap.add_argument("--gguf", required=True); ap.add_argument("--label", required=True)
ap.add_argument("--hangover_ms", type=int, default=700); ap.add_argument("--llm_threads", type=int, default=4)
ap.add_argument("--max_tokens", type=int, default=120); ap.add_argument("--modes", default="ABC"); ap.add_argument("--port", type=int, default=8090)
args = ap.parse_args()
SR = 16000
SYSTEM = "አንተ አጋዥ የድምፅ ረዳት ነህ። በአማርኛ በአጭሩ፣ በአንድ ወይም በሁለት ዓረፍተ ነገሮች ብቻ መልስ።"
CLAUSE = re.compile(r"[።፣፤?!፧]|\.(?=\s)")

# ---------------- models (loaded once; load time reported separately) ----------------
t = time.perf_counter()
if args.asr_backend == "ct2-whisper":
    asr = WhisperModel(args.asr, device="cpu", compute_type="int8", cpu_threads=4)
    PROMPT = asr.hf_tokenizer.encode("<|startoftranscript|><|am|><|transcribe|><|notimestamps|>", add_special_tokens=False).ids
else:
    from transformers import AutoModelForCTC, AutoProcessor
    ctc_proc = AutoProcessor.from_pretrained(args.asr); ctc = AutoModelForCTC.from_pretrained(args.asr).eval()
tts = VitsModel.from_pretrained("facebook/mms-tts-amh").eval(); tts_tok = AutoTokenizer.from_pretrained("facebook/mms-tts-amh")
roman = ur.Uroman(); torch.set_num_threads(2)
so = ort.SessionOptions(); so.intra_op_num_threads = 1
vad = ort.InferenceSession("models/vad/silero_vad.onnx", sess_options=so, providers=["CPUExecutionProvider"])
BIN = pathlib.Path("bin/llama-b11179")
srv = subprocess.Popen([str(BIN / "llama-server"), "-m", args.gguf, "-t", str(args.llm_threads), "-c", "2048", "--port",
                        str(args.port), "-np", "1", "--no-webui"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       env={**os.environ, "LD_LIBRARY_PATH": str(BIN)})
while True:
    try:
        if json.load(urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health")).get("status") == "ok": break
    except Exception: time.sleep(0.2)
load_s = time.perf_counter() - t

def synth(text):
    rom = roman.romanize_string(text, lcode="amh")
    if not re.search(r"[a-z]", rom):                       # nothing speakable (e.g. "1." or "-")
        return np.zeros(1600, np.float32)
    with torch.inference_mode():
        return tts(**tts_tok(rom, return_tensors="pt")).waveform[0].numpy()

def first_chunk(text, mode):
    """First speakable chunk: text up to the first clause mark that closes >= 2 words (B, C),
    or the first 5 words if no such mark has arrived yet (C only)."""
    for m in CLAUSE.finditer(text):
        if len(text[:m.end()].split()) >= 2:
            return text[:m.end()]
    if mode == "C" and len(text.split()) > 5:
        return " ".join(text.split()[:5])
    return None

# ---------------- user turns: synthetic Amharic questions + trailing silence ----------------
AUD = pathlib.Path("audio/user_turns"); AUD.mkdir(parents=True, exist_ok=True)
turns = []
for i, q in enumerate(USER_QUESTIONS):
    p = AUD / f"q{i}.wav"
    if not p.exists():
        w = synth(q); w = np.concatenate([np.zeros(int(0.3 * SR)), w, np.zeros(int(1.5 * SR))]).astype(np.float32)
        sf.write(p, w, tts.config.sampling_rate)
    turns.append((q, p))

def endpoint(wav):
    """Stream 32 ms frames through Silero; return (last_speech_s, endpoint_declared_s, cpu_s)."""
    state = np.zeros((2, 1, 128), np.float32); ctx = np.zeros(64, np.float32); last = None; cpu = 0.0
    for k, i in enumerate(range(0, len(wav) - 512 + 1, 512)):
        x = np.concatenate([ctx, wav[i:i + 512]])[None].astype(np.float32)
        t = time.perf_counter()
        out, state = vad.run(None, {"input": x, "state": state, "sr": np.array(SR, np.int64)}); cpu += time.perf_counter() - t
        ctx = wav[i + 448:i + 512]; now = (k + 1) * 0.032
        if out[0][0] > 0.5: last = now
        elif last is not None and (now - last) * 1000 >= args.hangover_ms:
            return last, now, cpu
    return last, len(wav) / SR, cpu

def transcribe(wav):
    if args.asr_backend == "ctc":
        torch.set_num_threads(4)
        with torch.inference_mode():
            logits = ctc(**ctc_proc(wav, sampling_rate=SR, return_tensors="pt")).logits
        torch.set_num_threads(2)
        return ctc_proc.batch_decode(logits.argmax(-1))[0].strip()
    f = asr.feature_extractor(wav)[:, :3000]; f = np.pad(f, ((0, 0), (0, 3000 - f.shape[1])))
    enc = asr.model.encode(ctranslate2.StorageView.from_array(np.ascontiguousarray(f[None])), to_cpu=False)
    return asr.hf_tokenizer.decode(asr.model.generate(enc, [PROMPT], max_length=880, beam_size=1)[0].sequences_ids[0]).strip()

def llm_stream(user_text, out_q):
    body = json.dumps({"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_text}],
                       "stream": True, "max_tokens": args.max_tokens, "temperature": 0.3, "repeat_penalty": 1.1,
                       "cache_prompt": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{args.port}/v1/chat/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        for line in r:
            line = line.decode().strip()
            if line.startswith("data:") and not line.endswith("[DONE]"):
                d = (json.loads(line[5:]).get("choices") or [{}])[0].get("delta", {}).get("content")
                if d: out_q.put((time.perf_counter(), d))
    out_q.put((time.perf_counter(), None))

def run_mode(mode, user_text):
    """Returns dict of stage timings (s) measured from the moment ASR text is ready."""
    q = queue.Queue(); t0 = time.perf_counter()
    th = threading.Thread(target=llm_stream, args=(user_text, q)); th.start()
    text, t_first_tok, chunk, t_chunk = "", None, None, None
    while True:
        ts, d = q.get()
        if d is None: t_llm_done = ts - t0; break
        t_first_tok = t_first_tok or ts - t0; text += d
        if mode != "A" and chunk is None:
            chunk = first_chunk(text, mode)
            if chunk:
                t_chunk = ts - t0
                ts_tts = time.perf_counter(); audio = synth(chunk); t_tts = time.perf_counter() - ts_tts
                t_first_audio = time.perf_counter() - t0      # LLM keeps streaming in its thread meanwhile
    th.join()
    if mode == "A" or chunk is None:                           # sequential, or no clause found -> whole reply
        chunk = text; t_chunk = t_llm_done
        ts_tts = time.perf_counter(); audio = synth(text); t_tts = time.perf_counter() - ts_tts
        t_first_audio = time.perf_counter() - t0
    return {"llm_ttft": t_first_tok, "llm_chunk_ready": t_chunk, "llm_total": t_llm_done, "tts_first_chunk": t_tts,
            "after_asr_to_audio": t_first_audio, "first_chunk_words": len(chunk.split()),
            "first_chunk_audio_s": round(len(audio) / tts.config.sampling_rate, 2), "reply": text.strip(), "first_chunk": chunk}

results = []
try:
    llm_stream("ሰላም", queue.Queue())                          # warm-up + caches the system prompt
    for mode in args.modes:
        for qtext, path in turns:
            wav, sr = sf.read(path, dtype="float32")
            last_speech, declared, vad_cpu = endpoint(wav)
            t_ep = declared - last_speech                         # waiting for silence hangover
            ts = time.perf_counter(); hyp = transcribe(wav[: int(declared * SR)]); t_asr = time.perf_counter() - ts
            r = run_mode(mode, hyp)
            ttfa = t_ep + t_asr + r["after_asr_to_audio"]
            row = {"mode": mode, "question": qtext, "asr_text": hyp, "endpoint_s": round(t_ep, 3), "vad_cpu_s": round(vad_cpu, 3),
                   "asr_s": round(t_asr, 3), **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()},
                   "ttfa_s": round(ttfa, 3)}
            results.append(row)
            print(f"[{mode}] TTFA={ttfa:5.2f}s | ep={t_ep:.2f} asr={t_asr:.2f} llm_ttft={r['llm_ttft']:.2f} "
                  f"chunk@{r['llm_chunk_ready']:.2f} tts={r['tts_first_chunk']:.2f} llm_total={r['llm_total']:.2f} "
                  f"| {r['first_chunk_words']}w | ASR: {hyp[:40]} | REPLY: {r['reply'][:60]}", flush=True)
finally:
    srv.send_signal(signal.SIGINT); srv.wait(timeout=30)

summary = {"label": args.label, "asr": args.asr, "gguf": args.gguf, "hangover_ms": args.hangover_ms, "load_s": round(load_s, 1)}
for mode in args.modes:
    rows = [r for r in results if r["mode"] == mode]
    summary[mode] = {k: round(statistics.median(r[k] for r in rows), 3) for k in
                     ("ttfa_s", "endpoint_s", "asr_s", "llm_ttft", "llm_chunk_ready", "tts_first_chunk", "llm_total")}
    summary[mode]["ttfa_p90_s"] = round(float(np.percentile([r["ttfa_s"] for r in rows], 90)), 3)
summary["runs"] = results
pathlib.Path(f"results/e2e_{args.label}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=1, ensure_ascii=False))
