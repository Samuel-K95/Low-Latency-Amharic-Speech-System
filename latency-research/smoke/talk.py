"""Live Amharic speech-to-speech demo on the laptop CPU — talk to it.

  mic (pw-record, 16 kHz) -> Silero VAD (32 ms frames)
      -> ASR on pause-delimited segments *while you speak* (MMS-300M-WAXAL CTC)
      -> Amharic Llama-3.2-1B (llama.cpp, streaming, cached prompt)
      -> number/abbreviation normaliser -> MMS-TTS (first chunk <= 5 words, then clauses)
      -> speakers (aplay)

Half-duplex: the mic is ignored while the assistant speaks (laptop speakers would
otherwise be heard as the user). Each turn prints a latency breakdown and is logged
to results/live_turns.jsonl.

  ../.venv/bin/python talk.py                 # talk (Ctrl+C to quit)
  ../.venv/bin/python talk.py --text          # type Amharic instead of speaking
  ../.venv/bin/python talk.py --wav q.wav --out-wav reply.wav   # offline self-test
  ../.venv/bin/python talk.py --team          # the final pipeline's models: hohe ASR + Tiny Aya + OmniVoice
"""
import argparse, glob, json, os, pathlib, queue, re, signal, subprocess, sys, threading, time, urllib.request, wave
from collections import deque

import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--team", action="store_true",
                help="the final pipeline's models: hohe-asr-amharic + Tiny Aya Earth + OmniVoice (an explicit --asr/--llm/--tts still wins)")
ap.add_argument("--asr", choices=["ctc", "hohe", "whisper-small", "whisper-tiny"],
                help="ctc = MMS-300M-WAXAL (default, fastest); hohe = hohe-asr-amharic, the final pipeline's ASR (606M, slower on this CPU)")
ap.add_argument("--llm", choices=["llama1b", "gemma", "tinyaya"],
                help="llama1b = Llama-3.2-1B-Amharic (default, fast, weak answers); gemma = Gemma 4 E2B (better answers, ~2.3 s slower "
                     "on this CPU); tinyaya = Tiny Aya Earth 3.35B")
ap.add_argument("--llm-gguf", help="run any other GGUF file instead (e.g. a fine-tuned Gemma once merged and converted)")
ap.add_argument("--tts", choices=["mms", "omnivoice"],
                help="mms = MMS-TTS (default, fast, the final pipeline's TTS); omnivoice = OmniVoice Amharic (natural voice, slow on CPU)")
ap.add_argument("--omni-steps", type=int, default=8, help="OmniVoice diffusion steps (recommended: 8; 16/32 sound clearer but are slower)")
ap.add_argument("--mms-model", default="facebook/mms-tts-amh",
                help="MMS-TTS checkpoint: a Hugging Face repo or a local folder")
ap.add_argument("--no-lm", action="store_true", help="CTC/hohe: plain greedy decoding instead of the Amharic 5-gram LM")
ap.add_argument("--hangover-ms", type=int, default=700, help="silence that ends your turn")
ap.add_argument("--seg-pause-ms", type=int, default=250, help="pause that sends the speech so far to ASR early")
ap.add_argument("--max-tokens", type=int, default=80)
ap.add_argument("--llm-threads", type=int, default=2)
ap.add_argument("--first-chunk-words", type=int, default=5)
ap.add_argument("--max-sentences", type=int, default=2, help="stop the reply after this many sentences")
ap.add_argument("--no-pause", action="store_true", help="keep the LLM generating while the first chunk is synthesised (they then share the 2 cores)")
ap.add_argument("--llm-nice", type=int, default=0, help="OS priority of the LLM; had no effect on this 2-core/4-thread CPU")
ap.add_argument("--text", action="store_true", help="type questions instead of speaking")
ap.add_argument("--wav", help="use this 16 kHz wav as the microphone (self-test)")
ap.add_argument("--out-wav", help="write the spoken reply here instead of playing it")
ap.add_argument("--no-greeting", action="store_true")
args = ap.parse_args()
if args.team:
    args.asr, args.llm, args.tts = args.asr or "hohe", args.llm or "tinyaya", args.tts or "omnivoice"
args.asr, args.llm, args.tts = args.asr or "ctc", args.llm or "llama1b", args.tts or "mms"
for f in ("wav", "out_wav", "llm_gguf"):           # resolve paths before we chdir into smoke/
    if getattr(args, f):
        setattr(args, f, os.path.abspath(getattr(args, f)))
if os.path.isdir(args.mms_model):                   # a local folder rather than a Hugging Face repo id
    args.mms_model = os.path.abspath(args.mms_model)
if args.wav and not os.path.exists(args.wav):
    sys.exit(f"--wav file not found: {args.wav}")

HERE = pathlib.Path(__file__).resolve().parent
os.chdir(HERE)
SR, FRAME = 16000, 512
FRAME_S = FRAME / SR
CLAUSE = re.compile(r"[።፣፤?!፧]|\.(?=\s|$)")
C = {"dim": "\033[2m", "b": "\033[1m", "g": "\033[32m", "c": "\033[36m", "y": "\033[33m", "r": "\033[31m", "x": "\033[0m"}
if not sys.stdout.isatty():
    C = {k: "" for k in C}

def log(msg, end="\n"):
    print(msg, end=end, flush=True)

# ------------------------------------------------------------------ models
log(f"{C['dim']}Loading models ({'about 20 s' if args.asr == 'ctc' and args.tts == 'mms' else 'up to a few minutes'} on this laptop)...{C['x']}")
t_load = time.perf_counter()
import torch, onnxruntime as ort, uroman as ur
from transformers import VitsModel, AutoTokenizer
from am_numbers import normalize_for_tts

torch.set_num_threads(2)
so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
vad_sess = ort.InferenceSession("models/vad/silero_vad.onnx", sess_options=so, providers=["CPUExecutionProvider"])

CTC_MODELS = {"ctc": ("MMS-300M-WAXAL", "models--waxal-benchmarking--mms-300m-waxal-amh"),
              "hohe": ("hohe-asr-amharic", "models--snapwre--hohe-asr-amharic")}
LID = re.compile(r"\[(AMH|TIR|ORM|SID|WAL)\]")           # hohe starts its output with a language tag
decoder = None
if args.asr in CTC_MODELS:
    from transformers import AutoModelForCTC, AutoProcessor
    ASR_NAME, repo_dir = CTC_MODELS[args.asr]
    snaps = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/{repo_dir}/snapshots/*"))
    if not snaps:
        sys.exit(f"{ASR_NAME} not found in ~/.cache/huggingface/hub (download it once without HF_HUB_OFFLINE)")
    ctc_dir = snaps[0]
    ctc_proc = AutoProcessor.from_pretrained(ctc_dir); ctc = AutoModelForCTC.from_pretrained(ctc_dir).eval()
    if args.asr == "hohe":                 # hohe ships its own 5-gram LM; its decoder.json keeps pyctcdecode's default weights
        lm_files, lm_weights = glob.glob(f"{ctc_dir}/lm/am-5gram.bin"), {}
    else:                                  # Dataset.ET 5-gram. WAXAL test: WER 31.4 -> 26.4 for ~0.01 s per second of audio
        lm_files = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--snapwre--amharic-asr-benchmark/snapshots/*/kenlm/am-5gram-aggressive.bin"))
        lm_weights = {"alpha": 0.3, "beta": 1.5}
    lid_ids = []
    if lm_files and not args.no_lm:
        from pyctcdecode import build_ctcdecoder
        toks = ctc_proc.tokenizer.convert_ids_to_tokens(list(range(ctc.config.vocab_size)))
        labels = ["" if t == ctc_proc.tokenizer.pad_token else " " if t == "|" else (t or "⁇") for t in toks]
        lid_ids = [i for i, t in enumerate(toks) if LID.fullmatch(t or "")]   # never let the LM beam pick a language tag
        decoder = build_ctcdecoder(labels, lm_files[0], **lm_weights)
    def transcribe(wav):
        with torch.inference_mode():
            logits = ctc(**ctc_proc(wav, sampling_rate=SR, return_tensors="pt")).logits
        if decoder:
            lp = torch.log_softmax(logits[0], -1).numpy()
            lp[:, lid_ids] = -1e4
            text = decoder.decode(lp)
        else:
            text = ctc_proc.batch_decode(logits.argmax(-1))[0]
        return re.sub(r"\s+", " ", LID.sub("", text)).strip()
else:
    ASR_NAME = args.asr
    import ctranslate2
    from faster_whisper import WhisperModel
    wm = WhisperModel(f"models/{args.asr}-waxal-amh-ct2-int8", device="cpu", compute_type="int8", cpu_threads=2)
    PROMPT = wm.hf_tokenizer.encode("<|startoftranscript|><|am|><|transcribe|><|notimestamps|>", add_special_tokens=False).ids
    def transcribe(wav):
        f = wm.feature_extractor(wav)[:, :3000]; f = np.pad(f, ((0, 0), (0, 3000 - f.shape[1])))
        enc = wm.model.encode(ctranslate2.StorageView.from_array(np.ascontiguousarray(f[None])), to_cpu=False)
        return wm.hf_tokenizer.decode(wm.model.generate(enc, [PROMPT], max_length=880, beam_size=1)[0].sequences_ids[0]).strip()

if args.tts == "omnivoice":
    import torchaudio.functional as AF
    from omnivoice import OmniVoice, OmniVoiceGenerationConfig
    omni = OmniVoice.from_pretrained("Lab-et/omnivoice-amharic", dtype=torch.float32).eval()
    omni_cfg = OmniVoiceGenerationConfig(num_step=args.omni_steps, guidance_scale=2.0)
    TTS_NAME = f"OmniVoice {args.omni_steps}-step"

    def synth(text):
        text = normalize_for_tts(re.sub(r"[*#_`]+", " ", text)).strip()
        if not re.search(r"\w", text):
            return np.zeros(0, np.float32)
        torch.manual_seed(42)                             # same voice for every chunk (auto-voice mode picks a new one otherwise)
        with torch.inference_mode():
            w = omni.generate(text=text, language="Amharic", generation_config=omni_cfg)[0]
        w = torch.as_tensor(w).float().reshape(-1)
        return AF.resample(w, omni.sampling_rate or 24000, SR).numpy()   # the player runs at 16 kHz
else:
    tts = VitsModel.from_pretrained(args.mms_model).eval()
    tts_tok = AutoTokenizer.from_pretrained(args.mms_model)
    roman = ur.Uroman()
    TTS_NAME = "MMS-TTS" if args.mms_model == "facebook/mms-tts-amh" else f"MMS-TTS ({os.path.basename(args.mms_model.rstrip('/'))})"

    def synth(text):
        text = re.sub(r"[*#_`]+", " ", text)              # Gemma sometimes emits markdown emphasis
        rom = roman.romanize_string(normalize_for_tts(text), lcode="amh")
        if not re.search(r"[a-z]", rom):
            return np.zeros(0, np.float32)
        with torch.inference_mode():
            return tts(**tts_tok(rom, return_tensors="pt")).waveform[0].numpy()

# llama.cpp server with the Amharic Llama
BIN = HERE / "bin/llama-b11179"
LLMS = {"llama1b": ("Llama-3.2-1B-Amharic Q4_K_M", "models--mradermacher--Llama-3.2-1B-Amharic-Instruct-GGUF/snapshots/*/*.gguf", []),
        # Gemma 4's template turns on hidden "thinking" unless told not to: it then reasons in English first
        "gemma": ("Gemma 4 E2B Q4_K_M", "models--unsloth--gemma-4-E2B-it-GGUF/snapshots/*/gemma-4-E2B-it-Q4_K_M.gguf",
                  # --swa-full: Gemma's sliding-window attention otherwise disables prompt-cache reuse (~10 s re-prefill per turn)
                  ["--chat-template-kwargs", '{"enable_thinking": false}', "--swa-full"]),
        # Cohere's official 4-bit build of Tiny Aya (the original repo is gated, this GGUF repo is not).
        # It also uses sliding-window attention: without --swa-full every turn re-reads the ~510-token prompt (28 s here, 1.2 s with it)
        "tinyaya": ("Tiny Aya Earth Q4_K_M", "models--CohereLabs--tiny-aya-earth-GGUF/snapshots/*/tiny-aya-earth-q4_k_m.gguf", ["--swa-full"])}
if args.llm_gguf:
    LLM_NAME, GGUF, llm_flags = pathlib.Path(args.llm_gguf).name, args.llm_gguf, []
    if not os.path.exists(GGUF):
        sys.exit(f"--llm-gguf file not found: {GGUF}")
else:
    LLM_NAME, pattern, llm_flags = LLMS[args.llm]
    found = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/" + pattern))
    if not found:
        sys.exit(f"LLM file for --llm {args.llm} not found in ~/.cache/huggingface/hub")
    GGUF = found[0]
PORT = 8097
srv = subprocess.Popen([str(BIN / "llama-server"), "-m", GGUF, "-t", str(args.llm_threads), "-c", "2048", "--port", str(PORT),
                        "-np", "1", "--no-webui", *llm_flags], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       env={**os.environ, "LD_LIBRARY_PATH": str(BIN)}, preexec_fn=lambda: os.nice(args.llm_nice))
while True:
    try:
        if json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health")).get("status") == "ok":
            break
    except Exception:
        if srv.poll() is not None:
            sys.exit("llama-server failed to start")
        time.sleep(0.2)

SYSTEM = ("አንተ በአማርኛ የምትናገር አጋዥ የድምፅ ረዳት ነህ። ለጥያቄው በቀጥታና በአጭሩ፣ በአንድ ወይም በሁለት ዓረፍተ ነገሮች ብቻ መልስ። "
          "ዝርዝር ወይም ቁጥር ያላቸው ነጥቦች አትጠቀም።")
EXAMPLES = [("ሰላም፣ እንዴት ነህ?", "ሰላም! ደህና ነኝ፣ አመሰግናለሁ። ምን ልርዳዎት?"),
            ("የኢትዮጵያ ዋና ከተማ ማን ናት?", "የኢትዮጵያ ዋና ከተማ አዲስ አበባ ናት።")]
history = []                                   # last exchanges, kept short for fast prefill

def messages(user_text):
    m = [{"role": "system", "content": SYSTEM}]
    for q, a in EXAMPLES + history[-2:]:
        m += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    return m + [{"role": "user", "content": user_text}]

def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req)

def llm_stream(prompt, n_predict, out_q, stop):
    """Stream raw completions for an already-templated prompt. Closing the connection
    (stop) makes llama-server stop; a later call with prompt + partial reply resumes from
    the KV cache, so only the few new tokens need prefill (~70 ms)."""
    body = {"prompt": prompt, "stream": True, "n_predict": n_predict, "temperature": 0.3,
            "repeat_penalty": 1.15, "repeat_last_n": 64, "cache_prompt": True}
    try:
        with post("/completion", body) as r:
            for line in r:
                if stop.is_set():
                    break
                line = line.decode().strip()
                if line.startswith("data:"):
                    d = json.loads(line[5:])
                    if d.get("content"):
                        out_q.put(d["content"])
                    if d.get("stop"):
                        break
    finally:
        out_q.put(None)

def chat_prompt(user_text):
    return json.load(post("/apply-template", {"messages": messages(user_text)}))["prompt"]

# warm-up: first calls are slower (lazy init); prime the cached system prompt
transcribe(np.zeros(SR, np.float32)); synth("ሰላም")
llm_stream(chat_prompt("ሰላም"), 8, queue.Queue(), threading.Event())
log(f"{C['dim']}Ready in {time.perf_counter() - t_load:.0f} s  (ASR={ASR_NAME}{'' if decoder is None else ' + 5-gram LM'}, LLM={LLM_NAME}, TTS={TTS_NAME}){C['x']}")

# ------------------------------------------------------------------ audio out
class Player:
    """One aplay process per reply, started early so it is ready when the first chunk arrives."""
    def __init__(self):
        self.proc, self.frames, self.first_write = None, [], None
    def open(self):
        if args.out_wav:
            return
        self.proc = subprocess.Popen(["aplay", "-q", "-t", "raw", "-f", "S16_LE", "-r", str(SR), "-c", "1", "--buffer-time=150000", "-"],
                                     stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    def write(self, audio):
        if len(audio) == 0:
            return
        self.first_write = self.first_write or time.perf_counter()
        pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
        if args.out_wav:
            self.frames.append(pcm)
        else:
            self.proc.stdin.write(pcm); self.proc.stdin.flush()
    def close(self):
        if args.out_wav:
            with wave.open(args.out_wav, "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(b"".join(self.frames))
        elif self.proc:
            self.proc.stdin.close(); self.proc.wait()          # returns when playback has drained

def speak(text):
    p = Player(); p.open(); p.write(synth(text)); p.close()

# ------------------------------------------------------------------ one reply
def first_chunk_end(text):
    for m in CLAUSE.finditer(text):
        if len(text[:m.end()].split()) >= 2:
            return m.end()
    words = list(re.finditer(r"\S+", text))
    if len(words) > args.first_chunk_words:            # cap: 5 words even without punctuation
        return words[args.first_chunk_words - 1].end()
    return None

def next_chunk_end(text):
    for m in CLAUSE.finditer(text):
        if len(text[:m.end()].split()) >= 2:
            return m.end()
    return None

def degenerate(text):
    w = text.split()
    return len(w) >= 3 and w[-1] == w[-2] == w[-3]

def respond(user_text, t_user_end, t_endpoint, t_asr_done, extra):
    player = Player(); player.open()
    t_req = time.perf_counter(); prompt = chat_prompt(user_text)
    reply, pending, t_first_tok, t_chunk1, tts1, n_tok = "", "", None, None, None, 0
    log(f"{C['g']}{C['b']}assistant:{C['x']} ", end="")
    finished = False
    while not finished:
        # phase 1: until the first chunk; phase 2 (resumed from the KV cache): the rest
        toks, stop = queue.Queue(), threading.Event()
        th = threading.Thread(target=llm_stream, args=(prompt + reply, args.max_tokens - n_tok, toks, stop), daemon=True); th.start()
        while True:
            tok = toks.get()
            stream_end = tok is None
            if not stream_end:
                t_first_tok = t_first_tok or time.perf_counter()
                reply += tok; pending += tok; n_tok += 1; log(tok, end="")
                if len(re.findall(r"[።?!፧]|\.(?=\s|$)", reply)) >= args.max_sentences or degenerate(reply) \
                        or n_tok >= args.max_tokens:
                    stop.set(); stream_end = True; finished = True     # spoken replies: short and complete
            else:
                finished = True
            cut = (first_chunk_end if t_chunk1 is None else next_chunk_end)(pending)
            if stream_end and pending.strip():
                cut = len(pending)
            if cut:
                chunk, pending = pending[:cut], pending[cut:]
                first = t_chunk1 is None
                if first:
                    t_chunk1 = time.perf_counter()
                    if not args.no_pause and not finished:
                        stop.set(); th.join()                          # free both cores for the first chunk
                ts = time.perf_counter(); audio = synth(chunk); dt = time.perf_counter() - ts
                tts1 = tts1 if tts1 is not None else dt
                player.write(audio)
                if first and not args.no_pause and not finished:
                    break                                              # resume generation while chunk 1 plays
            if stream_end:
                break
        th.join()
    log("")
    t_first_audio = player.first_write
    player.close()
    history.append((user_text, reply.strip()))
    if t_first_audio is None:
        log(f"{C['r']}(no speakable reply){C['x']}"); return
    row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "asr": args.asr, "llm": LLM_NAME, "tts": TTS_NAME, "user": user_text, "reply": reply.strip(),
           "endpoint_wait_s": round(t_endpoint - t_user_end, 3) if t_endpoint else None,
           "asr_after_endpoint_s": round(t_asr_done - t_endpoint, 3) if t_endpoint else None,
           "llm_ttft_s": round(t_first_tok - t_req, 3) if t_first_tok else None,
           "llm_first_chunk_s": round(t_chunk1 - t_req, 3) if t_chunk1 else None,
           "tts_first_chunk_s": round(tts1, 3), "ttfa_s": round(t_first_audio - t_user_end, 3), **extra}
    with open("results/live_turns.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    parts = []
    if t_endpoint:
        parts += [f"endpoint {row['endpoint_wait_s']:.2f}", f"ASR {row['asr_after_endpoint_s']:.2f}"]
    parts += [f"LLM→chunk {row['llm_first_chunk_s']:.2f}", f"TTS {row['tts_first_chunk_s']:.2f}"]
    log(f"{C['y']}⏱  time to first audio: {C['b']}{row['ttfa_s']:.2f} s{C['x']}{C['y']}   ({' + '.join(parts)} s){C['x']}")

# ------------------------------------------------------------------ typed mode
def shutdown(*_):
    try:
        srv.send_signal(signal.SIGINT); srv.wait(timeout=10)
    except Exception:
        srv.kill()
    log(f"\n{C['dim']}bye{C['x']}"); os._exit(0)
signal.signal(signal.SIGINT, shutdown); signal.signal(signal.SIGTERM, shutdown)

if not args.no_greeting and not args.wav:
    log(f"{C['g']}{C['b']}assistant:{C['x']} ሰላም! በአማርኛ ይጠይቁኝ።"); speak("ሰላም! በአማርኛ ይጠይቁኝ።")

if args.text:
    log(f"{C['dim']}Type an Amharic question and press Enter (Ctrl+C to quit).{C['x']}")
    while True:
        try:
            q = input(f"{C['c']}{C['b']}you>{C['x']} ").strip()
        except EOFError:
            shutdown()
        if q:
            t = time.perf_counter(); respond(q, t, None, None, {"mode": "text"})

# ------------------------------------------------------------------ microphone
frames = queue.Queue()

def mic_reader():
    try:
        _mic_reader()
    except Exception as e:                                # never leave the main loop waiting forever
        log(f"{C['r']}audio input failed: {e}{C['x']}")
        frames.put((None, time.perf_counter()))

def _mic_reader():
    if args.wav:                                          # replay a file at real-time pace as if it were the mic
        with wave.open(args.wav) as w:
            assert w.getframerate() == SR and w.getnchannels() == 1
            data = np.frombuffer(w.readframes(w.getnframes()), "<i2")
        data = np.concatenate([data, np.zeros(int(1.5 * SR), "<i2")])
        t0 = time.perf_counter()
        for i in range(0, len(data) - FRAME + 1, FRAME):
            time.sleep(max(0, t0 + (i + FRAME) / SR - time.perf_counter()))
            frames.put((data[i:i + FRAME].astype(np.float32) / 32768, time.perf_counter()))
        frames.put((None, time.perf_counter())); return
    rec = subprocess.Popen(["pw-record", "--rate", str(SR), "--channels", "1", "--format", "s16", "--latency", "20ms", "-"],
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    while True:
        b = rec.stdout.read(FRAME * 2)
        if len(b) < FRAME * 2:
            log(f"{C['r']}microphone stream ended (is a mic connected? try: wpctl status){C['x']}")
            frames.put((None, time.perf_counter()))
            break
        frames.put((np.frombuffer(b, "<i2").astype(np.float32) / 32768, time.perf_counter()))

threading.Thread(target=mic_reader, daemon=True).start()

class Vad:
    def __init__(self): self.reset()
    def reset(self): self.state = np.zeros((2, 1, 128), np.float32); self.ctx = np.zeros(64, np.float32)
    def __call__(self, f):
        out, self.state = vad_sess.run(None, {"input": np.concatenate([self.ctx, f])[None].astype(np.float32),
                                              "state": self.state, "sr": np.array(SR, np.int64)})
        self.ctx = f[-64:]; return float(out[0][0])

# ASR worker: segments are transcribed while the user is still talking
asr_jobs, asr_results = queue.Queue(), {}
def asr_worker():
    while True:
        k, audio = asr_jobs.get()
        t = time.perf_counter(); txt = transcribe(audio); asr_results[k] = (txt, time.perf_counter() - t)
        asr_jobs.task_done()
threading.Thread(target=asr_worker, daemon=True).start()

vad = Vad()
hang_n = int(args.hangover_ms / 1000 / FRAME_S + 0.5)
seg_n = int(args.seg_pause_ms / 1000 / FRAME_S + 0.5)
log(f"{C['dim']}Speak Amharic after the prompt. Pause {args.hangover_ms} ms to finish your turn. Ctrl+C to quit.{C['x']}")

while True:
    log(f"{C['c']}🎤 listening…{C['x']}")
    vad.reset(); pre = deque(maxlen=10); run = 0
    audio, in_speech, silence, seg_start, seg_has_speech, n_seg, t_last_speech, speech_frames = [], False, 0, 0, False, 0, None, 0
    asr_results.clear()
    while True:
        f, t = frames.get()
        if f is None:                                     # --wav input finished
            shutdown()
        p = vad(f)
        if not in_speech:
            pre.append(f); run = run + 1 if p > 0.5 else 0
            if run >= 3:                                  # ~100 ms of speech starts a turn
                in_speech, audio, t_last_speech, speech_frames = True, list(pre), t, run
                log(f"{C['dim']}   (hearing you){C['x']}")
            continue
        audio.append(f)
        if p > 0.5:
            silence, t_last_speech, seg_has_speech = 0, t, True; speech_frames += 1
        else:
            silence += 1
        if seg_has_speech and silence == seg_n:          # short pause: send what we have to ASR now
            asr_jobs.put((n_seg, np.concatenate(audio[seg_start:]))); n_seg += 1
            seg_start, seg_has_speech = len(audio), False
        if silence >= hang_n or len(audio) * FRAME_S > 25:
            break
    t_endpoint = time.perf_counter()
    if seg_has_speech:
        asr_jobs.put((n_seg, np.concatenate(audio[seg_start:]))); n_seg += 1
    if speech_frames * FRAME_S < 0.3:
        continue                                          # a click or a cough
    asr_jobs.join(); t_asr_done = time.perf_counter()
    text = " ".join(asr_results[k][0] for k in range(n_seg) if asr_results[k][0]).strip()
    log(f"{C['c']}{C['b']}you:{C['x']} {text or '…'}")
    if not text:
        log(f"{C['r']}(didn't catch that — try again){C['x']}"); continue
    respond(text, t_last_speech, t_endpoint, t_asr_done,
            {"mode": "wav" if args.wav else "mic", "speech_s": round(len(audio) * FRAME_S, 2), "asr_segments": n_seg,
             "asr_compute_s": round(sum(v[1] for v in asr_results.values()), 3)})
    if args.wav:
        shutdown()
    time.sleep(0.3)                                       # let the room go quiet
    while not frames.empty():                             # drop everything the mic heard while we spoke
        frames.get_nowait()
