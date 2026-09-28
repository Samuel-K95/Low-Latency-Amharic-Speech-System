"""E-LLM-2 (CPU part): answer quality vs speed for candidate Amharic LLMs on the laptop.

Same system prompt and few-shot examples as the live demo (talk.py). Each question is asked once,
warm (system prompt cached). Records TTFT, time until 5 words exist (what the first TTS chunk needs),
decode speed, and the reply (stopped after 2 sentences, like the demo). Replies are rated by hand.
"""
import argparse, json, os, pathlib, re, signal, statistics, subprocess, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--gguf", required=True); ap.add_argument("--label", required=True)
ap.add_argument("--threads", type=int, default=2); ap.add_argument("--port", type=int, default=8101)
ap.add_argument("--max-tokens", type=int, default=120)
args = ap.parse_args()

SYSTEM = ("አንተ በአማርኛ የምትናገር አጋዥ የድምፅ ረዳት ነህ። ለጥያቄው በቀጥታና በአጭሩ፣ በአንድ ወይም በሁለት ዓረፍተ ነገሮች ብቻ መልስ። "
          "ዝርዝር ወይም ቁጥር ያላቸው ነጥቦች አትጠቀም።")
EXAMPLES = [("ሰላም፣ እንዴት ነህ?", "ሰላም! ደህና ነኝ፣ አመሰግናለሁ። ምን ልርዳዎት?"),
            ("የኢትዮጵያ ዋና ከተማ ማን ናት?", "የኢትዮጵያ ዋና ከተማ አዲስ አበባ ናት።")]
QUESTIONS = [                                       # (question, what a correct answer must contain)
    ("ቡና እንዴት ይፈላል?", "roast/grind beans, boil with water (jebena)"),
    ("ለጉንፋን ምን ማድረግ ይሻላል?", "rest, fluids, see a doctor if severe"),
    ("ስለ ላሊበላ አብያተ ክርስቲያናት በአጭሩ ንገረኝ።", "rock-hewn churches, King Lalibela, ~12th-13th c., UNESCO"),
    ("ዓባይ ወንዝ የሚነሳው ከየት ነው?", "Lake Tana / Gish Abay"),
    ("የኢትዮጵያ ትልቁ ሐይቅ የትኛው ነው?", "Lake Tana"),
    ("ውሃ በስንት ዲግሪ ሴልሺየስ ይፈላል?", "100 degrees"),
    ("እንጀራ ከምን ይሰራል?", "teff flour, fermented batter"),
    ("ከአዲስ አበባ ወደ ባህር ዳር እንዴት መሄድ እችላለሁ?", "bus or domestic flight"),
    ("ለፈተና በደንብ ለማጥናት ምን ላድርግ?", "plan, regular study, rest"),
    ("የአፍሪካ ህብረት ዋና መሥሪያ ቤት የት ነው?", "Addis Ababa"),
]
SENT_END = re.compile(r"[።?!፧]|\.(?=\s|$)")

BIN = pathlib.Path("bin/llama-b11179")
srv = subprocess.Popen([str(BIN / "llama-server"), "-m", args.gguf, "-t", str(args.threads), "-c", "2048", "--port", str(args.port),
                        "-np", "1", "--no-webui", "--chat-template-kwargs", "{\"enable_thinking\": false}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       env={**os.environ, "LD_LIBRARY_PATH": str(BIN)})
t0 = time.perf_counter()
while True:
    try:
        if json.load(urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health")).get("status") == "ok": break
    except Exception:
        if srv.poll() is not None: raise SystemExit("llama-server failed to start")
        time.sleep(0.3)
load_s = time.perf_counter() - t0

def ask(q):
    msgs = [{"role": "system", "content": SYSTEM}]
    for uq, ua in EXAMPLES:
        msgs += [{"role": "user", "content": uq}, {"role": "assistant", "content": ua}]
    msgs.append({"role": "user", "content": q})
    body = json.dumps({"messages": msgs, "stream": True, "max_tokens": args.max_tokens, "temperature": 0.3,
                       "repeat_penalty": 1.15, "cache_prompt": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{args.port}/v1/chat/completions", body, {"Content-Type": "application/json"})
    t = time.perf_counter(); ttft = t5 = None; text = ""; n = 0
    with urllib.request.urlopen(req) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"): continue
            d = (json.loads(line[5:]).get("choices") or [{}])[0].get("delta", {}).get("content")
            if not d: continue
            n += 1; text += d; ttft = ttft or time.perf_counter() - t
            if t5 is None and len(text.split()) > 5: t5 = time.perf_counter() - t
            if len(SENT_END.findall(text)) >= 2: break
    total = time.perf_counter() - t
    return {"q": q, "ttft_s": round(ttft, 2), "five_words_s": t5 and round(t5, 2), "total_s": round(total, 2), "tokens": n,
            "tok_s": round(n / max(1e-6, total - ttft), 1), "words": len(text.split()), "reply": text.strip()}

rows = []
try:
    ask("ሰላም")                                          # warm-up: caches the system prompt + examples
    for q, key in QUESTIONS:
        r = ask(q); r["expected"] = key; rows.append(r)
        print(f"ttft={r['ttft_s']:4.2f}s 5w={r['five_words_s']}s {r['tok_s']:4.1f}tok/s | {q}\n    -> {r['reply']}", flush=True)
finally:
    srv.send_signal(signal.SIGINT); srv.wait(timeout=30)

med = lambda k: round(statistics.median(r[k] for r in rows if r[k] is not None), 2)
out = {"label": args.label, "gguf": args.gguf, "load_s": round(load_s, 1), "ttft_s_median": med("ttft_s"),
       "five_words_s_median": med("five_words_s"), "tok_s_median": med("tok_s"),
       "tokens_per_word": round(sum(r["tokens"] for r in rows) / max(1, sum(r["words"] for r in rows)), 2), "rows": rows}
pathlib.Path(f"results/llm_compare_{args.label}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
