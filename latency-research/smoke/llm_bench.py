"""E-LLM-1: Amharic LLM on laptop CPU via llama.cpp server (streaming).

For each user question: TTFT (first token), time-to-first-clause (first ። ፣ ? !
-> what a streaming TTS would need), total time, tokens, words, decode tok/s.
Runs each question twice: cold (no prompt cache) and warm (system prompt cached).
"""
import argparse, json, subprocess, time, pathlib, re, statistics, urllib.request, signal, os
from sentences_am import USER_QUESTIONS

ap = argparse.ArgumentParser()
ap.add_argument("--gguf", required=True); ap.add_argument("--label", required=True)
ap.add_argument("--threads", type=int, default=4); ap.add_argument("--max_tokens", type=int, default=120)
ap.add_argument("--port", type=int, default=8089)
ap.add_argument("--system", default="አንተ አጋዥ የድምፅ ረዳት ነህ። በአማርኛ በአጭሩ፣ በአንድ ወይም በሁለት ዓረፍተ ነገሮች ብቻ መልስ።")
args = ap.parse_args()

BIN = pathlib.Path("bin/llama-b11179")
srv = subprocess.Popen([str(BIN / "llama-server"), "-m", args.gguf, "-t", str(args.threads), "-c", "2048",
                        "--port", str(args.port), "-np", "1", "--no-webui"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env={**os.environ, "LD_LIBRARY_PATH": str(BIN)})
t_load = time.perf_counter()
while True:
    try:
        if json.load(urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health")).get("status") == "ok": break
    except Exception: pass
    time.sleep(0.2)
load_s = time.perf_counter() - t_load
CLAUSE = re.compile(r"[።፣፤?!፧]|\.(?=\s)")   # models often end Amharic sentences with ASCII "."

def ask(q, cache):
    body = json.dumps({"messages": [{"role": "system", "content": args.system}, {"role": "user", "content": q}],
                       "stream": True, "max_tokens": args.max_tokens, "temperature": 0.3,
                       "repeat_penalty": 1.1, "cache_prompt": cache, "timings_per_token": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{args.port}/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = tclause = None; text = ""; ntok = 0; timings = {}
    with urllib.request.urlopen(req) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"): continue
            d = json.loads(line[5:])
            timings = d.get("timings", timings)
            delta = (d.get("choices") or [{}])[0].get("delta", {}).get("content") or ""
            if delta:
                ntok += 1; text += delta
                ttft = ttft or time.perf_counter() - t0
                if tclause is None and CLAUSE.search(text): tclause = time.perf_counter() - t0
    total = time.perf_counter() - t0
    return {"q": q, "cache": cache, "ttft_ms": round(ttft * 1000), "first_clause_ms": tclause and round(tclause * 1000),
            "total_ms": round(total * 1000), "tokens": ntok, "words": len(text.split()),
            "prompt_tokens": timings.get("prompt_n"), "prompt_ms": timings.get("prompt_ms") and round(timings["prompt_ms"]),
            "decode_tok_s": timings.get("predicted_per_second") and round(timings["predicted_per_second"], 1),
            "text": text.strip()}

runs = []
try:
    ask("ሰላም", False)                                   # warm-up
    for q in USER_QUESTIONS:
        for cache in (False, True):
            if cache:
                ask("ሰላም", True)          # prime ONLY the shared system prompt; the question itself stays uncached
            r = ask(q, cache); runs.append(r)
            print(f"cache={cache!s:<5} ttft={r['ttft_ms']:>5}ms clause={r['first_clause_ms']}ms total={r['total_ms']:>6}ms "
                  f"tok={r['tokens']:>3} words={r['words']:>2} prompt={r['prompt_tokens']}tok/{r['prompt_ms']}ms "
                  f"dec={r['decode_tok_s']}t/s | {r['text'][:80]}", flush=True)
finally:
    srv.send_signal(signal.SIGINT); srv.wait(timeout=30)

def med(key, cache):
    v = [r[key] for r in runs if r["cache"] == cache and r[key] is not None]
    return round(statistics.median(v)) if v else None
summary = {"label": args.label, "gguf": args.gguf, "threads": args.threads, "load_s": round(load_s, 1),
           **{f"{k}_{'warm' if c else 'cold'}": med(k, c) for k in ("ttft_ms", "first_clause_ms", "total_ms", "prompt_ms") for c in (False, True)},
           "decode_tok_s": statistics.median(r["decode_tok_s"] for r in runs),
           "tokens_per_word": round(sum(r["tokens"] for r in runs) / max(1, sum(r["words"] for r in runs)), 2),
           "runs": runs}
pathlib.Path(f"results/llm_{args.label}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=1, ensure_ascii=False))
