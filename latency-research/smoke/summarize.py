"""Collect every smoke-test JSON into results/summary.json (the report's tables are copied from it)."""
import json, glob, pathlib, statistics as st
R = pathlib.Path("results"); out = {}
out["asr"] = [{k: d[k] for k in ("label", "n_utts", "wer", "cer", "latency_s_median", "latency_s_p90", "rtf_median", "decoder_tokens_per_s_audio")}
              for d in (json.load(open(f)) for f in sorted(glob.glob(str(R / "asr_*.json"))))]
e2e = {}
for f in sorted(glob.glob(str(R / "e2e_*.json"))):
    d = json.load(open(f))
    for mode in "ABC":
        rows = [r for r in d["runs"] if r["mode"] == mode]
        if not rows: continue
        med = lambda k: round(st.median(r[k] for r in rows), 2)
        e2e[f"{d['label']}:{mode}"] = {"n": len(rows), "ttfa_median": med("ttfa_s"), "ttfa_p90": d[mode]["ttfa_p90_s"],
            "ttfa_min": round(min(r["ttfa_s"] for r in rows), 2), "ttfa_max": round(max(r["ttfa_s"] for r in rows), 2),
            "endpoint": med("endpoint_s"), "asr": med("asr_s"), "llm_to_chunk": med("llm_chunk_ready"),
            "tts_chunk": med("tts_first_chunk"), "first_chunk_words": med("first_chunk_words"), "llm_total": med("llm_total")}
out["e2e"] = e2e
for name in ("endpointing", "tts_mms", "tokenizer_fertility", "llm_llama32-1b-am-q4km_t2", "whisper_encoder_decoder_split", "resources", "tts_roundtrip_mms"):
    d = json.load(open(R / f"{name}.json"))
    if name == "tts_mms":
        d = {"rtf_median_t2": st.median(r["rtf"] for r in d["runs"] if r["threads"] == 2),
             "rtf_median_t4": st.median(r["rtf"] for r in d["runs"] if r["threads"] == 4),
             "per_sentence_t2": [(r["key"], r["words"], r["audio_s"], r["synth_ms_median"], r["rtf"]) for r in d["runs"] if r["threads"] == 2]}
    if name.startswith("llm_"): d = {k: v for k, v in d.items() if k != "runs"}
    if name == "tts_roundtrip_mms": d = [(r["key"], r["cer"]) for r in d]
    out[name] = d
(R / "summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps({"asr": out["asr"], "e2e": out["e2e"], "tts": out["tts_mms"]}, indent=1, ensure_ascii=False))
