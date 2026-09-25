"""E-TOK: how many tokens does each candidate model spend per Amharic word?

Autoregressive decode time scales with the number of generated tokens, so a
tokenizer that splits Ge'ez script into bytes makes every Amharic reply slower
(LLM) and every Whisper transcript slower (ASR decoder). Only tokenizer.json
files are downloaded (a few MB each).
"""
import json, os, random, pathlib, statistics
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

# Any plain-text Amharic corpus, one sentence per line (the report used the normalized corpus from the Week-4 tokenization project).
CORPUS = pathlib.Path(os.environ.get("AMHARIC_CORPUS", "data/amharic_corpus.txt"))
if not CORPUS.exists():
    raise SystemExit(f"Amharic corpus not found: {CORPUS}. Set AMHARIC_CORPUS=/path/to/corpus.txt")
random.seed(13)
lines = [l.strip() for l in CORPUS.read_text(encoding="utf-8").splitlines() if len(l.split()) >= 4]
sample = random.sample(lines, 3000)
waxal = [json.loads(l)["text"] for l in open("audio/waxal_amh_test/manifest.jsonl", encoding="utf-8")]
english = ["The weather in Addis Ababa is pleasant today, with light rain expected in the evening.",
           "Please tell me the opening hours of the national museum on Saturday."] * 50

MODELS = {  # label: (repo, role)
    "Whisper (multilingual BPE)": ("waxal-benchmarking/whisper-small-waxal-amh", "ASR decoder"),
    "Qwen3 / Qwen3-ASR": ("Qwen/Qwen3-0.6B", "LLM + ASR decoder"),
    "Qwen3.5": ("Qwen/Qwen3.5-0.8B", "LLM"),
    "Llama 3.2 (base)": ("unsloth/Llama-3.2-1B-Instruct", "LLM"),
    "Llama-3.2-1B-Amharic (rasyosef)": ("rasyosef/Llama-3.2-1B-Amharic-Instruct", "LLM"),
    "Gemma 3": ("unsloth/gemma-3-1b-it", "LLM"),
    "Gemma 4": ("unsloth/gemma-4-E2B-it", "LLM"),
    "Phi-4-mini": ("microsoft/Phi-4-mini-instruct", "LLM"),
}

def stats(tok, texts):
    toks = [len(tok.encode(t, add_special_tokens=False).ids) for t in texts]
    words = [len(t.split()) for t in texts]
    chars = [len(t.replace(" ", "")) for t in texts]
    return sum(toks) / sum(words), sum(chars) / sum(toks)

rows = []
for label, (repo, role) in MODELS.items():
    tok = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
    tpw, cpt = stats(tok, sample)
    tpw_w, _ = stats(tok, waxal)
    tpw_en, _ = stats(tok, english)
    rows.append({"tokenizer": label, "repo": repo, "role": role, "vocab": tok.get_vocab_size(),
                 "tok_per_word_news": round(tpw, 2), "tok_per_word_speech": round(tpw_w, 2),
                 "chars_per_tok": round(cpt, 2), "tok_per_word_english": round(tpw_en, 2),
                 "amh_to_eng_ratio": round(tpw / tpw_en, 2),
                 "tokens_for_30_word_reply": round(30 * tpw)})
    print(f"{label:<34} vocab={tok.get_vocab_size():>7}  amh tok/word={tpw:5.2f} (speech {tpw_w:5.2f})  "
          f"eng tok/word={tpw_en:4.2f}  ratio={tpw/tpw_en:4.1f}x  30-word reply={30*tpw:4.0f} tok", flush=True)

pathlib.Path("results/tokenizer_fertility.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
print("mean words/line in sample:", round(statistics.mean(len(s.split()) for s in sample), 1))
