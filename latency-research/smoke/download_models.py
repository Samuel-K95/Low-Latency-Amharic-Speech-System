"""Download only the inference files needed for the CPU smoke tests (disk is tight)."""
from huggingface_hub import snapshot_download, hf_hub_download
import os

jobs = [
    ("facebook/mms-tts-amh", ["config.json", "model.safetensors", "tokenizer_config.json", "vocab.json", "special_tokens_map.json"]),
    ("waxal-benchmarking/whisper-tiny-waxal-amh", ["*.json", "model.safetensors"]),
    ("waxal-benchmarking/whisper-small-waxal-amh", ["*.json", "model.safetensors"]),
    ("pipecat-ai/smart-turn-v3", ["smart-turn-v3.2-cpu.onnx", "README.md"]),
]
for repo, pats in jobs:
    p = snapshot_download(repo, allow_patterns=pats)
    print("OK", repo, p, flush=True)

p = hf_hub_download("mradermacher/Llama-3.2-1B-Amharic-Instruct-GGUF", "Llama-3.2-1B-Amharic-Instruct.Q4_K_M.gguf")
print("OK gguf", p, flush=True)
