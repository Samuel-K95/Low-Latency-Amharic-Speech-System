"""Download Ethio-ASR-amharic (Dataset.ET benchmark winner) and the pruned Amharic 5-gram LM."""
from huggingface_hub import snapshot_download, hf_hub_download
print("OK", snapshot_download("badrex/Ethio-ASR-amharic", allow_patterns=["*.json", "model.safetensors"]), flush=True)
print("OK", hf_hub_download("snapwre/amharic-asr-benchmark", "kenlm/am-5gram-aggressive.bin", repo_type="dataset"), flush=True)
