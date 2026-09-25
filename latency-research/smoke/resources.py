"""Resident memory of each component when loaded (laptop CPU)."""
import os, json, time, subprocess, psutil, pathlib, glob, urllib.request
proc = psutil.Process()
def rss(): return proc.memory_info().rss / 1e6
out = {"python_baseline_MB": round(rss())}
import torch, numpy as np, ctranslate2; from transformers import VitsModel, AutoModelForCTC, AutoTokenizer; from faster_whisper import WhisperModel
x = np.random.randn(16000 * 5).astype(np.float32) * 0.01           # 5 s dummy turn: RSS measured AFTER one inference
def run_whisper(m):
    f = m.feature_extractor(x)[:, :3000]; f = np.pad(f, ((0, 0), (0, 3000 - f.shape[1])))
    m.model.generate(m.model.encode(ctranslate2.StorageView.from_array(np.ascontiguousarray(f[None]))), [[50258, 50334, 50359, 50363]], max_length=40)
b = rss(); w = WhisperModel("models/whisper-small-waxal-amh-ct2-int8", device="cpu", compute_type="int8"); run_whisper(w); out["whisper_small_ct2_int8_MB"] = round(rss() - b)
b = rss(); t = WhisperModel("models/whisper-tiny-waxal-amh-ct2-int8", device="cpu", compute_type="int8"); run_whisper(t); out["whisper_tiny_ct2_int8_MB"] = round(rss() - b)
b = rss(); c = AutoModelForCTC.from_pretrained(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--waxal-benchmarking--mms-300m-waxal-amh/snapshots/*"))[0]).eval(); c(torch.from_numpy(x)[None]); out["mms300m_ctc_fp32_MB"] = round(rss() - b)
b = rss(); v = VitsModel.from_pretrained("facebook/mms-tts-amh").eval(); v(**AutoTokenizer.from_pretrained("facebook/mms-tts-amh")("selam", return_tensors="pt")); out["mms_tts_vits_MB"] = round(rss() - b)
BIN = pathlib.Path("bin/llama-b11179"); g = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--mradermacher--Llama-3.2-1B-Amharic-Instruct-GGUF/snapshots/*/*.gguf"))[0]
s = subprocess.Popen([str(BIN / "llama-server"), "-m", g, "-t", "2", "-c", "2048", "--port", "8093", "--no-webui"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env={**os.environ, "LD_LIBRARY_PATH": str(BIN)})
while True:
    try:
        if json.load(urllib.request.urlopen("http://127.0.0.1:8093/health")).get("status") == "ok": break
    except Exception: time.sleep(0.3)
out["llama_server_1B_q4km_ctx2048_MB"] = round(psutil.Process(s.pid).memory_info().rss / 1e6); s.terminate(); s.wait()
out["system_total_RAM_GB"] = round(psutil.virtual_memory().total / 1e9, 1)
print(json.dumps(out, indent=1)); pathlib.Path("results/resources.json").write_text(json.dumps(out, indent=2))
