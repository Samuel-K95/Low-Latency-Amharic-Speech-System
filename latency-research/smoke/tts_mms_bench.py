"""E-TTS-1: MMS-TTS (VITS) Amharic on laptop CPU.

Measures uroman romanisation time, synthesis time, audio length and RTF per
sentence; cold (first call) vs warm; 2 vs 4 threads. VITS is non-streaming, so
time-to-first-audio == full synthesis time of the first text chunk.
"""
import time, json, pathlib, statistics, torch, soundfile as sf
import uroman as ur
from transformers import VitsModel, AutoTokenizer
from sentences_am import TTS_SENTENCES

OUT = pathlib.Path("results"); WAV = pathlib.Path("audio/tts_mms"); WAV.mkdir(parents=True, exist_ok=True)
t0 = time.perf_counter()
model = VitsModel.from_pretrained("facebook/mms-tts-amh").eval()
tok = AutoTokenizer.from_pretrained("facebook/mms-tts-amh")
roman = ur.Uroman()
load_s = time.perf_counter() - t0
sr = model.config.sampling_rate

def synth(text):
    t = time.perf_counter(); rom = roman.romanize_string(text, lcode="amh"); t_rom = time.perf_counter() - t
    t = time.perf_counter()
    with torch.inference_mode():
        wav = model(**tok(rom, return_tensors="pt")).waveform[0].numpy()
    return rom, wav, t_rom, time.perf_counter() - t

results = {"model": "facebook/mms-tts-amh", "params_M": round(sum(p.numel() for p in model.parameters())/1e6, 1),
           "load_s": round(load_s, 2), "sample_rate": sr, "runs": []}
for threads in (4, 2):
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    _, _, _, cold = synth("ሰላም")            # first call includes lazy init
    for key, text in TTS_SENTENCES.items():
        times = []
        for rep in range(3):
            rom, wav, t_rom, t_syn = synth(text)
            times.append(t_syn)
        dur = len(wav) / sr
        if threads == 4:
            sf.write(WAV / f"{key}.wav", wav, sr)
        r = {"threads": threads, "key": key, "chars": len(text), "words": len(text.split()),
             "uroman_ms": round(t_rom*1000, 1), "synth_ms_median": round(statistics.median(times)*1000),
             "audio_s": round(dur, 2), "rtf": round(statistics.median(times)/dur, 3), "romanized": rom}
        results["runs"].append(r)
        print(f"thr={threads} {key:<13} words={r['words']:>2} audio={dur:5.2f}s synth={r['synth_ms_median']:>5}ms "
              f"RTF={r['rtf']:.3f} uroman={r['uroman_ms']}ms | {rom[:70]}", flush=True)
    results[f"cold_first_call_ms_threads{threads}"] = round(cold*1000)
OUT.joinpath("tts_mms.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
print("load", results["load_s"], "s; params", results["params_M"], "M")
