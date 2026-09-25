"""E-TTS-2: intelligibility proxy — transcribe MMS-TTS output with the best ASR we have and compute CER.
High CER on 'digits_date' / 'code_switch' exposes text-normalisation gaps rather than acoustic quality."""
import json, pathlib, numpy as np, soundfile as sf, ctranslate2, jiwer
from faster_whisper import WhisperModel
from am_text import normalize
from sentences_am import TTS_SENTENCES
m = WhisperModel("models/whisper-small-waxal-amh-ct2-int8", device="cpu", compute_type="int8", cpu_threads=4)
P = m.hf_tokenizer.encode("<|startoftranscript|><|am|><|transcribe|><|notimestamps|>", add_special_tokens=False).ids
rows = []
for key, text in TTS_SENTENCES.items():
    wav, sr = sf.read(f"audio/tts_mms/{key}.wav", dtype="float32")
    assert sr == 16000, sr                      # MMS-TTS already outputs 16 kHz
    f = m.feature_extractor(wav)[:, :3000]; f = np.pad(f, ((0, 0), (0, 3000 - f.shape[1])))
    enc = m.model.encode(ctranslate2.StorageView.from_array(np.ascontiguousarray(f[None])), to_cpu=False)
    hyp = m.hf_tokenizer.decode(m.model.generate(enc, [P], max_length=880, beam_size=1)[0].sequences_ids[0]).strip()
    cer = jiwer.cer(normalize(text), normalize(hyp) or "-")
    rows.append({"key": key, "ref": text, "asr_of_tts": hyp, "cer": round(cer * 100, 1)}); print(rows[-1], flush=True)
pathlib.Path("results/tts_roundtrip_mms.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
