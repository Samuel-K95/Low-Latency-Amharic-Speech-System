#!/usr/bin/env python3
"""
Hohe Amharic Automatic Speech Recognition (ASR) Service.
Handles in-memory audio decoding (base64 / raw WAV bytes) and GPU CTC transcription.
"""
import io
import re
import time
import base64
import numpy as np
import torch
try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    from transformers import AutoProcessor, AutoModelForCTC
except ImportError:
    AutoProcessor = None
    AutoModelForCTC = None

from server.config import ASR_MODEL_ID, ASR_DEVICE, ASR_SAMPLE_RATE

class ASRService:
    def __init__(self, model_id: str = ASR_MODEL_ID, device: str = ASR_DEVICE):
        self.model_id = model_id
        self.device = device
        self.dtype = torch.float16 if "cuda" in self.device else torch.float32
        self.processor = None
        self.model = None
        self.is_ready = False

    def load(self):
        """Loads ASR processor and model into memory/GPU and performs warm-up."""
        if self.is_ready:
            return

        print(f"[ASRService] Loading {self.model_id} onto {self.device} ({self.dtype})...")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForCTC.from_pretrained(
            self.model_id,
            dtype=self.dtype,
            low_cpu_mem_usage=True,
        ).to(self.device).eval()

        # Warm-up pass to eliminate initial CUDA kernel compilation lag
        dummy = self.processor(
            np.zeros(ASR_SAMPLE_RATE, dtype=np.float32),
            sampling_rate=ASR_SAMPLE_RATE,
            return_tensors="pt"
        )
        gpu_inputs = self._inputs_to_device(dummy)
        with torch.inference_mode():
            _ = self.model(**gpu_inputs).logits
        if torch.cuda.is_available() and "cuda" in self.device:
            torch.cuda.synchronize()

        self.is_ready = True
        print("[ASRService] Warm-up complete. ASR engine ready.")

    def _inputs_to_device(self, inputs):
        return {
            k: v.to(self.device, dtype=self.dtype) if v.is_floating_point() else v.to(self.device)
            for k, v in inputs.items()
        }

    def clean_text(self, text: str) -> str:
        """Removes [AMH] artifacts and standardizes whitespace."""
        return re.sub(r"\[AMH\]", "", text).strip()

    def _run_inference(self, audio: np.ndarray, t_start: float, load_s: float) -> dict:
        """Features -> CTC Model -> Decoded Text."""
        if not self.is_ready:
            self.load()

        t_prep = time.perf_counter()
        inputs = self.processor(audio, sampling_rate=ASR_SAMPLE_RATE, return_tensors="pt")
        gpu_inputs = self._inputs_to_device(inputs)
        prep_s = time.perf_counter() - t_prep

        t_infer = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(**gpu_inputs).logits
        if torch.cuda.is_available() and "cuda" in self.device:
            torch.cuda.synchronize()
        infer_s = time.perf_counter() - t_infer

        pred_ids = logits.argmax(dim=-1)
        raw = self.processor.batch_decode(pred_ids)[0]
        cleaned = self.clean_text(raw)

        return {
            "raw": raw,
            "transcript": cleaned,
            "load_s": load_s,
            "preprocessing_s": prep_s,
            "inference_s": infer_s,
            "total_s": time.perf_counter() - t_start,
        }

    def transcribe_bytes(self, wav_bytes: bytes) -> dict:
        """In-memory zero-copy transcription from raw WAV bytes."""
        t_start = time.perf_counter()
        try:
            audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
        except Exception:
            # If standard read fails, try librosa
            import librosa
            audio, sr = librosa.load(io.BytesIO(wav_bytes), sr=None, mono=True)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sr != ASR_SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=ASR_SAMPLE_RATE)

        audio = np.ascontiguousarray(audio, dtype=np.float32)
        load_s = time.perf_counter() - t_start
        return self._run_inference(audio, t_start, load_s)

    def transcribe_base64(self, b64_audio: str) -> dict:
        """Transcribes base64-encoded audio payload from WebSocket / REST."""
        if "," in b64_audio:
            b64_audio = b64_audio.split(",", 1)[1]
        raw_bytes = base64.b64decode(b64_audio)
        return self.transcribe_bytes(raw_bytes)

    def transcribe_file(self, file_path: str) -> dict:
        """Transcribes audio file from disk."""
        t_start = time.perf_counter()
        import librosa
        audio, _ = librosa.load(file_path, sr=ASR_SAMPLE_RATE, mono=True)
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        load_s = time.perf_counter() - t_start
        return self._run_inference(audio, t_start, load_s)
