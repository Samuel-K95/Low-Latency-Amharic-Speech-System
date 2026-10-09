#!/usr/bin/env python3
"""
OmniVoice Amharic Text-to-Speech (TTS) Service.
Features in-memory base64 WAV streaming, fast 8-step first-sentence synthesis,
fixed male voice locking via OmniVoice voice cloning, and pre-cached clips.
"""
import io
import re
import time
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    from omnivoice import OmniVoice
    from omnivoice import OmniVoiceGenerationConfig
except Exception as exc:
    OmniVoice = None
    OmniVoiceGenerationConfig = None
    _OMNIVOICE_IMPORT_ERROR = exc
else:
    _OMNIVOICE_IMPORT_ERROR = None

from server.config import (
    TTS_MODEL_ID,
    TTS_DEVICE,
    TTS_SAMPLE_RATE,
    TTS_STEPS_FIRST,
    TTS_STEPS_FULL,
    TTS_GUIDANCE,
    TTS_SEED,
    TTS_VOICE_LOCK,
    TTS_VOICE_REF_SEED,
    TTS_VOICE_REF_TEXT,
    TTS_VOICE_REF_PATH,
    CHUNK_GAP_S,
    IDK_TEXT,
    IMPROVEMENTS,
)
from src.spoken import convert_to_spoken


class TTSService:
    def __init__(
        self,
        model_id: str = TTS_MODEL_ID,
        device: str = TTS_DEVICE,
        sample_rate: int = TTS_SAMPLE_RATE,
    ):
        self.model_id = model_id
        self.device = device
        self.sample_rate = sample_rate
        self.model = None
        self.idk_audio: Optional[Dict] = None
        self.router_audio_cache: Dict[str, Dict] = {}
        self.audio_memo: Dict[str, Dict] = {}
        self.voice_clone_prompt = None
        self.voice_ref_path: Optional[Path] = None
        self.is_ready = False

    def load(self):
        """Loads OmniVoice model into GPU memory and caches critical responses."""
        if self.is_ready:
            return

        if OmniVoice is None or OmniVoiceGenerationConfig is None:
            raise RuntimeError(
                "OmniVoice could not be imported. Install omnivoice==0.2.1 with "
                "transformers>=5.3.0 (required for HiggsAudioV2TokenizerModel). "
                "Do not re-pin transformers to 4.x — that breaks OmniVoice."
            ) from _OMNIVOICE_IMPORT_ERROR

        print(f"[TTSService] Loading OmniVoice TTS {self.model_id} on {self.device}...")
        self.model = OmniVoice.from_pretrained(
            self.model_id,
            device_map=self.device,
            dtype=torch.float16 if "cuda" in self.device else torch.float32
        )
        self.model.eval()
        # Prefer model-reported rate when available.
        self.sample_rate = int(getattr(self.model, "sampling_rate", None) or self.sample_rate)

        # Lock a consistent male voice BEFORE warm-up / cache fills.
        self._ensure_voice_lock()

        self.is_ready = True

        # Warm-up pass
        print("[TTSService] Pre-warming TTS engine...")
        try:
            _ = self._synthesize_raw("ሰላም", num_step=TTS_STEPS_FIRST)
        except Exception as e:
            print(f"[TTSService] Warm-up warning: {e}")

        # Pre-cache IDK ("I don't know") spoken audio
        print("[TTSService] Pre-caching IDK audio chunk...")
        try:
            self.idk_audio = self._synthesize_raw(IDK_TEXT, num_step=TTS_STEPS_FULL)
            self.idk_audio["cached"] = True
            self.idk_audio["time"] = 0.0
            self.idk_audio["rtf"] = 0.0
        except Exception as e:
            print(f"[TTSService] IDK cache warning: {e}")

        # Pre-cache common direct router responses (exact sentences the router returns).
        try:
            from src.router import route_query
        except ImportError:
            from src.router_latest import route_query

        for sample_q in ["ሰላም", "አመሰግናለሁ", "ቻው", "ማን ነህ?"]:
            try:
                routed = route_query(sample_q)
                direct = (routed.get("direct_response") or "").strip()
            except Exception:
                continue
            sentences = [p.strip() for p in re.findall(r"[^።!?]+[።!?]?", direct) if p.strip()]
            for sent in sentences:
                if sent not in self.router_audio_cache:
                    try:
                        cached_chunk = self._synthesize_raw(sent, num_step=TTS_STEPS_FULL)
                        cached_chunk["cached"] = True
                        cached_chunk["time"] = 0.0
                        cached_chunk["rtf"] = 0.0
                        self.router_audio_cache[sent] = cached_chunk
                    except Exception:
                        pass

        locked = "yes" if self.voice_clone_prompt is not None else "no"
        print(
            f"[TTSService] Ready. Cached {len(self.router_audio_cache)} router clips. "
            f"Voice lock: {locked}."
        )

    def _ensure_voice_lock(self) -> None:
        """
        Build / reuse a reference WAV and create an OmniVoice voice-clone prompt.

        OmniVoice.generate() without a prompt auto-picks a new voice each call.
        Passing the same voice_clone_prompt keeps a consistent male speaker.
        """
        if not TTS_VOICE_LOCK:
            self.voice_clone_prompt = None
            print("[TTSService] Voice lock disabled (TTS_VOICE_LOCK=0).")
            return

        if self.voice_clone_prompt is not None:
            return

        if self.model is None:
            return

        ref_path = Path(TTS_VOICE_REF_PATH)
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        ref_text = (TTS_VOICE_REF_TEXT or "").strip()
        if not ref_text:
            print("[TTSService] Voice lock skipped: empty TTS_VOICE_REF_TEXT.")
            return

        try:
            if not ref_path.exists():
                print(
                    f"[TTSService] Generating fixed male voice reference "
                    f"(seed={TTS_VOICE_REF_SEED}) -> {ref_path}"
                )
                config = OmniVoiceGenerationConfig(
                    num_step=TTS_STEPS_FULL, guidance_scale=TTS_GUIDANCE
                )
                torch.manual_seed(TTS_VOICE_REF_SEED)
                with torch.inference_mode():
                    audio = self.model.generate(
                        text=ref_text,
                        language="Amharic",
                        generation_config=config,
                    )
                wave = audio[0]
                if isinstance(wave, torch.Tensor):
                    wave = wave.detach().cpu().float().numpy()
                wave = np.asarray(wave, dtype=np.float32).squeeze()
                if sf is None:
                    raise RuntimeError("soundfile is required to save the voice reference WAV")
                sf.write(str(ref_path), wave, self.sample_rate)
            else:
                print(f"[TTSService] Reusing voice reference: {ref_path}")

            # Supply ref_text so OmniVoice skips whisper auto-transcribe of the clip.
            self.voice_clone_prompt = self.model.create_voice_clone_prompt(
                ref_audio=str(ref_path),
                ref_text=ref_text,
            )
            self.voice_ref_path = ref_path
            print(
                f"[TTSService] Voice locked (seed {TTS_VOICE_REF_SEED}). "
                "All synthesis will clone this speaker."
            )
        except Exception as e:
            # Non-fatal: keep serving with auto voice rather than failing startup.
            self.voice_clone_prompt = None
            print(f"[TTSService] Voice lock failed (continuing without lock): {e}")

    def prep_text(self, text: str) -> str:
        """Converts numbers to words, expands acronyms, and cleans for speech synthesis."""
        if not text:
            return ""
        if IMPROVEMENTS.get("numbers_to_words", True):
            text = convert_to_spoken(text)
        # Acronym mapping
        text = re.sub(r"\bAI\b", "ኤ አይ", text)
        text = re.sub(r"\bRAG\b", "ራግ", text)
        return text.strip()

    def _waveform_to_wav_bytes(self, wave: np.ndarray) -> bytes:
        """Encodes float32 waveform array to 16-bit PCM WAV bytes in-memory."""
        if sf is None:
            return b"RIFF" + bytes(len(wave) * 2)
        buf = io.BytesIO()
        sf.write(buf, wave, self.sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    def _synthesize_raw(self, text: str, num_step: int = TTS_STEPS_FULL) -> Dict:
        """Synthesizes text into waveform without disk I/O."""
        text = self.prep_text(text)
        if not text:
            raise ValueError("Empty text provided for synthesis.")

        config = OmniVoiceGenerationConfig(num_step=num_step, guidance_scale=TTS_GUIDANCE)
        torch.manual_seed(TTS_SEED)

        gen_kwargs = {
            "text": text,
            "language": "Amharic",
            "generation_config": config,
        }
        if self.voice_clone_prompt is not None:
            gen_kwargs["voice_clone_prompt"] = self.voice_clone_prompt

        t0 = time.perf_counter()
        with torch.inference_mode():
            audio = self.model.generate(**gen_kwargs)
        if torch.cuda.is_available() and "cuda" in self.device:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        wave = audio[0]
        if isinstance(wave, torch.Tensor):
            wave = wave.detach().cpu().float().numpy()
        wave = np.asarray(wave, dtype=np.float32).squeeze()

        wav_bytes = self._waveform_to_wav_bytes(wave)
        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
        duration = len(wave) / self.sample_rate
        rtf = elapsed / duration if duration > 0 else 0.0

        return {
            "text": text,
            "waveform": wave,
            "wav_bytes": wav_bytes,
            "audio_b64": audio_b64,
            "duration": duration,
            "time": elapsed,
            "rtf": rtf,
            "steps": num_step,
            "cached": False,
            "voice_locked": self.voice_clone_prompt is not None,
        }

    def is_idk(self, text: str) -> bool:
        """True for refusal-like text (fragments or full). Used by pipeline / history."""
        if not text:
            return False
        clean = re.sub(r"\s+", " ", text).strip()
        if IDK_TEXT and IDK_TEXT in clean:
            return True
        if "መረጃ የለኝም" in clean:
            return True
        if "ይቅርታ" in clean and "መልስ" in clean:
            return True
        return False

    def is_complete_idk(self, text: str) -> bool:
        """
        True only for a full refusal utterance.

        Mid-phrase fragments must NOT return the full cached IDK clip
        (that replayed the whole phrase twice when early_first_words split it).
        """
        if not text:
            return False
        clean = re.sub(r"\s+", " ", text).strip()
        if IDK_TEXT and (clean == IDK_TEXT or IDK_TEXT in clean):
            return True
        return "ይቅርታ" in clean and "መረጃ የለኝም" in clean

    def speak_sentence(self, text: str, is_first: bool = False) -> Optional[Dict]:
        """
        Synthesizes a sentence chunk.
        - Returns pre-cached audio for IDK or router answers.
        - Uses 8 steps for fast first sentence (TTFS < 1s) and 16 steps for subsequent sentences.
        """
        if not self.is_ready:
            self.load()

        clean_text = self.prep_text(text)
        if not clean_text:
            return None

        # 1. Full IDK only → one cached clip (never for mid-phrase fragments)
        if self.is_complete_idk(clean_text):
            if not self.idk_audio:
                return None
            cached = dict(self.idk_audio)
            cached["text"] = IDK_TEXT
            cached["time"] = 0.0
            cached["rtf"] = 0.0
            cached["cached"] = True
            return cached

        # 2. Check Router Cache
        if clean_text in self.router_audio_cache:
            cached = dict(self.router_audio_cache[clean_text])
            cached["time"] = 0.0
            cached["rtf"] = 0.0
            cached["cached"] = True
            return cached

        # 3. Check Audio Memo
        if clean_text in self.audio_memo:
            cached = dict(self.audio_memo[clean_text])
            cached["time"] = 0.0
            cached["rtf"] = 0.0
            cached["cached"] = True
            return cached

        # 4. Synthesize dynamically
        steps = TTS_STEPS_FIRST if is_first else TTS_STEPS_FULL
        result = self._synthesize_raw(clean_text, num_step=steps)

        if IMPROVEMENTS.get("answer_cache", True):
            self.audio_memo[clean_text] = result

        return result

    def join_waveforms(self, chunks: List[Dict]) -> Tuple[bytes, str, float]:
        """Concatenates multiple audio chunks with silence gap into single WAV bytes + base64."""
        if not chunks:
            return b"", "", 0.0

        if len(chunks) == 1:
            return chunks[0]["wav_bytes"], chunks[0]["audio_b64"], chunks[0]["duration"]

        gap_len = int(CHUNK_GAP_S * self.sample_rate)
        gap = np.zeros(gap_len, dtype=np.float32)

        parts = []
        for i, c in enumerate(chunks):
            parts.append(c["waveform"])
            if i < len(chunks) - 1:
                parts.append(gap)

        joined_wave = np.concatenate(parts)
        joined_bytes = self._waveform_to_wav_bytes(joined_wave)
        joined_b64 = base64.b64encode(joined_bytes).decode("utf-8")
        total_duration = len(joined_wave) / self.sample_rate

        return joined_bytes, joined_b64, total_duration
