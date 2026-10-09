import sys
import os
from pathlib import Path

# Setup path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import pytest
import numpy as np

def test_config_gpu_topology():
    """Verify GPU allocation logic."""
    from server.config import NUM_GPUS, MULTI_GPU, LLM_GPU_ID, ASR_DEVICE
    assert isinstance(NUM_GPUS, int)
    assert isinstance(MULTI_GPU, bool)
    if MULTI_GPU:
        assert LLM_GPU_ID == 1
    else:
        assert LLM_GPU_ID == 0

def test_llm_sanitize_ethiopic():
    """Verify Ethiopic text cleaning and thinking tag removal."""
    from server.llm_service import sanitize_to_ethiopic
    dirty_text = "<|channel>thinking deeply<channel|> ሰላም እንደምን ዋልክ 123 !? [noise]"
    cleaned = sanitize_to_ethiopic(dirty_text)
    assert "thinking" not in cleaned
    assert "ሰላም እንደምን ዋልክ" in cleaned
    assert "123" in cleaned

def test_split_ready_phrases():
    """Verify phrase boundary splitting on Ethiopian punctuation."""
    from server.llm_service import split_ready
    buf = "የሙከራ ጊዜ ስልሳ ቀናት ነው። ሁለተኛው ደንብ ይሄ ነው።"
    phrases, rest, pending = split_ready(buf, first=True)
    assert len(phrases) >= 1
    assert "የሙከራ ጊዜ" in phrases[0]

def test_minmax_scaling():
    """Verify score min-max scaling."""
    from server.retrieval_service import _minmax
    scores = np.array([0.2, 0.5, 0.8, 1.0])
    scaled = _minmax(scores)
    assert np.isclose(scaled.min(), 0.0)
    assert np.isclose(scaled.max(), 1.0)

def test_tts_service_helpers():
    """Verify TTS text preparation and IDK refusal detection."""
    from server.tts_service import TTSService
    from server.config import IDK_TEXT
    tts = TTSService(model_id="omnivoice-amharic", device="cpu")
    assert tts.is_idk(IDK_TEXT)
    assert tts.is_idk("ይቅርታ፣ ለዚህ ጥያቄ መልስ የሚሆን መረጃ የለኝም።")
    assert not tts.is_idk("የሙከራ ጊዜ ስልሳ ቀን ነው።")

    # Test waveform joining
    w1 = np.ones(1000, dtype=np.float32)
    w2 = np.ones(1000, dtype=np.float32)
    c1 = {"waveform": w1, "wav_bytes": b"1", "audio_b64": "MQ==", "duration": 0.5}
    c2 = {"waveform": w2, "wav_bytes": b"2", "audio_b64": "Mg==", "duration": 0.5}
    _, b64, total_dur = tts.join_waveforms([c1, c2])
    assert total_dur > 0.0
    assert len(b64) > 0

def test_router_direct_intents():
    """Verify direct intent bypass set."""
    from server.orchestrator import DIRECT_INTENTS
    assert "GREETING" in DIRECT_INTENTS
    assert "THANK_YOU" in DIRECT_INTENTS
    assert "GOODBYE" in DIRECT_INTENTS
    assert "LEGAL_RETRIEVAL" not in DIRECT_INTENTS

if __name__ == "__main__":
    test_config_gpu_topology()
    test_llm_sanitize_ethiopic()
    test_split_ready_phrases()
    test_minmax_scaling()
    test_tts_service_helpers()
    test_router_direct_intents()
    print("ALL SERVER TESTS PASSED SUCCESSFULLY!")
