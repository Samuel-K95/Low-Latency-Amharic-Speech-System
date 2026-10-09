"""Voice-lock wiring tests (no GPU / no OmniVoice weights required)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


def test_config_voice_lock_defaults():
    from server import config

    assert config.TTS_VOICE_LOCK is True
    assert config.TTS_VOICE_REF_SEED == 42
    assert "ረዳትዎ" in config.TTS_VOICE_REF_TEXT
    assert config.TTS_VOICE_REF_PATH.name == "voice_ref_male.wav"


def test_synthesize_raw_passes_voice_clone_prompt():
    from server.tts_service import TTSService
    import server.tts_service as tts_mod

    tts = TTSService(device="cpu")
    tts.is_ready = True
    tts.sample_rate = 24000
    tts.voice_clone_prompt = object()

    fake_wave = np.zeros(2400, dtype=np.float32)
    tts.model = MagicMock()
    tts.model.generate.return_value = [fake_wave]

    with patch.object(tts_mod, "OmniVoiceGenerationConfig", MagicMock(return_value=MagicMock())):
        with patch.object(tts_mod.torch, "manual_seed"):
            result = tts._synthesize_raw("ሰላም", num_step=8)

    kwargs = tts.model.generate.call_args.kwargs
    assert kwargs["voice_clone_prompt"] is tts.voice_clone_prompt
    assert result["voice_locked"] is True
    assert result["duration"] > 0


def test_ensure_voice_lock_disabled():
    from server.tts_service import TTSService
    import server.tts_service as tts_mod

    tts = TTSService(device="cpu")
    tts.model = MagicMock()
    with patch.object(tts_mod, "TTS_VOICE_LOCK", False):
        tts._ensure_voice_lock()
    assert tts.voice_clone_prompt is None
    tts.model.generate.assert_not_called()
    tts.model.create_voice_clone_prompt.assert_not_called()


def test_ensure_voice_lock_creates_ref_and_prompt(tmp_path):
    from server.tts_service import TTSService
    import server.tts_service as tts_mod

    ref_path = tmp_path / "voice_ref_male.wav"
    tts = TTSService(device="cpu")
    tts.sample_rate = 24000

    fake_wave = np.zeros(2400, dtype=np.float32)
    tts.model = MagicMock()
    tts.model.generate.return_value = [fake_wave]
    tts.model.create_voice_clone_prompt.return_value = {"locked": True}

    sf_stub = tts_mod.sf if tts_mod.sf is not None else MagicMock()
    with patch.object(tts_mod, "TTS_VOICE_LOCK", True), patch.object(
        tts_mod, "TTS_VOICE_REF_PATH", ref_path
    ), patch.object(tts_mod, "TTS_VOICE_REF_SEED", 42), patch.object(
        tts_mod, "TTS_VOICE_REF_TEXT", "ሰላም፣ እኔ ረዳትዎ ነኝ።"
    ), patch.object(
        tts_mod, "OmniVoiceGenerationConfig", MagicMock(return_value=MagicMock())
    ), patch.object(tts_mod.torch, "manual_seed"), patch.object(
        tts_mod, "sf", sf_stub
    ), patch.object(sf_stub, "write") as write_mock:
        tts._ensure_voice_lock()
        write_mock.assert_called_once()

    assert tts.voice_clone_prompt == {"locked": True}
    tts.model.create_voice_clone_prompt.assert_called_once()
    call_kwargs = tts.model.create_voice_clone_prompt.call_args.kwargs
    assert call_kwargs["ref_audio"] == str(ref_path)
    assert "ረዳትዎ" in call_kwargs["ref_text"]


def test_ensure_voice_lock_reuses_existing_wav(tmp_path):
    from server.tts_service import TTSService
    import server.tts_service as tts_mod

    ref_path = tmp_path / "voice_ref_male.wav"
    ref_path.write_bytes(b"RIFF....")

    tts = TTSService(device="cpu")
    tts.model = MagicMock()
    tts.model.create_voice_clone_prompt.return_value = "PROMPT"

    with patch.object(tts_mod, "TTS_VOICE_LOCK", True), patch.object(
        tts_mod, "TTS_VOICE_REF_PATH", ref_path
    ), patch.object(tts_mod, "TTS_VOICE_REF_TEXT", "ሰላም"):
        tts._ensure_voice_lock()

    tts.model.generate.assert_not_called()
    assert tts.voice_clone_prompt == "PROMPT"


def test_v2_notebook_no_mic_test_cells_removed():
    import json

    nb_path = ROOT_DIR.parent / "amharic-speech-to-speech-ragv2.ipynb"
    assert nb_path.exists()
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    joined = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
    assert "no microphone needed" not in joined
    assert "TEST_QUESTION_AUDIO_PATH" not in joined
    assert "run_s2s(TEST_QUESTION_AUDIO_PATH)" not in joined
    # Voice lock cells should still be present in the v2 notebook.
    assert "create_voice_clone_prompt" in joined
    assert "VOICE_CLONE_PROMPT" in joined
