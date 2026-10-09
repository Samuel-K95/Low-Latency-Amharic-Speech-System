"""IDK phrase must not emit / play two full TTS clips when streamed."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


def test_streaming_idk_no_longer_early_splits():
    from server.llm_service import split_ready
    from server.config import IDK_TEXT

    buf = ""
    first = True
    emits = []
    for i in range(0, len(IDK_TEXT), 2):
        buf += IDK_TEXT[i : i + 2]
        phrases, buf, first = split_ready(buf, first)
        emits.extend(phrases)
    if buf.strip():
        emits.append(buf.strip())

    # Should arrive as one refusal (period boundary), not two early-cut halves.
    assert len(emits) == 1
    assert "ይቅርታ" in emits[0] and "መረጃ የለኝም" in emits[0]


def test_complete_idk_vs_fragment():
    from server.tts_service import TTSService
    from server.config import IDK_TEXT

    tts = TTSService.__new__(TTSService)
    half = "ይቅርታ፣ ለዚህ ጥያቄ መልስ "
    assert tts.is_complete_idk(IDK_TEXT)
    assert tts.is_idk(half)
    assert not tts.is_complete_idk(half)
    assert tts.is_idk("የሚሆን መረጃ የለኝም።")
    assert not tts.is_complete_idk("የሚሆን መረጃ የለኝም።")


def test_idk_guards_in_sources():
    orch = (ROOT_DIR / "server" / "orchestrator.py").read_text(encoding="utf-8")
    llm = (ROOT_DIR / "server" / "llm_service.py").read_text(encoding="utf-8")
    app = (ROOT_DIR / "server" / "app.py").read_text(encoding="utf-8")
    assert "idk_audio_emitted" in orch
    assert "_is_idk_buffer" in llm
    assert "is_complete_idk" in (ROOT_DIR / "server" / "tts_service.py").read_text(
        encoding="utf-8"
    )
    assert "audioQueue[audioQueue.length - 1] === b64" in app
