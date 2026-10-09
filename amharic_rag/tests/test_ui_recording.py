"""
Phase 4 regression: embedded UI recording / reconnect lifecycle markers.

Browser MediaRecorder cannot run in this environment; we assert the shipped
HTML/JS in ``server/app.py`` implements the Phase 4 contract.
"""
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

APP_PATH = ROOT_DIR / "server" / "app.py"


def _ui_script() -> str:
    src = APP_PATH.read_text(encoding="utf-8")
    match = re.search(r"<script>(.*?)</script>", src, flags=re.DOTALL)
    assert match, "Embedded UI <script> block missing from app.py"
    return match.group(1)


def test_ui_reuses_single_mic_stream():
    script = _ui_script()
    assert "let micStream = null" in script
    assert "async function ensureMicStream" in script
    assert "micStreamIsLive" in script
    # Must not open a fresh getUserMedia on every click without reuse helper.
    assert script.count("getUserMedia") >= 1
    assert "ensureMicStream()" in script


def test_ui_releases_tracks_and_resets_recorder():
    script = _ui_script()
    assert "function releaseMicStream" in script
    assert "t.stop()" in script or "track.stop()" in script or ".stop()" in script
    assert "function resetRecorderState" in script
    assert "onStopHandled" in script
    assert "if (onStopHandled) return" in script


def test_ui_bounded_reconnect_backoff():
    script = _ui_script()
    assert "MAX_RECONNECT_ATTEMPTS" in script
    assert "RECONNECT_BASE_MS" in script
    assert "RECONNECT_MAX_MS" in script
    assert "function scheduleReconnect" in script
    assert "Math.pow(2, reconnectAttempt)" in script
    assert "ws.onerror" in script
    assert "ws.onclose" in script


def test_ui_guards_socket_and_inflight():
    script = _ui_script()
    assert "function canSendOnSocket" in script
    assert "WebSocket.OPEN" in script
    assert "requestInFlight" in script
    assert "Socket not ready" in script or "canSendOnSocket()" in script


def test_ui_reenables_mic_after_terminal_events():
    script = _ui_script()
    assert "function setMicEnabled" in script
    assert "function armRequestTimeout" in script
    assert "REQUEST_TIMEOUT_MS" in script
    assert "markRequestFinished" in script
    # done / error / cancelled / timeout paths re-enable mic
    assert "setMicEnabled(true)" in script


def test_ui_clears_playback_on_new_request_and_disconnect():
    script = _ui_script()
    assert "function clearPlayback" in script
    assert "currentAudio" in script
    assert "clearPlayback()" in script


def test_ui_preserves_pcm_wav_conversion():
    script = _ui_script()
    assert "function encodeWav" in script
    assert "async function mediaBlobToWav" in script
    assert "decodeAudioData" in script
    assert "'RIFF'" in script or '"RIFF"' in script


def test_ui_pagehide_releases_resources():
    script = _ui_script()
    assert "pagehide" in script
    assert "releaseMicStream()" in script
