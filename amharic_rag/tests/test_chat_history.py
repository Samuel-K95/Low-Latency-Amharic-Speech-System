"""Chat-history unit tests (no GPU / no models)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


def test_config_history_and_max_tokens():
    from server import config

    assert config.LLM_MAX_TOKENS == 200
    assert config.CHAT_HISTORY_ENABLED is True
    assert config.CHAT_HISTORY_MAX_TURNS == 2
    assert config.CHAT_HISTORY_MAX_CHARS == 280
    assert config.IMPROVEMENTS.get("chat_history") is True


def test_cap_and_idk_helpers():
    from server.chat_history import cap_text, looks_like_idk
    from server.config import IDK_TEXT

    assert cap_text("  hello   world  ", 20) == "hello world"
    long = "x" * 400
    capped = cap_text(long, 280)
    assert len(capped) <= 280
    assert capped.endswith("…")
    assert looks_like_idk(IDK_TEXT)
    assert looks_like_idk("sorry IDK " + "መረጃ የለኝም")
    assert not looks_like_idk("normal legal answer about probation")


def test_history_keeps_last_two_turns_and_caps():
    from server.chat_history import ChatHistory

    h = ChatHistory(max_turns=2, max_chars=20, enabled=True)
    assert h.record_turn("q1 " + "x" * 50, "a1 " + "y" * 50)
    assert h.record_turn("q2", "a2")
    assert h.record_turn("q3", "a3")
    assert len(h) == 2
    snap = h.snapshot()
    assert snap[0][0] == "q2"
    assert snap[1][0] == "q3"

    h2 = ChatHistory(max_turns=2, max_chars=20, enabled=True)
    h2.record_turn("question-one-long-text-here", "answer-one-long-text-here")
    u, a = h2.snapshot()[0]
    assert len(u) <= 20 and u.endswith("…")
    assert len(a) <= 20 and a.endswith("…")


def test_messages_for_llm_shape():
    from server.chat_history import ChatHistory

    h = ChatHistory(max_turns=2, enabled=True)
    h.record_turn("probation days?", "sixty days.")
    h.record_turn("and salary?", "per the contract.")
    msgs = h.messages_for_llm()
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert msgs[0]["content"] == "probation days?"
    assert msgs[-1]["content"] == "per the contract."


def test_retrieval_query_prefixes_last_user():
    from server.chat_history import ChatHistory

    h = ChatHistory(max_turns=2, enabled=True)
    assert h.retrieval_query("alone") == "alone"
    h.record_turn("probation how many days?", "sixty.")
    q = h.retrieval_query("and which article?")
    assert q.startswith("probation how many days?")
    assert "and which article?" in q
    # No LLM rewrite — just string concat.
    assert h.retrieval_query("probation how many days?") == "probation how many days?"


def test_skip_greetings_idk_and_clear_on_greeting():
    from server.chat_history import ChatHistory
    from server.config import IDK_TEXT

    h = ChatHistory(max_turns=2, enabled=True)
    assert not h.record_turn("hi", "hello!", intent="GREETING_SMALLTALK")
    assert not h.record_turn("q", IDK_TEXT, intent="LEGAL_RETRIEVAL")
    assert h.record_turn("probation?", "sixty days.")
    assert len(h) == 1
    h.on_intent("GREETING")
    assert len(h) == 0
    h.record_turn("salary?", "answer.")
    h.on_intent("GOODBYE")
    assert len(h) == 0


def test_generate_stream_includes_history_before_current_user():
    from server.llm_service import LLMService

    llm = LLMService.__new__(LLMService)
    llm.system_prompt = "SYS"
    llm.server_url = "http://127.0.0.1:9"
    llm.ensure_server = MagicMock()

    captured = {}

    class FakeResp:
        def raise_for_status(self):
            return None

        def iter_lines(self):
            return iter([])

    def fake_post(url, json=None, stream=None, timeout=None):
        captured["payload"] = json
        return FakeResp()

    history = [
        {"role": "user", "content": "prev q"},
        {"role": "assistant", "content": "prev a"},
    ]
    with patch("server.llm_service.requests.post", side_effect=fake_post):
        list(
            llm.generate_stream(
                "CONTEXT:\nx\n\nQUESTION:\nnow",
                history_messages=history,
            )
        )

    roles = [m["role"] for m in captured["payload"]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert captured["payload"]["messages"][1]["content"] == "prev q"
    assert captured["payload"]["messages"][-1]["content"].startswith("CONTEXT:")
    assert captured["payload"]["max_tokens"] == 200


def test_orchestrator_wires_history_without_retrieval_rewrite():
    """Source contract: optional simple concat retrieve_q; no LLM rewriter."""
    src = (ROOT_DIR / "server" / "orchestrator.py").read_text(encoding="utf-8")
    assert "chat_history: Optional[ChatHistory] = None" in src
    assert "chat_history.retrieval_query(transcript)" in src
    assert "self.retrieval.retrieve(retrieve_q, k=k)" in src
    assert "history_messages=history_messages" in src
    assert "chat_history.messages_for_llm()" in src
    assert "chat_history.record_turn(" in src
    assert "chat_history.on_intent(" in src
    # Follow-ups must not hit semantic answer cache.
    assert "history_enabled and chat_history.has_context" in src
    # Still no LLM query-rewriter service.
    assert "query_rewrite" not in src
    assert "QueryRewriter" not in src


def test_app_ws_passes_session_history():
    app_src = (ROOT_DIR / "server" / "app.py").read_text(encoding="utf-8")
    assert "session_history = ChatHistory" in app_src
    assert "chat_history=session_history" in app_src
    assert "ChatHistory" in app_src


def test_idk_dedupe_guard_in_orchestrator():
    """IDK must not be spoken twice when early_first_words splits the refusal."""
    src = (ROOT_DIR / "server" / "orchestrator.py").read_text(encoding="utf-8")
    assert "idk_audio_emitted" in src
    assert "sent_text = IDK_TEXT" in src
