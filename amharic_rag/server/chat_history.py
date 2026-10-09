#!/usr/bin/env python3
"""
Bounded multi-turn chat history for the S2S RAG assistant.

Latency-safe rules:
- History is injected only into LLM chat messages (never into retrieval).
- Cap length per message; keep only the last N user/assistant turns.
- Drop greetings / thanks / goodbye / identity and IDK refusals.
- Clear on greeting / goodbye (topic reset).
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from server.config import (
    CHAT_HISTORY_ENABLED,
    CHAT_HISTORY_MAX_TURNS,
    CHAT_HISTORY_MAX_CHARS,
    IDK_TEXT,
)

# Intents that reset the conversation (new topic / social open-close).
CLEAR_INTENTS = frozenset(
    {"GREETING", "GREETING_SMALLTALK", "GOODBYE"}
)

# Direct social / meta replies are not stored (and greetings clear history).
SKIP_STORE_INTENTS = frozenset(
    {
        "GREETING",
        "GREETING_SMALLTALK",
        "THANK_YOU",
        "GOODBYE",
        "META_IDENTITY",
    }
)


def cap_text(text: str, max_chars: int = CHAT_HISTORY_MAX_CHARS) -> str:
    """Whitespace-normalize and hard-cap a turn for the prompt."""
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    if max_chars <= 0 or len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max(1, max_chars - 1)].rstrip() + "…"


def looks_like_idk(text: str) -> bool:
    """True for refusal / empty answers that must not enter history."""
    if not text or not str(text).strip():
        return True
    clean = re.sub(r"\s+", " ", str(text)).strip()
    if IDK_TEXT and IDK_TEXT in clean:
        return True
    if "መረጃ የለኝም" in clean:
        return True
    if "ይቅርታ" in clean and "መልስ" in clean:
        return True
    return False


class ChatHistory:
    """
    Stores up to ``max_turns`` (user, assistant) pairs for one WS session.

    ``messages_for_llm()`` returns prior turns as plain chat messages.
    The current turn's RAG ``CONTEXT`` + ``QUESTION`` is appended separately
    by ``LLMService.generate_stream`` — retrieval stays single-question.
    """

    def __init__(
        self,
        max_turns: int = CHAT_HISTORY_MAX_TURNS,
        max_chars: int = CHAT_HISTORY_MAX_CHARS,
        enabled: bool = CHAT_HISTORY_ENABLED,
    ):
        self.max_turns = max(0, int(max_turns))
        self.max_chars = max(8, int(max_chars))
        self.enabled = bool(enabled)
        self._turns: List[Tuple[str, str]] = []

    def __len__(self) -> int:
        return len(self._turns)

    @property
    def has_context(self) -> bool:
        return self.enabled and bool(self._turns)

    def clear(self) -> None:
        self._turns.clear()

    def messages_for_llm(self) -> List[Dict[str, str]]:
        """Prior turns only (no system, no current question)."""
        if not self.enabled or not self._turns:
            return []
        out: List[Dict[str, str]] = []
        for user, assistant in self._turns:
            out.append({"role": "user", "content": user})
            out.append({"role": "assistant", "content": assistant})
        return out

    def last_user(self) -> str:
        """Most recent stored user question, or empty."""
        if not self._turns:
            return ""
        return self._turns[-1][0]

    def retrieval_query(self, current_question: str) -> str:
        """
        Build a search string for FAISS/TF-IDF without an LLM rewrite.

        Follow-ups like "እና የትኛው አንቀጽ?" have almost no keywords alone.
        Prefixing the previous user question keeps retrieval sharp and adds
        negligible latency (one concatenated string, same retrieve() call).
        """
        current = re.sub(r"\s+", " ", (current_question or "")).strip()
        if not self.has_context:
            return current
        prev = self.last_user()
        if not prev:
            return current
        if not current:
            return prev
        # Avoid duplicating if the user repeated the same question.
        if prev in current or current in prev:
            return current
        return f"{prev} {current}"

    def on_intent(self, intent: str) -> None:
        """Clear history when the user greets or says goodbye."""
        if not self.enabled:
            return
        intent_str = str(getattr(intent, "value", intent) or "")
        if intent_str in CLEAR_INTENTS:
            self.clear()

    def record_turn(
        self,
        user_text: str,
        assistant_text: str,
        *,
        intent: Optional[str] = None,
        is_idk: Optional[bool] = None,
    ) -> bool:
        """
        Store a completed Q/A turn if it is worth keeping for follow-ups.

        Returns True if the turn was stored.
        """
        if not self.enabled or self.max_turns <= 0:
            return False

        intent_str = str(getattr(intent, "value", intent) or "")
        if intent_str in SKIP_STORE_INTENTS:
            return False

        if is_idk is None:
            is_idk = looks_like_idk(assistant_text)
        if is_idk:
            return False

        user = cap_text(user_text, self.max_chars)
        assistant = cap_text(assistant_text, self.max_chars)
        if not user or not assistant:
            return False

        self._turns.append((user, assistant))
        if len(self._turns) > self.max_turns:
            self._turns = self._turns[-self.max_turns :]
        return True

    def snapshot(self) -> Sequence[Tuple[str, str]]:
        """Immutable view for tests / telemetry."""
        return tuple(self._turns)
