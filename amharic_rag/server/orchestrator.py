#!/usr/bin/env python3
"""
Speech-to-Speech & Text-to-Speech Streaming Pipeline Orchestrator.
Coordinates ASR -> Query Intent Router -> Hybrid Retrieval -> llama-server SSE -> OmniVoice TTS.
Provides zero-latency intent bypass, semantic answer caching, and complete milestone telemetry.
"""
import re
import json
import time
from typing import AsyncGenerator, Dict, Generator, List, Optional, Tuple, Any

import numpy as np
import torch

from server.config import (
    MIN_DENSE_THRESHOLD,
    TOP_K_RETRIEVAL,
    IMPROVEMENTS,
    IDK_TEXT,
    FALLBACK_REPLY,
    DATA_DIR,
    ANSWER_CACHE_MAX,
    ANSWER_CACHE_WARM_N,
    WS_OMIT_FULL_AUDIO,
)
from server.asr_service import ASRService
from server.retrieval_service import RetrievalService
from server.llm_service import LLMService, sanitize_to_ethiopic
from server.tts_service import TTSService
from server.chat_history import ChatHistory, looks_like_idk
from server.event_compact import (
    compact_asr_metrics,
    compact_retrieved,
    build_stage_timings,
)
try:
    from src.router import route_query, QueryIntent
except ImportError:
    try:
        from src.router_latest import route_query, QueryIntent
    except ImportError:
        def route_query(q):
            return {"intent": "LEGAL_RETRIEVAL", "direct_response": None}

from src.fidel_fold import fold

DIRECT_INTENTS = {"GREETING", "GREETING_SMALLTALK", "THANK_YOU", "GOODBYE", "META_IDENTITY"}

class S2SOrchestrator:
    def __init__(
        self,
        asr_service: Optional[ASRService] = None,
        retrieval_service: Optional[RetrievalService] = None,
        llm_service: Optional[LLMService] = None,
        tts_service: Optional[TTSService] = None,
    ):
        self.asr = asr_service or ASRService()
        self.retrieval = retrieval_service or RetrievalService()
        self.llm = llm_service or LLMService()
        self.tts = tts_service or TTSService()
        self.answer_cache: List[Dict] = []
        self.is_initialized = False

    def initialize(self):
        """Loads and warms up all 4 AI engines."""
        if self.is_initialized:
            return

        print("\n=======================================================")
        print("  🚀 INITIALIZING AMHARIC SPEECH-TO-SPEECH RAG PIPELINE ")
        print("=======================================================")
        
        # 1. Load ASR
        self.asr.load()

        # 2. Load Hybrid Retrieval
        self.retrieval.load()

        # 3. Start llama-server
        self.llm.start()

        # 4. Load OmniVoice TTS
        self.tts.load()

        # 5. Warm retrieval + answer caches (no architecture change)
        self.warm_answer_cache()

        self.is_initialized = True
        print("=======================================================")
        print("  ✅ ALL 4 ENGINES LOADED & READY FOR LIVE TRAFFIC     ")
        print("=======================================================\n")

    def _trim_answer_cache(self) -> None:
        if len(self.answer_cache) > ANSWER_CACHE_MAX:
            self.answer_cache = self.answer_cache[-ANSWER_CACHE_MAX:]

    def warm_answer_cache(self, limit: Optional[int] = None) -> int:
        """
        Seed the semantic answer cache from curated FAQ pairs so repeated
        common labour-law questions skip LLM on first live traffic.
        """
        if not IMPROVEMENTS.get("answer_cache", True):
            return 0
        limit = ANSWER_CACHE_WARM_N if limit is None else limit
        path = DATA_DIR / "questions.json"
        if not path.exists() or not self.retrieval.is_ready:
            return 0

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            questions = data.get("questions") or []
        except Exception as e:
            print(f"[Orchestrator] Answer-cache warm skipped: {e}")
            return 0

        added = 0
        existing = {c["key"] for c in self.answer_cache}
        for item in questions:
            if added >= limit:
                break
            if not item.get("answerable"):
                continue
            q = (item.get("question") or "").strip()
            a = (item.get("answer") or "").strip()
            if not q or not a:
                continue
            key = self._cache_key(q)
            if not key or key in existing:
                continue
            sentences = [
                s.strip()
                for s in re.findall(r"[^።!?]+[።!?]?", a)
                if sanitize_to_ethiopic(s)
            ] or [a]
            try:
                qv = self.retrieval.encode_query(q)
            except Exception:
                continue
            self.answer_cache.append(
                {"key": key, "qv": qv, "sentences": sentences, "warm": True}
            )
            existing.add(key)
            added += 1

        self._trim_answer_cache()
        if added:
            print(f"[Orchestrator] Warmed answer cache with {added} FAQ entries.")
        return added

    def _done_event(
        self,
        *,
        full_text: str,
        audio_chunks: List[Dict],
        marks: Dict[str, float],
        stream_audio: bool,
        asr_res=None,
        ret_timing=None,
        retrieved_docs=None,
        final_llm_meta=None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        full_bytes, full_b64, full_dur = self.tts.join_waveforms(audio_chunks)
        stage = build_stage_timings(
            marks,
            asr_res=asr_res,
            ret_timing=ret_timing,
            llm_meta=final_llm_meta,
            audio_chunks=audio_chunks,
        )
        payload: Dict[str, Any] = {
            "type": "done",
            "full_text": full_text,
            "full_audio_duration_s": full_dur,
            "marks": marks,
            "stage_timings": stage,
            "total_compute_s": marks.get("all_done", stage["total_s"]),
        }
        # REST / non-stream callers still get the joined WAV; WS streaming skips it.
        if not (stream_audio and WS_OMIT_FULL_AUDIO):
            payload["full_audio_b64"] = full_b64
        if extra:
            payload.update(extra)
        return payload

    def _cache_key(self, text: str) -> str:
        return " ".join(re.findall(r"[ሀ-ፚ]+", fold(text)))

    def _cache_lookup(self, question: str) -> Tuple[Optional[Dict], Optional[str]]:
        """Looks up exact and semantic matches in the answer cache."""
        if not self.answer_cache:
            return None, None

        key = self._cache_key(question)
        for c in self.answer_cache:
            if c["key"] == key:
                return c, "exact"

        # Cosine lookup
        qv = self.retrieval.encode_query(question)
        sims = [float(c["qv"] @ qv) for c in self.answer_cache]
        best_idx = int(np.argmax(sims))
        if sims[best_idx] >= IMPROVEMENTS.get("cache_min", 0.95):
            return self.answer_cache[best_idx], f"semantic ({sims[best_idx]:.3f})"

        return None, None

    def stream_query(
        self,
        question: Optional[str] = None,
        audio_bytes: Optional[bytes] = None,
        audio_b64: Optional[str] = None,
        k: int = TOP_K_RETRIEVAL,
        stream_audio: bool = True,
        chat_history: Optional[ChatHistory] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Main pipeline generator. Accepts either text or audio input.
        Yields real-time events containing transcripts, retrieval context,
        synthesized sentence audio chunks, milestone metrics, and completion summaries.

        ``chat_history`` (optional, typically one per WebSocket session):
        - Prior turns go only into LLM messages.
        - Retrieval always uses the current transcript alone (no rewrite).
        """
        if not self.is_initialized:
            self.initialize()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t_start = time.perf_counter()
        marks: Dict[str, float] = {}
        history_enabled = bool(
            IMPROVEMENTS.get("chat_history", True) and chat_history is not None
        )

        def since():
            return time.perf_counter() - t_start

        # -------------------------------------------------------------
        # 1. ASR Phase (if audio provided)
        # -------------------------------------------------------------
        asr_res = None
        transcript = (question or "").strip()

        if audio_b64:
            asr_res = self.asr.transcribe_base64(audio_b64)
            transcript = asr_res["transcript"]
            marks["asr_done"] = since()
        elif audio_bytes:
            asr_res = self.asr.transcribe_bytes(audio_bytes)
            transcript = asr_res["transcript"]
            marks["asr_done"] = since()

        if not transcript:
            yield {"type": "error", "message": "No input text or speech provided."}
            return

        yield {
            "type": "transcript",
            "text": transcript,
            "asr_metrics": compact_asr_metrics(asr_res),
            "t": since()
        }

        # -------------------------------------------------------------
        # 2. Query Routing & Answer Cache Phase
        # -------------------------------------------------------------
        router_res = route_query(transcript) if IMPROVEMENTS.get("router", True) else {"intent": "LEGAL_RETRIEVAL"}
        intent_raw = router_res.get("intent", "LEGAL_RETRIEVAL")
        intent_str = str(getattr(intent_raw, "value", intent_raw))

        # Greeting / goodbye resets the session topic (history not used for social turns).
        if history_enabled:
            chat_history.on_intent(intent_str)

        yield {
            "type": "intent",
            "intent": intent_str,
            "is_direct": intent_str in DIRECT_INTENTS,
            "history_turns": len(chat_history) if history_enabled else 0,
            "t": since()
        }

        # Case A: Direct Conversational Bypass (Greetings, Thanks, Identity)
        if intent_str in DIRECT_INTENTS and router_res.get("direct_response"):
            direct_text = router_res["direct_response"]
            sentences = [s.strip() for s in re.findall(r"[^።!?]+[።!?]?", direct_text) if sanitize_to_ethiopic(s)]
            if not sentences:
                sentences = [direct_text]

            audio_chunks = []
            for i, sent in enumerate(sentences):
                audio_res = self.tts.speak_sentence(sent, is_first=(i == 0))
                if audio_res:
                    audio_chunks.append(audio_res)
                    if "first_audio_ready" not in marks:
                        marks["first_audio_ready"] = since()
                    yield {
                        "type": "audio",
                        "sentence": sent,
                        "audio_b64": audio_res["audio_b64"],
                        "duration": audio_res["duration"],
                        "is_first": (i == 0),
                        "cached": audio_res.get("cached", False),
                        "t": since()
                    }

            # Do not store greetings / thanks / goodbye / identity in history.
            marks["all_done"] = since()
            yield self._done_event(
                full_text=direct_text,
                audio_chunks=audio_chunks,
                marks=marks,
                stream_audio=stream_audio,
                asr_res=asr_res,
                extra={"retrieved": []},
            )
            return

        # Case B: Answer Cache Hit
        # Skip semantic cache when prior turns exist — short follow-ups must not
        # replay an unrelated cached answer ("እና ስንት?" ≠ prior FAQ).
        use_answer_cache = IMPROVEMENTS.get("answer_cache", True) and not (
            history_enabled and chat_history.has_context
        )
        if use_answer_cache:
            cache_hit, cache_how = self._cache_lookup(transcript)
            if cache_hit:
                cached_sentences = cache_hit["sentences"]
                cached_text = " ".join(cached_sentences)
                audio_chunks = []
                for i, sent in enumerate(cached_sentences):
                    audio_res = self.tts.speak_sentence(sent, is_first=(i == 0))
                    if audio_res:
                        audio_chunks.append(audio_res)
                        if "first_audio_ready" not in marks:
                            marks["first_audio_ready"] = since()
                        yield {
                            "type": "audio",
                            "sentence": sent,
                            "audio_b64": audio_res["audio_b64"],
                            "duration": audio_res["duration"],
                            "is_first": (i == 0),
                            "cached": True,
                            "cache_source": cache_how,
                            "t": since()
                        }
                if history_enabled:
                    chat_history.record_turn(
                        transcript,
                        cached_text,
                        intent=intent_str,
                        is_idk=looks_like_idk(cached_text),
                    )
                marks["all_done"] = since()
                yield self._done_event(
                    full_text=cached_text,
                    audio_chunks=audio_chunks,
                    marks=marks,
                    stream_audio=stream_audio,
                    asr_res=asr_res,
                    extra={"cached": True, "cache_source": cache_how},
                )
                return

        # -------------------------------------------------------------
        # 3. Hybrid RAG Retrieval Phase
        # No LLM query-rewrite. If prior turns exist, concatenate the last
        # user question with the current one so anaphoric follow-ups
        # ("እና የትኛው አንቀጽ?") still retrieve the right articles.
        # -------------------------------------------------------------
        retrieve_q = (
            chat_history.retrieval_query(transcript)
            if history_enabled
            else transcript
        )
        retrieved_docs, ret_timing = self.retrieval.retrieve(retrieve_q, k=k)
        marks["retrieval_done"] = since()
        top_dense = max((d["dense"] for d in retrieved_docs), default=0.0)

        yield {
            "type": "retrieved",
            "results": compact_retrieved(retrieved_docs) if stream_audio else retrieved_docs,
            "retrieval_timing": ret_timing,
            "top_dense": top_dense,
            "retrieval_query": retrieve_q,
            "t": since()
        }

        # -------------------------------------------------------------
        # 4. LLM Generation + OmniVoice TTS Streaming Phase
        # History = prior plain Q/A in chat messages.
        # RAG user message still uses the *current* transcript as QUESTION
        # (CONTEXT = chunks from retrieve_q above).
        # -------------------------------------------------------------
        history_messages = (
            chat_history.messages_for_llm() if history_enabled else []
        )
        user_prompt_content = self.llm.build_prompt_user_content(transcript, retrieved_docs)
        llm_gen = self.llm.generate_stream(
            user_prompt_content,
            history_messages=history_messages,
        )

        accumulated_sentences = []
        audio_chunks = []
        sentence_idx = 0
        final_llm_meta = {}
        # early_first_words can split the IDK refusal mid-phrase; each half then
        # matches is_idk() and would replay the *full* cached clip twice.
        idk_audio_emitted = False

        for ev in llm_gen:
            ev_type = ev.get("type")
            if ev_type == "first_token":
                marks["first_token"] = since()
                yield {"type": "first_token", "t": marks["first_token"]}

            elif ev_type == "sentence":
                sent_text = ev.get("text", "").strip()
                if not sent_text:
                    continue

                sent_is_idk = self.tts.is_idk(sent_text) or looks_like_idk(sent_text)
                if sent_is_idk:
                    if idk_audio_emitted:
                        continue
                    idk_audio_emitted = True
                    sent_text = IDK_TEXT  # one canonical refusal phrase / one clip

                if "first_sentence" not in marks:
                    marks["first_sentence"] = since()

                accumulated_sentences.append(sent_text)
                is_first = (sentence_idx == 0)

                # Synthesize sentence
                if stream_audio:
                    audio_res = self.tts.speak_sentence(sent_text, is_first=is_first)
                    if audio_res:
                        audio_chunks.append(audio_res)
                        if "first_audio_ready" not in marks:
                            marks["first_audio_ready"] = since()

                        yield {
                            "type": "audio",
                            "sentence": sent_text,
                            "audio_b64": audio_res["audio_b64"],
                            "duration": audio_res["duration"],
                            "is_first": is_first,
                            "cached": audio_res.get("cached", False),
                            "t": since()
                        }

                sentence_idx += 1

            elif ev_type == "done":
                marks["llm_done"] = since()
                final_llm_meta = ev

        full_response = " ".join(accumulated_sentences) or final_llm_meta.get("text", "")
        if not full_response:
            full_response = FALLBACK_REPLY

        # If audio streaming was disabled, synthesize the full response here
        if not stream_audio and full_response:
            audio_res = self.tts.speak_sentence(full_response, is_first=False)
            if audio_res:
                audio_chunks.append(audio_res)
                marks["first_audio_ready"] = since()

        response_is_idk = self.tts.is_idk(full_response) or looks_like_idk(full_response)

        # Update answer cache for successful legal answers
        if IMPROVEMENTS.get("answer_cache", True) and accumulated_sentences and not response_is_idk:
            try:
                qv = self.retrieval.encode_query(transcript)
                self.answer_cache.append({
                    "key": self._cache_key(transcript),
                    "qv": qv,
                    "sentences": accumulated_sentences
                })
                self._trim_answer_cache()
            except Exception:
                pass

        # Store legal Q/A for follow-ups (capped; IDK / empty skipped).
        if history_enabled:
            chat_history.record_turn(
                transcript,
                full_response,
                intent=intent_str,
                is_idk=response_is_idk,
            )

        marks["all_done"] = since()
        tts_total_s = sum(c["time"] for c in audio_chunks)
        peak_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

        metrics = {
            "transcript": transcript,
            "full_response": full_response,
            "idk": response_is_idk,
            "history_turns": len(chat_history) if history_enabled else 0,
            "retrieval_ms": ret_timing["total_ms"],
            "top_dense": top_dense,
            "retrieved_ids": [d["id"] for d in retrieved_docs],
            "llm_ttft_s": final_llm_meta.get("ttft_s", 0.0),
            "llm_first_sentence_s": final_llm_meta.get("first_sentence_s", 0.0),
            "new_tokens": final_llm_meta.get("new_tokens", 0),
            "tokens_per_sec": final_llm_meta.get("tok_s", 0.0),
            "tts_first_s": audio_chunks[0]["time"] if audio_chunks else 0.0,
            "tts_total_s": tts_total_s,
            "n_audio_chunks": len(audio_chunks),
            "first_audio_ready_s": marks.get("first_audio_ready", marks["all_done"]),
            "total_compute_s": marks["all_done"],
            "peak_vram_gb": peak_vram,
            "marks": marks,
        }

        yield self._done_event(
            full_text=full_response,
            audio_chunks=audio_chunks,
            marks=marks,
            stream_audio=stream_audio,
            asr_res=asr_res,
            ret_timing=ret_timing,
            retrieved_docs=retrieved_docs,
            final_llm_meta=final_llm_meta,
            extra={"metrics": metrics},
        )