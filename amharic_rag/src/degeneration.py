#!/usr/bin/env python3
"""
Degeneration detection for LLM output text - pure functions, no model/GPU dependency.

Adapted from the equivalent module built for the sibling Kaggle project (cells/), where these
exact heuristics (not invented ones) caught real production failures: repeated-syllable
collapse, fake numbered lists, and sentences re-stated with a few words corrupted by typos.
This project serves its LLM through llama.cpp at temperature=0.0 (greedy) rather than the
hand-rolled transformers decode loop that produced those failures, but greedy decoding without
a repeat-last-n window is its own well-known path to exact-token-loop repetition - a different
mechanism, the same observable shape. These checks exist to find out whether it happens here,
not to assume a more correct decode engine makes the check unnecessary.
"""
import difflib
import re
import unicodedata
from dataclasses import dataclass

ETHIOPIC = re.compile(r"[ሀ-፿]")
LIST_MARKER = re.compile(r"^\s*\d+\s*[\.\)]")

ACCEPT, FLAG = "accept", "flag"


@dataclass(frozen=True)
class GuardConfig:
    max_char_run: int = 4
    min_ethiopic_ratio: float = 0.60
    min_distinct_ratio: float = 0.35
    min_chars: int = 4
    max_word_repeats: int = 3
    near_dup_jaccard: float = 0.90
    near_dup_char_ratio: float = 0.80


@dataclass(frozen=True)
class Verdict:
    action: str
    reason: str = ""


def normalize_for_dedup(text):
    """Punctuation- and spacing-insensitive form. Keeps only U+1200-U+135F (Ethiopic letters
    and combining marks) - the range must stop short of U+1360, because Ethiopic punctuation
    (። ፣ ፤ ፧, U+1360-U+1368) lives inside the Ethiopic block. Filtering on the whole block would
    keep that punctuation and make two otherwise-identical sentences compare as different."""
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", re.sub(r"[^ሀ-፟\s]", "", text)).strip()


def longest_char_run(text):
    core = re.sub(r"\s", "", text)
    if not core:
        return 0
    best = run = 1
    for a, b in zip(core, core[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best


def ethiopic_ratio(text):
    core = re.sub(r"\s", "", text)
    return len(ETHIOPIC.findall(core)) / len(core) if core else 0.0


def distinct_char_ratio(text):
    core = re.sub(r"\s", "", text)
    return len(set(core)) / len(core) if core else 0.0


def inspect_text(text, cfg=GuardConfig()):
    """Flags a single answer for internal degeneration (no cross-answer state needed - this
    project's answers are meant to be one to two sentences, so a repeated word/char run within
    one answer is already a clear signal)."""
    stripped = text.strip()
    dense = re.sub(r"\s", "", stripped)

    if len(stripped) < cfg.min_chars:
        return Verdict(FLAG, "empty-or-near-empty")
    if LIST_MARKER.match(stripped):
        return Verdict(FLAG, "list-marker")
    if longest_char_run(stripped) >= cfg.max_char_run:
        return Verdict(FLAG, "char-run")
    if len(dense) > 6 and ethiopic_ratio(stripped) < cfg.min_ethiopic_ratio:
        return Verdict(FLAG, "non-ethiopic")
    if len(dense) > 12 and distinct_char_ratio(stripped) < cfg.min_distinct_ratio:
        return Verdict(FLAG, "low-entropy")
    words = stripped.split()
    if words and max(words.count(w) for w in set(words)) >= cfg.max_word_repeats:
        return Verdict(FLAG, "word-loop")
    return Verdict(ACCEPT)


def is_near_duplicate(text_a, text_b, cfg=GuardConfig()):
    """True if two answers are the same content restated - either verbatim, or with a few
    words corrupted by typos (word-set Jaccard misses this: a corrupted word doesn't match its
    original at all, so Jaccard can score as low as 0.40 on a pair a human reads as
    "obviously the same sentence, slightly garbled" - character-level similarity catches it)."""
    norm_a, norm_b = normalize_for_dedup(text_a), normalize_for_dedup(text_b)
    if not norm_a or not norm_b:
        return False
    if norm_a == norm_b:
        return True
    words_a, words_b = set(norm_a.split()), set(norm_b.split())
    if words_a and words_b:
        jaccard = len(words_a & words_b) / len(words_a | words_b)
        if jaccard >= cfg.near_dup_jaccard:
            return True
    return difflib.SequenceMatcher(None, norm_a, norm_b).ratio() >= cfg.near_dup_char_ratio
