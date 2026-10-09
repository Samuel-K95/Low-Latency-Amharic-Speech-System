"""
Unit tests for src/degeneration.py.

Fixtures are drawn from the failure patterns actually observed in the sibling Kaggle project
(cells/), not invented - see src/degeneration.py's module docstring for why they transfer here
despite the different inference engine (llama.cpp vs. transformers): greedy decoding without a
repeat-last-n window is a different path to the same observable shape (repeated syllables,
fake lists, restated sentences), not a reason to assume it can't happen.
"""
from src.degeneration import (
    ACCEPT, FLAG, GuardConfig, inspect_text, is_near_duplicate, normalize_for_dedup,
)

GOOD_SHORT_ANSWER = "በአዋጁ አንቀጽ አስራ አንድ መሠረት የሙከራ ጊዜ ስልሳ ቀናት ነው።"
GOOD_IDK = "ይቅርታ፣ ለዚህ ጥያቄ መልስ የሚሆን መረጃ የለኝም።"


def test_good_answers_are_accepted():
    for text in (GOOD_SHORT_ANSWER, GOOD_IDK, "ሰላም! ጤና ይስጥልኝ።"):
        assert inspect_text(text).action == ACCEPT


def test_char_run_is_flagged():
    assert inspect_text("ኡኡኡኡ ሌላ ነገር እዚህ አለ").reason == "char-run"


def test_low_entropy_repeated_syllable_is_flagged():
    assert inspect_text("ቃልቃልቃልቃልቃልቃልቃል").reason == "low-entropy"


def test_fake_list_marker_is_flagged():
    assert inspect_text("1. የሙከራ ጊዜ ስልሳ ቀናት ነው").reason == "list-marker"


def test_latin_punctuation_salad_is_flagged():
    assert inspect_text("2. . , , . 3. . . , . . , , , .").reason in ("non-ethiopic", "list-marker")


def test_word_loop_is_flagged():
    assert inspect_text("ቀናት ቀናት ቀናት ስልሳ ነው").reason == "word-loop"


def test_near_empty_is_flagged():
    assert inspect_text(".").action == FLAG


def test_short_legitimate_answer_is_not_penalized_for_latin_punctuation():
    """The non-ethiopic rule is length-gated - a short, real answer with ordinary punctuation
    must not be flagged just because the dense character count is small."""
    assert inspect_text("አዎ ነው!!").action == ACCEPT


def test_exact_repeat_is_a_near_duplicate():
    assert is_near_duplicate(GOOD_SHORT_ANSWER, GOOD_SHORT_ANSWER)


def test_typo_corrupted_repeat_is_a_near_duplicate():
    """Verbatim shape of a real production failure in the sibling project: the same sentence
    restated with a couple of words corrupted by typos. Word-set Jaccard alone misses this."""
    first = "ችግር የሌለኝ ከሆነ ሌላ ጥያቄ መጠየቅ ትችላለህ።"
    second = "ችግር የሌሌኝ ከሆነ ሌላ ጥቄ መጠየቅ ትቻለህ።"
    assert is_near_duplicate(first, second)


def test_genuinely_different_answers_are_not_near_duplicates():
    assert not is_near_duplicate(GOOD_SHORT_ANSWER, GOOD_IDK)
    assert not is_near_duplicate(
        "ኢትዮጵያ በአፍሪካ ቀንድ የምትገኝ ሀገር ናት።", "ዋና ከተማዋ አዲስ አበባ ናት።",
    )


def test_normalize_strips_ethiopic_punctuation_not_just_latin():
    """። ፣ ፤ ፧ live inside the Ethiopic Unicode block (U+1360-U+1368) - a naive
    'keep the whole Ethiopic block' filter would keep them and silently break dedup."""
    assert normalize_for_dedup("ሰላም ነው።") == normalize_for_dedup("ሰላም  ነው")
    assert normalize_for_dedup("ሰላም፣ ነው፤") == normalize_for_dedup("ሰላም ነው")
