"""
Unit and integration tests for normalization pipeline and fidel folding.
"""
import pytest
from src.normalize import normalize_query, normalize_doc_text
from src.fidel_fold import fold

def test_query_and_doc_normalization_identity():
    """
    Test that normalize_query(x) == normalize_doc_text(x) across diverse sample inputs.
    """
    sample_strings = [
        "አዋጅ ቁጥር ፩ሺ፩፻፶፮/፪ሺ፲፩ ዓ.ም",
        "አንቀጽ ፴፪ (፩) (ለ)",
        "በዚህ አንቀጽ ንዑስ አንቀጽ (፩) የተመለከተው የሥራ ሰዓት ፵፰ ሰዓት ነው",
        "የትርፍ ሰዓት ክፍያ ተመን 30% ወይም 1.75 ተባዝቶ ይሰጣል",
        "ማንኛውም ሠራተኛ የሙከራ ጊዜው ከ፷ የሥራ ቀናት መብለጥ የለበትም",
        "ደመወዝ በወር ብር 5,000 ይከፈለዋል",
        "የሠራተኛው የዓመት ደመወዝ በ፭ ተባዝቶ የሚገኘው የጉዳት ካሣ",
        "ስለ ኅብረት ስምምነት አፈፃፀም እና የሥራ ክርክር ችሎት",
        "የሴት ሠራተኞች የወሊድ ፈቃድ ፺ ቀናት ነው",
        "ነሐሴ ፴ ቀን ፪ሺ፲፩ ዓ.ም በአዲስ አበባ ታትሞ የወጣ"
    ]
    
    assert len(sample_strings) == 10
    for s in sample_strings:
        q_norm = normalize_query(s)
        d_norm = normalize_doc_text(s)
        assert q_norm == d_norm, f"Mismatch for '{s}': '{q_norm}' != '{d_norm}'"

def test_fidel_folding_homophones():
    """
    Test that spellings differing only by homophones yield identical normalized strings.
    """
    # ሠ vs ሰ
    assert normalize_query("ሥራ") == normalize_query("ስራ")
    assert normalize_query("የሠላሳ ቀን ማስጠንቀቂያ") == normalize_query("የሰላሳ ቀን ማስጠንቀቂያ")
    
    # ሀ vs ሐ vs ኀ
    assert normalize_query("ኃላፊነት") == normalize_query("ሃላፊነት") == normalize_query("ሓላፊነት")
    assert normalize_query("ሕግ") == normalize_query("ህግ") == normalize_query("ኅግ")
    assert normalize_query("ምክክርና ኅብረት") == normalize_query("ምክክርና ህብረት") == normalize_query("ምክክርና ሕብረት")
    
    # አ vs ዐ
    assert normalize_query("ዓመተ ምሕረት") == normalize_query("አመተ ምህረት")
    assert normalize_query("ዕረፍት") == normalize_query("እረፍት")
    assert normalize_query("አዋጅ") == normalize_query("ዐዋጅ")
    
    # ጸ vs ፀ
    assert normalize_query("ፀሐይ") == normalize_query("ጸሀይ")
    assert normalize_query("ተፈጻሚነት") == normalize_query("ተፈፃሚነት")

def test_complex_sentence_normalization():
    """
    Test complex legal phrase with cross-references, Ge'ez numerals, fractions, and abbreviations.
    """
    raw_query = 'በአዋጅ ቁጥር ፩ሺ፩፻፶፮/፪ሺ፲፩ አንቀጽ ፳፯ (፩) (ሀ) መሠረት "የሥራ ሰዓት" አለማክበር'
    norm = normalize_query(raw_query)
    
    # Should contain words for numbers, no quotes, no parens, and folded letters
    assert "፩ሺ፩፻፶፮" not in norm
    assert fold("አንድ ሺህ አንድ መቶ ሃምሳ ስድስት ከሁለት ሺህ አስራ አንድ") in norm
    assert fold("አንቀጽ ሃያ ሰባት ንዑስ አንቀጽ አንድ ፊደል ሀ") in norm
    assert '"' not in norm
    assert '(' not in norm
    assert ')' not in norm
