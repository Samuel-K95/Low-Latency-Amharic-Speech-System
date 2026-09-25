"""Amharic text normalisation for fair WER/CER (punctuation strip + homophone folding)."""
import re, unicodedata

# Letters that are pronounced identically in modern Amharic; fold the whole 7-order row.
_FOLD_ROWS = {0x1210: 0x1200, 0x1280: 0x1200,   # ሐ, ኀ -> ሀ
              0x1220: 0x1230,                   # ሠ -> ሰ
              0x12D0: 0x12A0,                   # ዐ -> አ
              0x1340: 0x1338}                   # ፀ -> ጸ
_FOLD = {chr(src + i): chr(dst + i) for src, dst in _FOLD_ROWS.items() for i in range(7)}
_PUNCT = re.compile(r"[፡-፨!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~«»“”‘’…]")

def normalize(text: str, fold: bool = True) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = _PUNCT.sub(" ", text)
    if fold:
        text = "".join(_FOLD.get(ch, ch) for ch in text)
    return re.sub(r"\s+", " ", text).strip()
