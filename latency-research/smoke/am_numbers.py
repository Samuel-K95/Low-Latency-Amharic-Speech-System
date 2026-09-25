"""E-TTS-3 (first cut): Amharic text normaliser for TTS.

MMS-TTS romanises its input and has no digits in its vocabulary, so "15" or "2019"
are silently dropped. This spells numbers out in Amharic words and expands a few
common abbreviations. Cost: well under 1 ms per sentence.
"""
import re

ONES = ["", "አንድ", "ሁለት", "ሶስት", "አራት", "አምስት", "ስድስት", "ሰባት", "ስምንት", "ዘጠኝ"]
TENS = ["", "አስር", "ሃያ", "ሰላሳ", "አርባ", "ሃምሳ", "ስልሳ", "ሰባ", "ሰማንያ", "ዘጠና"]
SCALES = [(10**9, "ቢሊዮን"), (10**6, "ሚሊዮን"), (1000, "ሺህ")]
ABBREVIATIONS = {"ዓ.ም.": "ዓመተ ምሕረት", "ዓ.ም": "ዓመተ ምሕረት", "ዓ.ዓ.": "ዓመተ ዓለም", "ዶ/ር": "ዶክተር",
                 "ት/ቤት": "ትምህርት ቤት", "ኪ.ሜ.": "ኪሎ ሜትር", "ኪ.ሜ": "ኪሎ ሜትር", "%": " በመቶ"}


def _below_100(n: int) -> str:
    tens, ones = divmod(n, 10)
    if tens == 0:
        return ONES[ones]
    if tens == 1:
        return "አስር" if ones == 0 else "አስራ " + ONES[ones]
    return TENS[tens] + ("" if ones == 0 else " " + ONES[ones])


def _below_1000(n: int) -> str:
    hundreds, rest = divmod(n, 100)
    parts = [ONES[hundreds] + " መቶ"] if hundreds else []
    if rest:
        parts.append(_below_100(rest))
    return " ".join(parts)


def int_to_amharic(n: int) -> str:
    if n == 0:
        return "ዜሮ"
    parts = []
    for value, word in SCALES:
        count, n = divmod(n, value)
        if count:
            parts.append(_below_1000(count) + " " + word)
    if n:
        parts.append(_below_1000(n))
    return " ".join(parts)


def _number(match: re.Match) -> str:
    whole, frac = match.group(1).replace(",", ""), match.group(2)
    words = int_to_amharic(int(whole))
    if frac:
        words += " ነጥብ " + " ".join(int_to_amharic(int(d)) for d in frac)
    return words


def normalize_for_tts(text: str) -> str:
    for abbr, full in ABBREVIATIONS.items():
        text = text.replace(abbr, full)
    text = re.sub(r"(?<=\d)\s*-\s*(?=\d)", " እስከ ", text)                 # 2-3 -> 2 እስከ 3
    text = re.sub(r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?", _number, text)
    return re.sub(r"\s+", " ", text).strip()


if __name__ == "__main__":
    for s in ["ስብሰባው መስከረም 15 ቀን 2019 ዓ.ም. በ3 ሰዓት ይጀምራል።", "ዋጋው 4,500 ብር ነው።",
              "ቡና 1.5 ኪ.ሜ ርቀት ላይ ነው፣ ከ2-3 ሰዓት ይወስዳል።", "5% ቅናሽ", "100 እና 1000 እና 21"]:
        print(s, "->", normalize_for_tts(s))
