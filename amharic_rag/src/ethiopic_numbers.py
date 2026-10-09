"""
Ethiopic/Ge'ez number conversion utilities.
Supports parsing Ge'ez numerals to int, converting integers to Amharic spoken words,
converting Arabic numerals / percentages / decimals to Amharic words, and int to Ge'ez.
"""
import re

GEEZ_UNITS = {'፩': 1, '፪': 2, '፫': 3, '፬': 4, '፭': 5, '፮': 6, '፯': 7, '፰': 8, '፱': 9}
GEEZ_TENS = {'፲': 10, '፳': 20, '፴': 30, '፵': 40, '፶': 50, '፷': 60, '፸': 70, '፹': 80, '፺': 90}

UNITS_WORDS = {
    1: 'አንድ', 2: 'ሁለት', 3: 'ሦስት', 4: 'አራት', 5: 'አምስት',
    6: 'ስድስት', 7: 'ሰባት', 8: 'ስምንት', 9: 'ዘጠኝ'
}

TENS_WORDS = {
    10: 'አስር', 20: 'ሃያ', 30: 'ሠላሳ', 40: 'አርባ', 50: 'ሃምሳ',
    60: 'ስድሳ', 70: 'ሰባ', 80: 'ሰማንያ', 90: 'ዘጠና'
}

GEEZ_DIGITS_LIST = ['', '፩', '፪', '፫', '፬', '፭', '፮', '፯', '፰', '፱']
GEEZ_TENS_LIST = ['', '፲', '፳', '፴', '፵', '፶', '፷', '፸', '፹', '፺']

def parse_part(s):
    """
    Parse a sub-part of Ge'ez numeral without thousands marker ሺ.
    Within a part: keep current = sum of unit and tens symbols;
    when ፻ appears, total += (current or 1) * 100 and current = 0;
    at the end total += current.
    Enforces valid numeral ordering (tens before units).
    """
    if not s:
        return 0
    total = 0
    current = 0
    last_type = None  # None, 'tens', 'units'
    
    for c in s:
        if c == '፻':
            total += (current if current != 0 else 1) * 100
            current = 0
            last_type = None
        elif c in GEEZ_TENS:
            if last_type in ('units', 'tens'):
                return None  # Malformed: unit before tens or multiple tens in same rank
            current += GEEZ_TENS[c]
            last_type = 'tens'
        elif c in GEEZ_UNITS:
            if last_type == 'units':
                return None  # Malformed: multiple units in same rank
            current += GEEZ_UNITS[c]
            last_type = 'units'
        else:
            return None  # Unknown character
            
    total += current
    return total

def geez_to_int(s):
    """
    Parse a Ge'ez numeral string to integer.
    Uses ሺ as the thousands marker.
    Returns None on malformed input (never raises).
    """
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None
    
    try:
        if 'ሺ' in s:
            parts = s.split('ሺ')
            if len(parts) != 2:
                return None
            left, right = parts
            val_left = parse_part(left) if left else 1
            if val_left is None:
                return None
            val_right = parse_part(right) if right else 0
            if val_right is None:
                return None
            return val_left * 1000 + val_right
        else:
            return parse_part(s)
    except Exception:
        return None

def int_to_words_under_100(n):
    if n == 0:
        return ''
    if n in UNITS_WORDS:
        return UNITS_WORDS[n]
    if n in TENS_WORDS:
        return TENS_WORDS[n]
    if 11 <= n <= 19:
        return 'አስራ ' + UNITS_WORDS[n - 10]
    tens = (n // 10) * 10
    ones = n % 10
    return TENS_WORDS[tens] + ' ' + UNITS_WORDS[ones]

def int_to_words_under_1000(n):
    if n == 0:
        return ''
    if n < 100:
        return int_to_words_under_100(n)
    h = n // 100
    rem = n % 100
    h_str = UNITS_WORDS[h] + ' መቶ'
    if rem > 0:
        return h_str + ' ' + int_to_words_under_100(rem)
    return h_str

def int_to_words(n):
    """
    Convert integer (0 to 999,999) to Amharic spoken words.
    """
    if not isinstance(n, int):
        try:
            n = int(n)
        except Exception:
            return None
    if n < 0 or n > 999999:
        return None
    if n == 0:
        return 'ዜሮ'
    if n < 1000:
        return int_to_words_under_1000(n)
    
    th = n // 1000
    rem = n % 1000
    th_str = int_to_words_under_1000(th) + ' ሺህ'
    if rem > 0:
        return th_str + ' ' + int_to_words_under_1000(rem)
    return th_str

def to_words_arabic(s):
    """
    Convert Arabic digit string to Amharic words.
    Handles:
      - Integers with optional commas ("5,000" -> "አምስት ሺህ")
      - Percentages ("30%" -> "ሠላሳ በመቶ")
      - Decimals ("1.75" -> "አንድ ነጥብ ሰባ አምስት")
    """
    if not isinstance(s, str):
        s = str(s)
    s = s.strip().replace(',', '')
    if not s:
        return None
    
    # Percentage
    if s.endswith('%'):
        num_part = s[:-1].strip()
        val = to_words_arabic(num_part)
        return f"{val} በመቶ" if val else None
    
    # Decimal
    if '.' in s:
        parts = s.split('.')
        if len(parts) == 2:
            try:
                int_val = int(parts[0]) if parts[0] else 0
                int_words = int_to_words(int_val)
                frac_str = parts[1]
                if frac_str:
                    frac_val = int(frac_str)
                    frac_words = int_to_words(frac_val)
                    return f"{int_words} ነጥብ {frac_words}"
                return int_words
            except Exception:
                return None
    
    # Integer
    try:
        val = int(s)
        return int_to_words(val)
    except Exception:
        return None

def int_to_geez_under_100(n):
    if n <= 0:
        return ''
    tens = (n % 100) // 10
    ones = n % 10
    return GEEZ_TENS_LIST[tens] + GEEZ_DIGITS_LIST[ones]

def int_to_geez(n):
    """Convert integer to Ge'ez numeral string."""
    if n <= 0:
        return ''
    if n >= 10000:
        ten_th = n // 10000
        rem = n % 10000
        prefix = int_to_geez(ten_th) if ten_th > 1 else ''
        return prefix + '፼' + (int_to_geez(rem) if rem > 0 else '')
    if n >= 100:
        h = n // 100
        rem = n % 100
        prefix = int_to_geez_under_100(h) if h > 0 else ''
        return prefix + '፻' + (int_to_geez_under_100(rem) if rem > 0 else '')
    return int_to_geez_under_100(n)
