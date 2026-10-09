"""
Fidel (Amharic homophone) folding utilities.
Maps homophonic fidel characters (ሀ/ሐ/ኀ, ሠ/ሰ, አ/ዐ, ጸ/ፀ) to canonical forms.
"""

FOLD_MAP = {
    # ሐ, ኀ -> ሀ
    'ሐ': 'ሀ', 'ሑ': 'ሁ', 'ሒ': 'ሂ', 'ሓ': 'ሃ', 'ሔ': 'ሄ', 'ሕ': 'ህ', 'ሖ': 'ሆ',
    'ኀ': 'ሀ', 'ኁ': 'ሁ', 'ኂ': 'ሂ', 'ኃ': 'ሃ', 'ኄ': 'ሄ', 'ኅ': 'ህ', 'ኆ': 'ሆ',
    'ኋ': 'ኋ', 'ሗ': 'ኋ', 'ኈ': 'ሁ', 'ኊ': 'ሂ', 'ኌ': 'ሄ', 'ኍ': 'ህ',
    
    # ሠ -> ሰ
    'ሠ': 'ሰ', 'ሡ': 'ሱ', 'ሢ': 'ሲ', 'ሣ': 'ሳ', 'ሤ': 'ሴ', 'ሥ': 'ስ', 'ሦ': 'ሶ', 'ሧ': 'ሷ',
    
    # ዐ -> አ
    'ዐ': 'አ', 'ዑ': 'ኡ', 'ዒ': 'ኢ', 'ዓ': 'አ', 'ዔ': 'ኤ', 'ዕ': 'እ', 'ዖ': 'ኦ',
    'ኣ': 'አ',
    
    # ፀ -> ጸ
    'ፀ': 'ጸ', 'ፁ': 'ጹ', 'ፂ': 'ጺ', 'ፃ': 'ጻ', 'ፄ': 'ጼ', 'ፅ': 'ጽ', 'ፆ': 'ጾ', 'ፇ': 'ጿ'
}

TRANSLATION_TABLE = str.maketrans(FOLD_MAP)

def fold(text):
    """
    Apply homophone character folding to the given Amharic string.
    """
    if not text or not isinstance(text, str):
        return '' if text is None else str(text)
    return text.translate(TRANSLATION_TABLE)
