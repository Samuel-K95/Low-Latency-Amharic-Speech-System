#!/usr/bin/env python3
"""
Step 5: Convert written text with numerals and structural markers to spoken Amharic text.
Code in src/spoken.py.
For every sub-article, generates text_spoken while preserving text_clean intact.
"""
import sys
import re
import json
from pathlib import Path
from src.ethiopic_numbers import geez_to_int, int_to_words, to_words_arabic

# Ge'ez numeral sequence that must contain at least one Ge'ez digit (፩-፼)
ETH_NUM_STR = r'(?=[፩-፼ሺ]*[፩-፼])[፩-፼ሺ]+'
ETH_LETTER = r'[ሀ-ፐ\u1200-\u135A]'

def convert_to_spoken(text, log_stats=None):
    """
    Convert written Amharic text to spoken form following Prompt 5 rules:
    1. Remove structural markers and lettered item markers; join items with ፤.
    2. Convert cross-references (አንቀጽ, ንዑስ አንቀጽ, ፊደል).
    3. Convert remaining Ge'ez numerals and fractions/proclamation numbers (with ከ).
    4. Convert dates.
    5. Convert Arabic digits via to_words_arabic.
    6. Expand abbreviations (ዓ.ም -> ዓመተ ምሕረት) while keeping acronyms like ኤች.አይ.ቪ.
    7. Remove quotation marks and parentheses, normalize whitespace.
    """
    if not text:
        return ""
    
    num_converted = 0
    
    # -------------------------------------------------------------------------
    # 1. Structural markers removal at start of sub-articles / lettered items
    # -------------------------------------------------------------------------
    lines = text.splitlines()
    processed_lines = []
    for l in lines:
        l_str = l.strip()
        # Remove sub-article marker if present
        l_str = re.sub(rf'^{ETH_NUM_STR}\s*[\/\)]\s*', '', l_str)
        # Remove letter marker if present (e.g. "ሀ) ", "ለ) ")
        m_let = re.match(rf'^({ETH_LETTER})\s*[\/\)]\s*(.*)', l_str)
        if m_let:
            item_body = m_let.group(2)
            processed_lines.append(item_body)
        else:
            processed_lines.append(l_str)
            
    # Join items with ፤ if not already ending in punctuation
    joined_lines = []
    for line in processed_lines:
        if not line:
            continue
        if joined_lines:
            prev = joined_lines[-1]
            if not re.search(r'[።፡፡፤፣፡]$', prev):
                joined_lines[-1] = prev + "፤"
        joined_lines.append(line)
        
    s = ' '.join(joined_lines)
    
    # -------------------------------------------------------------------------
    # 2. Cross-references
    # -------------------------------------------------------------------------
    # "በአንቀጽ ፳፯(ለ-ተ)" or "አንቀጽ ፳፯ (ለ - ተ)" -> "በአንቀጽ ሃያ ሰባት ከፊደል ለ እስከ ተ"
    def repl_art_letter_range(m):
        nonlocal num_converted
        prefix = m.group(1) or ""
        art_w = m.group(2)
        art_g, l1, l2 = m.group(3), m.group(4), m.group(5)
        art_val = geez_to_int(art_g)
        if art_val is not None:
            num_converted += 1
            return f"{prefix}{art_w} {int_to_words(art_val)} ከፊደል {l1} እስከ {l2}"
        return m.group(0)
    s = re.sub(rf'\b(በ|ከ|ለ|የ)?(አንቀጽ|አንቀፅ)\s*({ETH_NUM_STR})\s*\(\s*({ETH_LETTER})\s*[-–—]\s*({ETH_LETTER})\s*\)', repl_art_letter_range, s)

    # "አንቀጽ ፴፪ (፩) (ለ)" or "አንቀጽ ፴፪(፩)(ለ)" or "በአንቀጽ ፴፪(፩)(ለ)"
    def repl_art_sub_let(m):
        nonlocal num_converted
        prefix = m.group(1) or ""
        art_w = m.group(2)
        art_g, sub_g, let = m.group(3), m.group(4), m.group(5)
        art_val = geez_to_int(art_g)
        sub_val = geez_to_int(sub_g)
        if art_val is not None and sub_val is not None:
            num_converted += 2
            return f"{prefix}{art_w} {int_to_words(art_val)} ንዑስ አንቀጽ {int_to_words(sub_val)} ፊደል {let}"
        return m.group(0)
    s = re.sub(rf'\b(በ|ከ|ለ|የ)?(አንቀጽ|አንቀፅ)\s*({ETH_NUM_STR})\s*\(\s*({ETH_NUM_STR})\s*\)\s*\(\s*({ETH_LETTER})\s*\)', repl_art_sub_let, s)
    
    # "አንቀጽ ፴፪ (፩)" or "አንቀጽ ፴፪(፩)"
    def repl_art_sub(m):
        nonlocal num_converted
        prefix = m.group(1) or ""
        art_w = m.group(2)
        art_g, sub_g = m.group(3), m.group(4)
        art_val = geez_to_int(art_g)
        sub_val = geez_to_int(sub_g)
        if art_val is not None and sub_val is not None:
            num_converted += 2
            return f"{prefix}{art_w} {int_to_words(art_val)} ንዑስ አንቀጽ {int_to_words(sub_val)}"
        return m.group(0)
    s = re.sub(rf'\b(በ|ከ|ለ|የ)?(አንቀጽ|አንቀፅ)\s*({ETH_NUM_STR})\s*\(\s*({ETH_NUM_STR})\s*\)', repl_art_sub, s)

    # "ንዑስ አንቀጽ (፬)(ሀ)" or "ንዑስ አንቀጽ (፬) (ሀ)" -> "ንዑስ አንቀጽ አራት ፊደል ሀ"
    def repl_sub_let(m):
        nonlocal num_converted
        prefix = m.group(1)
        sub_str = m.group(2)
        let = m.group(3)
        sub_val = geez_to_int(sub_str)
        if sub_val is not None:
            num_converted += 1
            return f"{prefix} {int_to_words(sub_val)} ፊደል {let}"
        return m.group(0)
    s = re.sub(rf'\b(ንዑስ\s+አንቀጽ|ንኡስ\s+አንቀጽ|ንዑስ\s+አንቀጾች|ንኡስ\s+አንቀጾች)\s*\(\s*({ETH_NUM_STR})\s*\)\s*\(\s*({ETH_LETTER})\s*\)', repl_sub_let, s)

    # "ንዑስ አንቀጽ (፩)" or "ንዑስ አንቀጽ ፩" or "ንኡስ አንቀጽ (፩)"
    def repl_sub_only(m):
        nonlocal num_converted
        prefix = m.group(1)
        sub_str = m.group(2) or m.group(3)
        sub_val = geez_to_int(sub_str)
        if sub_val is not None:
            num_converted += 1
            return f"{prefix} {int_to_words(sub_val)}"
        return m.group(0)
    s = re.sub(rf'\b(ንዑስ\s+አንቀጽ|ንኡስ\s+አንቀጽ|ንዑስ\s+አንቀጾች|ንኡስ\s+አንቀጾች)\s*(?:\(\s*({ETH_NUM_STR})\s*\)|({ETH_NUM_STR}))', repl_sub_only, s)

    # "፳፯(ለ-ተ)" standalone number + letter range
    def repl_num_letter_range(m):
        nonlocal num_converted
        num_str, l1, l2 = m.group(1), m.group(2), m.group(3)
        val = geez_to_int(num_str)
        if val is not None:
            num_converted += 1
            return f"{int_to_words(val)} ከፊደል {l1} እስከ {l2}"
        return m.group(0)
    s = re.sub(rf'\b({ETH_NUM_STR})\s*\(\s*({ETH_LETTER})\s*[-–—]\s*({ETH_LETTER})\s*\)', repl_num_letter_range, s)

    # "፳፱ (፩)" or "፳፱(፫)" standalone number + subarticle in parens
    def repl_standalone_num_sub(m):
        nonlocal num_converted
        n1_str, n2_str = m.group(1), m.group(2)
        v1 = geez_to_int(n1_str)
        v2 = geez_to_int(n2_str)
        if v1 is not None and v2 is not None:
            num_converted += 2
            return f"{int_to_words(v1)} ንዑስ አንቀጽ {int_to_words(v2)}"
        return m.group(0)
    s = re.sub(rf'\b({ETH_NUM_STR})\s*\(\s*({ETH_NUM_STR})\s*\)', repl_standalone_num_sub, s)

    # General pattern: "(፬)(ሀ)" or "(፬) (ሀ)" -> "<number words> ፊደል <letter>"
    def repl_num_let(m):
        nonlocal num_converted
        num_str = m.group(1)
        let = m.group(2)
        val = geez_to_int(num_str)
        if val is not None:
            num_converted += 1
            return f"{int_to_words(val)} ፊደል {let}"
        return m.group(0)
    s = re.sub(rf'\(\s*({ETH_NUM_STR})\s*\)\s*\(\s*({ETH_LETTER})\s*\)', repl_num_let, s)

    # Standalone letter in parens after "ከፊደል (ሀ) እስከ (ሐ)" or "(ሀ)"
    s = re.sub(rf'ፊደል\s*\(\s*({ETH_LETTER})\s*\)', r'ፊደል \1', s)
    
    # "አንቀጽ ፲" or "በአንቀጽ ፲"
    def repl_art_only(m):
        nonlocal num_converted
        prefix = m.group(1) or ""
        art_w = m.group(2)
        art_g = m.group(3)
        art_val = geez_to_int(art_g)
        if art_val is not None:
            num_converted += 1
            return f"{prefix}{art_w} {int_to_words(art_val)}"
        return m.group(0)
    s = re.sub(rf'\b(በ|ከ|ለ|የ)?(አንቀጽ|አንቀፅ)\s+({ETH_NUM_STR})', repl_art_only, s)
    
    # -------------------------------------------------------------------------
    # 3. Proclamation / fraction numbers: "፩ሺ፩፻፶፮/፪ሺ፲፩" -> "አንድ ሺህ አንድ መቶ ሃምሳ ስድስት ከሁለት ሺህ አስራ አንድ"
    # -------------------------------------------------------------------------
    def repl_fraction(m):
        nonlocal num_converted
        n1_g, n2_g = m.group(1), m.group(2)
        v1 = geez_to_int(n1_g)
        v2 = geez_to_int(n2_g)
        if v1 is not None and v2 is not None:
            num_converted += 2
            return f"{int_to_words(v1)} ከ{int_to_words(v2)}"
        return m.group(0)
    s = re.sub(rf'({ETH_NUM_STR})\s*\/\s*({ETH_NUM_STR})', repl_fraction, s)
    
    # -------------------------------------------------------------------------
    # 4. Standalone Ge'ez numerals, quantities, amounts, dates
    # -------------------------------------------------------------------------
    def repl_geez_standalone(m):
        nonlocal num_converted
        g_str = m.group(0)
        val = geez_to_int(g_str)
        if val is not None:
            num_converted += 1
            return int_to_words(val)
        return g_str  # Malformed like ፬፵፱ is left unconverted
    s = re.sub(rf'{ETH_NUM_STR}', repl_geez_standalone, s)
    
    # -------------------------------------------------------------------------
    # 5. Arabic numbers via to_words_arabic
    # -------------------------------------------------------------------------
    def repl_arabic(m):
        nonlocal num_converted
        a_str = m.group(0)
        w = to_words_arabic(a_str)
        if w:
            num_converted += 1
            return w
        return a_str
    s = re.sub(r'\d+(?:,\d+)*(?:\.\d+)?%?', repl_arabic, s)
    
    # -------------------------------------------------------------------------
    # 6. Abbreviations
    # -------------------------------------------------------------------------
    s = s.replace('ዓ.ም.', 'ዓመተ ምሕረት')
    s = s.replace('ዓ.ም', 'ዓመተ ምሕረት')
    
    # -------------------------------------------------------------------------
    # 7. Remove quotation marks and parentheses, keep punctuation
    # -------------------------------------------------------------------------
    s = re.sub(r'["“”\'‘’\(\)\[\]]', '', s)
    s = re.sub(r'[ \t]+', ' ', s)
    s = s.strip()
    
    if log_stats is not None:
        log_stats['num_converted'] = log_stats.get('num_converted', 0) + num_converted
        
    return s

def process_articles(articles):
    """
    Process list of article dictionaries, adding text_clean and text_spoken to each sub-article.
    """
    total_converted = 0
    leftovers = []
    
    for art in articles:
        for sa in art['sub_articles']:
            if 'text_clean' not in sa:
                sa['text_clean'] = sa['text']
            
            stats = {'num_converted': 0}
            sa['text_spoken'] = convert_to_spoken(sa['text_clean'], log_stats=stats)
            total_converted += stats['num_converted']
            
            g_left = re.findall(r'[\u1369-\u137C]', sa['text_spoken'])
            a_left = re.findall(r'[0-9]', sa['text_spoken'])
            if g_left or a_left:
                leftovers.append({
                    'article_number': art['article_number'],
                    'marker': sa['marker'],
                    'geez_leftovers': g_left,
                    'arabic_leftovers': a_left,
                    'snippet': sa['text_spoken'][:100]
                })
                
    return articles, total_converted, leftovers

def run():
    base_dir = Path(__file__).resolve().parent.parent
    debug_path = base_dir / 'data' / 'structure_debug.json'
    report_path = base_dir / 'data' / 'cleaning_report.txt'
    
    if not debug_path.exists():
        print(f"Error: structure_debug.json not found at {debug_path}")
        sys.exit(1)
        
    articles = json.loads(debug_path.read_text(encoding='utf-8'))
    articles, total_converted, leftovers = process_articles(articles)
    
    debug_path.write_text(json.dumps(articles, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Processed {len(articles)} articles with spoken text. Total numerals converted: {total_converted}")
    
    write_stage5_report(report_path, articles, total_converted, leftovers)

def write_stage5_report(report_path, articles, total_converted, leftovers):
    import datetime
    
    with open(report_path, 'a', encoding='utf-8') as f:
        f.write("\n=== STAGE 5: SPOKEN TEXT ===\n")
        f.write(f"Date: {datetime.datetime.now().isoformat()}\n")
        f.write(f"Total numerals and numbers converted: {total_converted}\n")
        f.write(f"Leftover unconvertible numerals: {len(leftovers)}\n")
        for lo in leftovers:
            f.write(f"  - Article {lo['article_number']} (marker {lo['marker']}): Ge'ez={lo['geez_leftovers']}, Arabic={lo['arabic_leftovers']}\n")
            f.write(f"    Snippet: {lo['snippet']}\n")
            
        f.write("\n--- SAMPLES BEFORE AND AFTER (Articles 11, 35, 40, 68, 109, 160) ---\n")
        for target in [11, 35, 40, 68, 109, 160]:
            art = next(a for a in articles if a['article_number'] == target)
            f.write(f"\n=======================================================\n")
            f.write(f"Article {target}: {art['title']}\n")
            for j, sa in enumerate(art['sub_articles'], 1):
                f.write(f"\n[Sub-article {j} | Marker: {sa['marker']}]\n")
                f.write(f"BEFORE (Ge'ez clean):\n{sa['text_clean']}\n")
                f.write(f"AFTER (Spoken):\n{sa['text_spoken']}\n")

if __name__ == '__main__':
    run()
