#!/usr/bin/env python3
"""
Step 2: Remove noise and join lines -> clean_raw.txt
Reads data/raw.txt and outputs data/clean_raw.txt.
Logs all decisions and statistics to data/cleaning_report.txt.
"""
import sys
import re
import random
from pathlib import Path
from collections import Counter

def run():
    base_dir = Path(__file__).resolve().parent.parent
    raw_path = base_dir / 'data' / 'raw.txt'
    clean_path = base_dir / 'data' / 'clean_raw.txt'
    report_path = base_dir / 'data' / 'cleaning_report.txt'
    
    if not raw_path.exists():
        print(f"Error: raw.txt not found at {raw_path}")
        sys.exit(1)
    
    raw = raw_path.read_text(encoding='utf-8')
    cleaned_paragraphs, report_data = process_cleaning(raw)
    
    clean_text = '\n'.join(cleaned_paragraphs)
    clean_path.write_text(clean_text, encoding='utf-8')
    print(f"Wrote {len(cleaned_paragraphs)} paragraphs ({len(clean_text)} chars) to {clean_path}")
    
    write_stage2_report(report_path, report_data, cleaned_paragraphs)

def process_cleaning(raw):
    raw_lines = raw.splitlines()
    total_raw_lines = len(raw_lines)
    
    # -------------------------------------------------------------------------
    # Step A: Front matter removal
    # -------------------------------------------------------------------------
    title_idx = -1
    for i, l in enumerate(raw_lines):
        if l.strip() == "አዋጅ ቁጥር ፩ሺ፩፻፶፮/፪ሺ፲፩":
            title_idx = i
            break
    
    if title_idx == -1:
        # Fallback if exact line not found
        title_idx = 0
        front_matter_removed = []
        kept_after_front_matter = raw_lines
    else:
        front_matter_removed = raw_lines[:title_idx]
        kept_after_front_matter = ["===PAGE 1==="] + raw_lines[title_idx:]
    
    # -------------------------------------------------------------------------
    # Step A: Signature block removal
    # -------------------------------------------------------------------------
    end_idx = -1
    for i in range(len(kept_after_front_matter) - 1, -1, -1):
        if "ከወጣበት ቀን ጀምሮ የጸና ይሆናል።" in kept_after_front_matter[i]:
            end_idx = i
            break
    
    if end_idx == -1:
        signature_block_removed = []
        working_lines = kept_after_front_matter
    else:
        signature_block_removed = kept_after_front_matter[end_idx+1:]
        working_lines = kept_after_front_matter[:end_idx+1]
    
    # -------------------------------------------------------------------------
    # Step A & B: Filter lines (headers, page numbers, zero-Ethiopic, blank, mixed)
    # -------------------------------------------------------------------------
    dropped_headers = []
    dropped_page_numbers = []
    dropped_zero_eth = []
    dropped_blanks = []
    kept_lines = []
    mixed_lines_for_review = []
    
    geez_page_pattern = re.compile(r'^፲፩ሺ[፩-፼]+$')
    
    for idx, line in enumerate(working_lines, 1):
        s = line.strip()
        if s.startswith('===PAGE'):
            kept_lines.append(s)
            continue
        if not s:
            dropped_blanks.append(line)
            continue
        # Page headers
        if any(k in s for k in ['Federal Negarit Gazette', 'gA Ød', 'ነU¶T']) or re.search(r'…page\s*\d+', s):
            dropped_headers.append(line)
            continue
        # Page numbers
        if geez_page_pattern.match(s):
            dropped_page_numbers.append(line)
            continue
        # English or garbage lines (zero Ethiopic chars)
        has_eth = bool(re.search(r'[\u1200-\u137F]', s))
        if not has_eth:
            dropped_zero_eth.append(line)
            continue
        
        # Step B: Mixed lines
        has_lat = bool(re.search(r'[a-zA-Z]', s))
        if has_eth and has_lat:
            mixed_lines_for_review.append((idx, line))
        
        kept_lines.append(s)
    
    # -------------------------------------------------------------------------
    # Step C: Join wrapped lines & word frequency gluing
    # -------------------------------------------------------------------------
    tokens = re.findall(r'[\u1200-\u135A]+', raw)
    word_freq = Counter(tokens)
    
    # Compute typical line length in Amharic column (80th percentile)
    non_marker_lengths = [
        len(l.strip()) for l in kept_lines 
        if not l.startswith('===PAGE') and len(l.strip()) > 0
    ]
    if non_marker_lengths:
        sorted_lens = sorted(non_marker_lengths)
        pct80_idx = int(len(sorted_lens) * 0.80)
        full_line_len = sorted_lens[pct80_idx]
    else:
        full_line_len = 23
    full_width_threshold = full_line_len * 0.90  # 90% of 80th percentile (~20.7 chars)
    
    article_heading_pattern = re.compile(r'^[፩-፼ሺ]+\s*\.')
    sub_article_start = re.compile(r'^[፩-፼]+\s*[\/\)](?!\s*[፩-፼ሺ])')
    letter_start = re.compile(r'^[ሀ-ፐ\u1200-\u135A]\s*[\/\)](?!\s*[፩-፼ሺ])')
    struct_start = re.compile(r'^(ክፍል|ምዕራፍ|ዘርፍ|ንዑስ\s*ዘርፍ|አዋጅ ቁጥር|ስለ አሠሪና|በኢትዮጵያ ፌደራላዊ)')
    preamble_start = re.compile(r'^(አሠሪና ሠራተኛ የጋራ ራዕይ|ሠራተኞች እና አሠሪዎች|መሠረታዊ የሥራ ላይ)')
    
    def clean_zero_width_and_spaces(text):
        text = re.sub(r'[\u200b\ufeff\u200e\u200f\u202a-\u202e]', '', text)
        text = text.replace('\u00a0', ' ')
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()
    
    def should_glue(w1, w2):
        glued = w1 + w2
        cnt_glued = word_freq.get(glued, 0)
        cnt_w1 = word_freq.get(w1, 0)
        cnt_w2 = word_freq.get(w2, 0)
        if cnt_glued > 0 and (cnt_w1 <= 2 or cnt_w2 <= 2):
            return True, glued, f"glued word '{glued}' exists ({cnt_glued}x) in corpus while parts are rare (w1={cnt_w1}, w2={cnt_w2})"
        if w1 == 'እንዲ' and cnt_w1 <= 2 and cnt_w2 <= 2:
            return True, glued, f"prefix 'እንዲ' split from verb '{w2}'"
        if cnt_w2 == 1 and w2 in ['ነት', 'ዎች', 'ዊ', 'ያዊ']:
            return True, glued, f"suffix '{w2}' split from root '{w1}'"
        return False, None, None
    
    glue_decisions = []
    
    def join_lines(p_lines):
        if not p_lines:
            return ""
        result = p_lines[0]
        for nxt in p_lines[1:]:
            w1_match = re.search(r'([\u1200-\u135A]+)[^\u1200-\u135A]*$', result)
            w2_match = re.search(r'^[^\u1200-\u135A]*([\u1200-\u135A]+)', nxt)
            glue = False
            if w1_match and w2_match:
                w1 = w1_match.group(1)
                w2 = w2_match.group(1)
                do_glue, glued_word, reason = should_glue(w1, w2)
                if do_glue:
                    glue = True
                    glue_decisions.append({
                        'w1': w1, 'w2': w2, 'glued': glued_word, 'reason': reason
                    })
            if glue:
                result = result + nxt
            else:
                result = result + " " + nxt
        return clean_zero_width_and_spaces(result)
    
    # Process lines: handle article headings with wrapping rule vs body paragraphs
    # Filter out page markers first
    content_lines = [l for l in kept_lines if not l.startswith('===PAGE')]
    
    def is_structural_boundary(l_text):
        s_val = l_text.strip()
        return bool(
            article_heading_pattern.match(s_val) or
            sub_article_start.match(s_val) or
            letter_start.match(s_val) or
            struct_start.match(s_val) or
            preamble_start.match(s_val)
        )
    
    paragraphs = []
    idx = 0
    while idx < len(content_lines):
        line = content_lines[idx]
        s_line = line.strip()
        
        # Check if line is an article heading
        if article_heading_pattern.match(s_line):
            heading_chunk = [s_line]
            # Rule: Heading wrapping check
            # Only join with next line if heading line length >= full_width_threshold (90% of 80th percentile)
            # AND next line is not a structural marker, has no sentence mark, and total title words <= 12
            if len(s_line) >= full_width_threshold and (idx + 1) < len(content_lines):
                next_l = content_lines[idx + 1].strip()
                if not is_structural_boundary(next_l):
                    has_sent_mark = bool(re.search(r'[።፡፡]', next_l))
                    cand_title = s_line + " " + next_l
                    cand_words = len(cand_title.split())
                    if not has_sent_mark and cand_words <= 12:
                        heading_chunk.append(next_l)
                        idx += 1
            
            p_text = join_lines(heading_chunk)
            if p_text:
                paragraphs.append(p_text)
            idx += 1
            continue
            
        # Non-heading lines: group until next structural boundary
        body_chunk = [s_line]
        idx += 1
        while idx < len(content_lines):
            nxt = content_lines[idx].strip()
            if is_structural_boundary(nxt):
                break
            body_chunk.append(nxt)
            idx += 1
            
        p_text = join_lines(body_chunk)
        if p_text:
            paragraphs.append(p_text)
    
    report_data = {
        'total_raw_lines': total_raw_lines,
        'front_matter_removed': front_matter_removed,
        'signature_block_removed': signature_block_removed,
        'dropped_headers': dropped_headers,
        'dropped_page_numbers': dropped_page_numbers,
        'dropped_zero_eth': dropped_zero_eth,
        'dropped_blanks': dropped_blanks,
        'mixed_lines_for_review': mixed_lines_for_review,
        'glue_decisions': glue_decisions,
        'total_paragraphs': len(paragraphs)
    }
    
    return paragraphs, report_data

def write_stage2_report(report_path, r, paragraphs):
    import datetime
    
    with open(report_path, 'a', encoding='utf-8') as f:
        f.write("\n=== STAGE 2: CLEAN ===\n")
        f.write(f"Date: {datetime.datetime.now().isoformat()}\n")
        f.write(f"Lines before cleaning (data/raw.txt): {r['total_raw_lines']}\n")
        f.write(f"Paragraphs after cleaning (data/clean_raw.txt): {r['total_paragraphs']}\n\n")
        
        f.write("--- DROPPED LINES SUMMARY ---\n")
        f.write(f"1. Front Matter lines dropped: {len(r['front_matter_removed'])}\n")
        f.write("   Content removed:\n")
        for l in r['front_matter_removed']:
            f.write(f"     {repr(l)}\n")
        
        f.write(f"\n2. Signature Block lines dropped: {len(r['signature_block_removed'])}\n")
        f.write("   Content removed:\n")
        for l in r['signature_block_removed']:
            f.write(f"     {repr(l)}\n")
        
        f.write(f"\n3. Page Headers dropped: {len(r['dropped_headers'])}\n")
        f.write("   Sample up to 10 examples:\n")
        for l in r['dropped_headers'][:10]:
            f.write(f"     {repr(l)}\n")
            
        f.write(f"\n4. Page Numbers (Ge'ez numerals) dropped: {len(r['dropped_page_numbers'])}\n")
        f.write("   Sample up to 10 examples:\n")
        for l in r['dropped_page_numbers'][:10]:
            f.write(f"     {repr(l)}\n")
            
        f.write(f"\n5. English/Garbage lines (zero Ethiopic characters) dropped: {len(r['dropped_zero_eth'])}\n")
        f.write("   Sample up to 10 examples:\n")
        for l in r['dropped_zero_eth'][:10]:
            f.write(f"     {repr(l)}\n")
            
        f.write(f"\n6. Blank lines dropped: {len(r['dropped_blanks'])}\n")
        
        f.write("\n--- MIXED LINES FOR REVIEW ---\n")
        if r['mixed_lines_for_review']:
            for line_no, l in r['mixed_lines_for_review']:
                f.write(f"   Line {line_no}: {repr(l)}\n")
        else:
            f.write("   None (all non-header lines contain purely Ethiopic characters and punctuation)\n")
        
        f.write(f"\n--- GLUE DECISIONS ({len(r['glue_decisions'])}) ---\n")
        for g in r['glue_decisions']:
            f.write(f"   '{g['w1']}' + '{g['w2']}' -> '{g['glued']}' | Reason: {g['reason']}\n")
        
        f.write("\n--- 25 RANDOM SAMPLE PARAGRAPHS ---\n")
        rng = random.Random(42)  # Deterministic seed for reproducible reporting
        sample_indices = sorted(rng.sample(range(len(paragraphs)), min(25, len(paragraphs))))
        for idx in sample_indices:
            f.write(f"   [{idx+1:>4}] {paragraphs[idx]}\n")

if __name__ == '__main__':
    run()
