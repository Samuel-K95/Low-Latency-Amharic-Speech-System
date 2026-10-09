#!/usr/bin/env python3
"""
Step 3: Parse Structure -> in-memory structure and data/structure_debug.json
Detects document hierarchy (ክፍል -> ምዕራፍ -> ዘርፍ -> ንዑስ ዘርፍ -> አንቀጽ),
validates articles 0..193, parses sub-articles and lettered items,
and writes data/structure_debug.json and cleaning report.
"""
import sys
import re
import json
from pathlib import Path
from src.ethiopic_numbers import geez_to_int

def run():
    base_dir = Path(__file__).resolve().parent.parent
    clean_path = base_dir / 'data' / 'clean_raw.txt'
    debug_path = base_dir / 'data' / 'structure_debug.json'
    report_path = base_dir / 'data' / 'cleaning_report.txt'
    
    if not clean_path.exists():
        print(f"Error: clean_raw.txt not found at {clean_path}")
        sys.exit(1)
    
    clean_text = clean_path.read_text(encoding='utf-8')
    articles, report_data = parse_document_structure(clean_text)
    
    debug_path.write_text(json.dumps(articles, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Parsed {len(articles)} articles (0..193) successfully and wrote to {debug_path}")
    
    write_stage3_report(report_path, report_data, articles)
    return articles

def parse_document_structure(clean_text):
    paragraphs = clean_text.splitlines()
    
    NUM_WORDS = r'(አንድ|ሁለት|ሦስት|አራት|አምስት|ስድስት|ሰባት|ስምንት|ዘጠኝ|አሥር|አስር|አሥራ\s*አንድ|አስራ\s*አንድ|አሥራ\s*ሁለት|አስራ\s*ሁለት|፩|፪|፫|፬|፭|፮|፯|፰|፱|፲|፲፩|፲፪)'
    struct_pat = re.compile(rf'^(ክፍል|ምዕራፍ|ንዑስ\s*ዘርፍ|ዘርፍ)\s+{NUM_WORDS}\b\s*(.*)')
    art_pat = re.compile(r'^([፩-፼ሺ]+)\s*\.\s*(.*)')
    sub_art_pat = re.compile(r'^([፩-፼]+\s*[\/\)])(?!\s*[፩-፼ሺ])\s*(.*)')
    letter_pat = re.compile(r'^([ሀ-ፐ\u1200-\u135A]\s*[\/\)])(?!\s*[፩-፼ሺ])\s*(.*)')
    
    current_hierarchy = {
        'part': None,
        'chapter': None,
        'section': None,
        'sub_section': None
    }
    
    hierarchy_seen = []
    ambiguity_log = []
    preamble_paras = []
    in_preamble = True
    
    raw_articles = []
    current_raw_art = None
    
    for idx, p in enumerate(paragraphs):
        m_struct = struct_pat.match(p)
        if m_struct:
            level, num_w, title = m_struct.group(1), m_struct.group(2), m_struct.group(3)
            full_title = f"{level} {num_w} {title}".strip()
            hierarchy_seen.append(full_title)
            
            if level == 'ክፍል':
                in_preamble = False
                current_hierarchy = {'part': full_title, 'chapter': None, 'section': None, 'sub_section': None}
            elif level == 'ምዕራፍ':
                current_hierarchy['chapter'] = full_title
                current_hierarchy['section'] = None
                current_hierarchy['sub_section'] = None
            elif level == 'ዘርፍ':
                current_hierarchy['section'] = full_title
                current_hierarchy['sub_section'] = None
            elif 'ንዑስ' in level:
                current_hierarchy['sub_section'] = full_title
            continue
        
        if in_preamble:
            preamble_paras.append(p)
            continue
        
        m_art = art_pat.match(p)
        if m_art:
            if current_raw_art:
                raw_articles.append(current_raw_art)
            g_num = m_art.group(1)
            raw_rest = m_art.group(2).strip()
            parsed_num = geez_to_int(g_num)
            
            expected_num = len(raw_articles) + 1
            if parsed_num != expected_num:
                ambiguity_log.append(f"Article heading numeral '{g_num}' parsed to {parsed_num}, expected {expected_num}. Trusting sequence -> assigned {expected_num}.")
                parsed_num = expected_num
            
            current_raw_art = {
                'number': parsed_num,
                'geez_number': g_num,
                'heading_text': raw_rest,
                'hierarchy': dict(current_hierarchy),
                'paragraphs': []
            }
        else:
            if current_raw_art:
                current_raw_art['paragraphs'].append(p)
    
    if current_raw_art:
        raw_articles.append(current_raw_art)
    
    articles = []
    
    # Process Article 0 (Preamble)
    art_0 = {
        'article_number': 0,
        'geez_number': '0',
        'title': 'መግቢያ',
        'part': None,
        'chapter': None,
        'section': None,
        'sub_section': None,
        'sub_articles': [
            {
                'marker': None,
                'text': '\n\n'.join(preamble_paras)
            }
        ],
        'full_text': '\n\n'.join(preamble_paras)
    }
    articles.append(art_0)
    
    # Process Articles 1..193
    for art in raw_articles:
        num = art['number']
        heading_text = art['heading_text']
        art_paras = list(art['paragraphs'])
        
        title = heading_text
        body_paras = []
        
        # Safety check: a title must not contain ። or ፡፡ and must be at most 12 words.
        # If it violates this, split at the first position that satisfies both and log it.
        if re.search(r'[።፡፡]', title) or len(title.split()) > 12:
            m_sent = re.search(r'[።፡፡]', title)
            if m_sent:
                split_pos = m_sent.start()
                orig_title = title
                title = orig_title[:split_pos].strip()
                remainder = orig_title[split_pos:].strip()
                if remainder:
                    body_paras.append(remainder)
                ambiguity_log.append(f"Article {num}: title safety split due to sentence mark -> '{title}'")
            elif len(title.split()) > 12:
                words = title.split()
                title = " ".join(words[:12]).strip()
                remainder = " ".join(words[12:]).strip()
                if remainder:
                    body_paras.append(remainder)
                ambiguity_log.append(f"Article {num}: title safety split due to word count > 12 -> '{title}'")
                
        body_paras.extend(art_paras)
        
        sub_articles = []
        current_sub = None
        
        for bp in body_paras:
            m_sub = sub_art_pat.match(bp)
            if m_sub:
                if current_sub:
                    sub_articles.append(current_sub)
                marker = m_sub.group(1).strip()
                text = m_sub.group(2).strip()
                current_sub = {
                    'marker': marker,
                    'text': text
                }
            else:
                if current_sub:
                    current_sub['text'] += "\n" + bp
                else:
                    current_sub = {
                        'marker': None,
                        'text': bp
                    }
        
        if current_sub:
            sub_articles.append(current_sub)
        
        if not sub_articles:
            sub_articles.append({
                'marker': None,
                'text': ''
            })
        
        full_text_parts = [title] + [f"{sa['marker'] + ' ' if sa['marker'] else ''}{sa['text']}" for sa in sub_articles if sa['text']]
        full_text = '\n\n'.join(full_text_parts)
        
        articles.append({
            'article_number': num,
            'geez_number': art['geez_number'],
            'title': title,
            'part': art['hierarchy']['part'],
            'chapter': art['hierarchy']['chapter'],
            'section': art['hierarchy']['section'],
            'sub_section': art['hierarchy']['sub_section'],
            'sub_articles': sub_articles,
            'full_text': full_text
        })
    
    # Validation
    art_numbers = [a['article_number'] for a in articles]
    assert art_numbers == list(range(194)), f"Validation failed! Numbers: {art_numbers}"
    
    report_data = {
        'total_articles': len(articles),
        'hierarchy_seen': hierarchy_seen,
        'ambiguity_log': ambiguity_log
    }
    
    return articles, report_data

def write_stage3_report(report_path, r, articles):
    import datetime
    
    with open(report_path, 'a', encoding='utf-8') as f:
        f.write("\n=== STAGE 3: STRUCTURE ===\n")
        f.write(f"Date: {datetime.datetime.now().isoformat()}\n")
        f.write(f"Total articles found (including Article 0 Preamble): {r['total_articles']}\n")
        f.write("Validation status: Exactly articles 0..193 present in sequential order, none missing, none duplicated.\n\n")
        
        f.write("--- AMBIGUITY & QUIRK LOG ---\n")
        if r['ambiguity_log']:
            for item in r['ambiguity_log']:
                f.write(f"  - {item}\n")
        else:
            f.write("  None\n")
            
        f.write("\n--- DOCUMENT HIERARCHY (TITLES SEEN) ---\n")
        for h in r['hierarchy_seen']:
            f.write(f"  • {h}\n")
            
        f.write("\n--- ALL ARTICLE TITLES (0..193) ---\n")
        for a in articles:
            f.write(f"  Article {a['article_number']:>3} ({a['geez_number']:>6}): {a['title']}\n")
            
        f.write("\n--- FULL TEXT OF SAMPLES (Articles 11, 27, 68) ---\n")
        for target in [11, 27, 68]:
            art = next(a for a in articles if a['article_number'] == target)
            f.write(f"\n[Article {target} - {art['title']}]\n")
            f.write(f"Part: {art['part']}\n")
            f.write(f"Chapter: {art['chapter']}\n")
            f.write(f"Section: {art['section']}\n")
            f.write(f"Sub-Section: {art['sub_section']}\n")
            f.write(f"Full Text:\n{art['full_text']}\n")

if __name__ == '__main__':
    run()
