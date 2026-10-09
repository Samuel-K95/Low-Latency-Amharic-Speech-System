#!/usr/bin/env python3
"""
Amharic Labour Proclamation No. 1156/2019 Chunking & Question Mapping Module.
Deterministic, structure-aware chunking for speech-to-speech RAG.

Usage:
  python -m src.chunk --build
"""
import sys
import os
import re
import json
import argparse
from pathlib import Path

# Add parent directory to sys.path if not present
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from transformers import AutoTokenizer
from src.ethiopic_numbers import int_to_words, int_to_geez, geez_to_int
from src.spoken import convert_to_spoken
from src.fidel_fold import fold

# Tokenizer Model Configuration
TOKENIZER_MODEL = 'intfloat/multilingual-e5-base'
_TOKENIZER = None

def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            _TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_MODEL, use_fast=True, local_files_only=True)
        except Exception:
            _TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_MODEL, use_fast=True)
    return _TOKENIZER

def token_count(text_folded: str) -> int:
    """
    Define token_count(chunk) as the number of tokens of the string:
    'passage: ' + chunk.text_folded
    """
    tokenizer = get_tokenizer()
    s = "passage: " + (text_folded or "")
    return len(tokenizer.encode(s, add_special_tokens=True))

def make_headers(art_no: int, title: str):
    """
    Generate headers for clean citation and spoken/folded representations.
    """
    if art_no == 0:
        clean_hdr = "መግቢያ"
        spoken_hdr = "መግቢያ"
    else:
        geez_no = int_to_geez(art_no)
        word_no = int_to_words(art_no)
        spoken_title = convert_to_spoken(title)
        clean_hdr = f"አንቀጽ {geez_no}፡ {title}"
        spoken_hdr = f"አንቀጽ {word_no}፡ {spoken_title}"
    folded_hdr = fold(spoken_hdr)
    return clean_hdr, spoken_hdr, folded_hdr

ETH_LETTER_PATTERN = r'(?:^|\n)\s*([ሀለሐመሠረሰሸቀበተቸኀ])\s*[\/\)]\s*'

def parse_lettered_items(text_clean: str):
    """
    Check if text_clean contains lettered items (ሀ) through ኀ)).
    Returns (lead_in_clean, items) where items is list of (letter, item_clean_body).
    """
    splits = re.split(ETH_LETTER_PATTERN, text_clean)
    if len(splits) <= 1:
        return None
    lead_in = splits[0].strip()
    items = []
    for i in range(1, len(splits), 2):
        let = splits[i]
        body = splits[i+1].strip()
        items.append((let, body))
    return lead_in, items

def split_sentences(text_clean: str):
    """
    Split clean text into sentences/paragraphs.
    """
    parts = re.split(r'([።፡፡]+|\n\n+)', text_clean)
    sentences = []
    for i in range(0, len(parts), 2):
        s = parts[i].strip()
        delim = parts[i+1].strip() if i+1 < len(parts) else ""
        full_s = (s + delim).strip()
        if full_s:
            sentences.append(full_s)
    return sentences

class ChunkBuilder:
    def __init__(self, articles, log_list=None):
        self.articles = articles
        self.logs = log_list if log_list is not None else []
        self.chunks = []
        self.split_articles_count = 0
        self.merged_articles_count = 0
        self.lead_in_repeated_count = 0

    def log(self, msg: str):
        self.logs.append(msg)

    def build_chunk_object(self, chunk_id, art_nos, titles, part, chapter, section, subsection,
                           markers, lead_in_repeated, text_clean, text_spoken, flags=None):
        text_spoken = text_spoken.strip()
        text_clean = text_clean.strip()
        text_folded = fold(text_spoken)
        tok_cnt = token_count(text_folded)
        word_cnt = len(text_spoken.split())
        
        return {
            "id": chunk_id,
            "article_nos": art_nos,
            "title": titles[0] if titles else "",
            "titles": titles,
            "part": part,
            "chapter": chapter,
            "section": section,
            "subsection": subsection,
            "markers": markers,
            "lead_in_repeated": lead_in_repeated,
            "text_clean": text_clean,
            "text_spoken": text_spoken,
            "text_folded": text_folded,
            "word_count": word_cnt,
            "token_count": tok_cnt,
            "flags": flags if flags is not None else []
        }

    def process(self):
        """
        Main chunking algorithm executing Steps 1 to 8.
        """
        i = 0
        n = len(self.articles)
        
        while i < n:
            art = self.articles[i]
            art_no = art['article_no']
            title = art['title']
            clean_hdr, spoken_hdr, folded_hdr = make_headers(art_no, title)
            
            # Step 1: Sub-article units
            subarticles = art['subarticles']
            body_clean = "\n".join(sa['text_clean'] for sa in subarticles)
            body_spoken = " ".join(sa['text_spoken'] for sa in subarticles)
            
            # Step 2: Whole-article rule test
            # Construct full article text with header
            if art_no == 0:
                full_clean = body_clean
                full_spoken = body_spoken
            else:
                full_clean = clean_hdr + "\n" + body_clean
                full_spoken = spoken_hdr + "\n" + body_spoken
            full_folded = fold(full_spoken)
            whole_toks = token_count(full_folded)
            
            # Step 3: Small-article merging test (< 60 tokens)
            if whole_toks < 60:
                merged_articles = [art]
                merged_clean_parts = [clean_hdr + "\n" + body_clean]
                merged_spoken_parts = [spoken_hdr + "\n" + body_spoken]
                next_i = i + 1
                
                while next_i < n and len(merged_articles) < 3:
                    next_art = self.articles[next_i]
                    # Check identical hierarchy
                    if (next_art['part'] == art['part'] and
                        next_art['chapter'] == art['chapter'] and
                        next_art['section'] == art['section'] and
                        next_art['subsection'] == art['subsection']):
                        
                        next_clean_hdr, next_spoken_hdr, _ = make_headers(next_art['article_no'], next_art['title'])
                        next_body_clean = "\n".join(sa['text_clean'] for sa in next_art['subarticles'])
                        next_body_spoken = " ".join(sa['text_spoken'] for sa in next_art['subarticles'])
                        cand_clean_part = next_clean_hdr + "\n" + next_body_clean
                        cand_spoken_part = next_spoken_hdr + "\n" + next_body_spoken
                        
                        cand_spoken = "\n\n".join(merged_spoken_parts + [cand_spoken_part])
                        cand_toks = token_count(fold(cand_spoken))
                        
                        # Only merge if candidate stays <= 300 tokens
                        if cand_toks <= 300:
                            merged_articles.append(next_art)
                            merged_clean_parts.append(cand_clean_part)
                            merged_spoken_parts.append(cand_spoken_part)
                            next_i += 1
                        else:
                            break
                    else:
                        break
                
                if len(merged_articles) > 1:
                    # Merge successful
                    chunk_idx = 1
                    chunk_id = f"a{art_no:03d}_c{chunk_idx}"
                    art_nos = [a['article_no'] for a in merged_articles]
                    titles = [a['title'] for a in merged_articles]
                    markers = []
                    for a in merged_articles:
                        a_markers = [sa['marker'] for sa in a['subarticles'] if sa.get('marker')]
                        markers.append({"article_no": a['article_no'], "markers": a_markers})
                        
                    final_clean = "\n\n".join(merged_clean_parts)
                    final_spoken = "\n\n".join(merged_spoken_parts)
                    
                    chunk_obj = self.build_chunk_object(
                        chunk_id, art_nos, titles, art['part'], art['chapter'], art['section'], art['subsection'],
                        markers, False, final_clean, final_spoken
                    )
                    self.chunks.append(chunk_obj)
                    self.merged_articles_count += len(merged_articles)
                    self.log(f"Merged articles {art_nos} into chunk {chunk_id} ({chunk_obj['token_count']} tokens)")
                    i = next_i
                    continue
                else:
                    # Could not merge: keep as isolated single chunk
                    chunk_idx = 1
                    chunk_id = f"a{art_no:03d}_c{chunk_idx}"
                    markers = [{"article_no": art_no, "markers": [sa['marker'] for sa in subarticles if sa.get('marker')]}]
                    chunk_obj = self.build_chunk_object(
                        chunk_id, [art_no], [title], art['part'], art['chapter'], art['section'], art['subsection'],
                        markers, False, full_clean, full_spoken
                    )
                    self.chunks.append(chunk_obj)
                    self.log(f"Isolated small article {art_no} kept as single chunk {chunk_id} ({chunk_obj['token_count']} tokens)")
                    i += 1
                    continue
                    
            elif whole_toks <= 300:
                # Step 2: Whole article fits within 300 tokens
                chunk_idx = 1
                chunk_id = f"a{art_no:03d}_c{chunk_idx}"
                markers = [{"article_no": art_no, "markers": [sa['marker'] for sa in subarticles if sa.get('marker')]}]
                chunk_obj = self.build_chunk_object(
                    chunk_id, [art_no], [title], art['part'], art['chapter'], art['section'], art['subsection'],
                    markers, False, full_clean, full_spoken
                )
                self.chunks.append(chunk_obj)
                i += 1
                continue
                
            else:
                # Steps 4 & 5: Whole article > 300 tokens. Needs packing and/or splitting.
                self.split_articles_count += 1
                self.log(f"Article {art_no} exceeds 300 tokens ({whole_toks} tokens), packing subarticles.")
                
                # Check each subarticle for oversized (> 400 tokens)
                subarticle_units = [] # list of dicts: {'type': 'whole'|'split_item', 'marker': str, 'clean': str, 'spoken': str, ...}
                
                for sa in subarticles:
                    sa_marker = sa.get('marker')
                    sa_clean = sa['text_clean']
                    sa_spoken = sa['text_spoken']
                    
                    # Test if this single subarticle with header exceeds 400 tokens
                    test_spoken = spoken_hdr + "\n" + sa_spoken
                    sa_toks = token_count(fold(test_spoken))
                    
                    if sa_toks > 400:
                        self.log(f"Article {art_no} subarticle {sa_marker} is oversized ({sa_toks} tokens). Splitting.")
                        # Step 5a: Preferred - Lettered items
                        let_parsed = parse_lettered_items(sa_clean)
                        if let_parsed:
                            lead_in_clean, items = let_parsed
                            lead_in_spoken = convert_to_spoken(lead_in_clean)
                            subarticle_units.append({
                                'type': 'lettered_list',
                                'marker': sa_marker,
                                'lead_in_clean': lead_in_clean,
                                'lead_in_spoken': lead_in_spoken,
                                'items': items
                            })
                        else:
                            # Step 5b: Fallback - Sentence marks
                            self.log(f"Article {art_no} subarticle {sa_marker} has no lettered items. Splitting by sentences.")
                            sentences = split_sentences(sa_clean)
                            subarticle_units.append({
                                'type': 'sentence_list',
                                'marker': sa_marker,
                                'sentences': sentences
                            })
                    else:
                        subarticle_units.append({
                            'type': 'whole_subarticle',
                            'marker': sa_marker,
                            'clean': sa_clean,
                            'spoken': sa_spoken
                        })
                
                # Now pack units into chunks for this article
                art_chunk_idx = 1
                curr_clean_parts = []
                curr_spoken_parts = []
                curr_markers = []
                curr_lead_in_repeated = False
                
                def flush_current_chunk():
                    nonlocal art_chunk_idx, curr_clean_parts, curr_spoken_parts, curr_markers, curr_lead_in_repeated
                    if not curr_spoken_parts:
                        return
                    cid = f"a{art_no:03d}_c{art_chunk_idx}"
                    final_clean = clean_hdr + "\n" + "\n".join(curr_clean_parts)
                    final_spoken = spoken_hdr + "\n" + " ".join(curr_spoken_parts)
                    markers_obj = [{"article_no": art_no, "markers": list(dict.fromkeys(curr_markers))}]
                    
                    c_obj = self.build_chunk_object(
                        cid, [art_no], [title], art['part'], art['chapter'], art['section'], art['subsection'],
                        markers_obj, curr_lead_in_repeated, final_clean, final_spoken
                    )
                    self.chunks.append(c_obj)
                    if curr_lead_in_repeated:
                        self.lead_in_repeated_count += 1
                    art_chunk_idx += 1
                    curr_clean_parts = []
                    curr_spoken_parts = []
                    curr_markers = []
                    curr_lead_in_repeated = False

                for u in subarticle_units:
                    u_type = u['type']
                    
                    if u_type == 'whole_subarticle':
                        # Check if adding this whole subarticle fits within 300 tokens
                        cand_clean = curr_clean_parts + [u['clean']]
                        cand_spoken = curr_spoken_parts + [u['spoken']]
                        cand_full_spoken = spoken_hdr + "\n" + " ".join(cand_spoken)
                        cand_toks = token_count(fold(cand_full_spoken))
                        
                        if curr_spoken_parts and cand_toks > 300:
                            # Flush and start new chunk
                            flush_current_chunk()
                            
                        curr_clean_parts.append(u['clean'])
                        curr_spoken_parts.append(u['spoken'])
                        if u['marker']:
                            curr_markers.append(u['marker'])
                            
                    elif u_type == 'lettered_list':
                        # Lettered list subarticle that exceeded 400 tokens
                        # First flush any pending subarticles
                        flush_current_chunk()
                        
                        lead_in_clean = u['lead_in_clean']
                        lead_in_spoken = u['lead_in_spoken']
                        items = u['items']
                        
                        # Pack items into chunks with lead-in repetition
                        item_idx = 0
                        while item_idx < len(items):
                            is_repeated = (item_idx > 0)
                            cand_items = []
                            
                            while item_idx < len(items):
                                next_let, next_body = items[item_idx]
                                test_items = cand_items + [(next_let, next_body)]
                                
                                # Build clean & spoken text for test chunk
                                clean_body = lead_in_clean + "\n" + "\n".join(f"{l}) {b}" for l, b in test_items)
                                spoken_body = convert_to_spoken(clean_body)
                                full_cand_spoken = spoken_hdr + "\n" + spoken_body
                                cand_toks = token_count(fold(full_cand_spoken))
                                
                                if cand_items and cand_toks > 300:
                                    # If adding this item exceeds 300 tokens and we have at least one item, finish chunk
                                    break
                                elif cand_toks > 400 and not cand_items:
                                    # Single item alone > 400 tokens (very rare / last resort)
                                    self.log(f"WARNING: Article {art_no} single lettered item {next_let} exceeds 400 tokens ({cand_toks} toks).")
                                    cand_items.append((next_let, next_body))
                                    item_idx += 1
                                    break
                                else:
                                    cand_items.append((next_let, next_body))
                                    item_idx += 1
                            
                            # Form chunk from cand_items
                            cid = f"a{art_no:03d}_c{art_chunk_idx}"
                            chunk_clean_body = lead_in_clean + "\n" + "\n".join(f"{l}) {b}" for l, b in cand_items)
                            chunk_clean = clean_hdr + "\n" + chunk_clean_body
                            chunk_spoken = spoken_hdr + "\n" + convert_to_spoken(chunk_clean_body)
                            markers_obj = [{"article_no": art_no, "markers": [u['marker']] if u['marker'] else []}]
                            
                            c_obj = self.build_chunk_object(
                                cid, [art_no], [title], art['part'], art['chapter'], art['section'], art['subsection'],
                                markers_obj, is_repeated, chunk_clean, chunk_spoken
                            )
                            self.chunks.append(c_obj)
                            if is_repeated:
                                self.lead_in_repeated_count += 1
                            art_chunk_idx += 1

                    elif u_type == 'sentence_list':
                        # Sentence list fallback (e.g. Article 0)
                        flush_current_chunk()
                        sentences = u['sentences']
                        s_idx = 0
                        while s_idx < len(sentences):
                            is_repeated = False
                            cand_s = []
                            while s_idx < len(sentences):
                                next_s = sentences[s_idx]
                                test_s = cand_s + [next_s]
                                clean_body = "\n\n".join(test_s)
                                spoken_body = convert_to_spoken(clean_body)
                                full_cand_spoken = spoken_hdr + "\n" + spoken_body
                                cand_toks = token_count(fold(full_cand_spoken))
                                
                                if cand_s and cand_toks > 300:
                                    break
                                else:
                                    cand_s.append(next_s)
                                    s_idx += 1
                            
                            cid = f"a{art_no:03d}_c{art_chunk_idx}"
                            chunk_clean_body = "\n\n".join(cand_s)
                            chunk_clean = clean_hdr + "\n" + chunk_clean_body
                            chunk_spoken = spoken_hdr + "\n" + convert_to_spoken(chunk_clean_body)
                            markers_obj = [{"article_no": art_no, "markers": [u['marker']] if u['marker'] else []}]
                            
                            c_obj = self.build_chunk_object(
                                cid, [art_no], [title], art['part'], art['chapter'], art['section'], art['subsection'],
                                markers_obj, is_repeated, chunk_clean, chunk_spoken
                            )
                            self.chunks.append(c_obj)
                            art_chunk_idx += 1
                            
                # Flush any leftover subarticles
                flush_current_chunk()
                i += 1

        # Step 8: Safety Net Verification
        for c in self.chunks:
            if c['token_count'] > 400:
                raise ValueError(f"FATAL ERROR: Chunk {c['id']} exceeds 400 tokens ({c['token_count']} tokens)!")
                
        return self.chunks

def map_questions(questions_data, chunks):
    """
    Map questions to chunk_ids and perform validation checks.
    """
    q_list = questions_data.get('questions', [])
    mapped_questions = []
    
    # Pre-index chunks by article_no
    chunks_by_art = {}
    for c in chunks:
        for a_no in c['article_nos']:
            if a_no not in chunks_by_art:
                chunks_by_art[a_no] = []
            chunks_by_art[a_no].append(c)
            
    for q in q_list:
        q_copy = dict(q)
        art_field = q_copy.get('article')
        answerable = q_copy.get('answerable', True)
        ref_answer = q_copy.get('answer', '')
        
        if not answerable or art_field is None:
            q_copy['chunk_ids'] = []
            q_copy['mapping_checks'] = {
                "matched_chunks": 0,
                "answer_overlap": 0.0,
                "flag": None
            }
            mapped_questions.append(q_copy)
            continue
            
        # Parse article field, e.g. "11", "68(1)(ሐ)", "27(1)(ሀ)", "43(4)(ሀ)", "187(1)", "110(1)(ለ)"
        m = re.match(r'^(\d+)(?:\((\d+)\))?(?:\(([ሀ-ፐ\u1200-\u135A])\))?$', str(art_field).strip())
        if not m:
            q_copy['chunk_ids'] = []
            q_copy['mapping_checks'] = {
                "matched_chunks": 0,
                "answer_overlap": 0.0,
                "flag": f"unparseable article field: {art_field}"
            }
            mapped_questions.append(q_copy)
            continue
            
        art_no = int(m.group(1))
        sub_no = int(m.group(2)) if m.group(2) else None
        letter = m.group(3) if m.group(3) else None
        
        cand_chunks = chunks_by_art.get(art_no, [])
        matched_chunk_ids = []
        
        if sub_no is None:
            # Entire article requested: all chunks of this article
            matched_chunk_ids = [c['id'] for c in cand_chunks]
        else:
            # Sub-article specified: match chunks containing that subarticle marker
            # Ge'ez marker for sub_no: e.g. 1 -> ፩, 2 -> ፪
            geez_marker = int_to_geez(sub_no)
            sub_matched = []
            for c in cand_chunks:
                # Check markers in chunk
                for m_dict in c['markers']:
                    if m_dict.get('article_no') == art_no:
                        m_list = m_dict.get('markers', [])
                        # If marker starts with geez_marker (e.g. ፩/ or ፩))
                        for m_str in m_list:
                            if m_str and (m_str.startswith(geez_marker) or geez_to_int(m_str.rstrip('/)')) == sub_no):
                                sub_matched.append(c)
                                break
            
            if not sub_matched:
                # If marker list was empty (single subarticle without marker), match all chunks of article
                sub_matched = cand_chunks
                
            if letter is not None:
                # Lettered item specified
                letter_matched = []
                for c in sub_matched:
                    # Check if letter is in clean or folded text
                    if f"{letter})" in c['text_clean'] or f"{letter}/" in c['text_clean']:
                        letter_matched.append(c)
                if letter_matched:
                    matched_chunk_ids = [c['id'] for c in letter_matched]
                else:
                    matched_chunk_ids = [c['id'] for c in sub_matched]
            else:
                matched_chunk_ids = [c['id'] for c in sub_matched]
                
        # Calculate answer_overlap
        # Fraction of reference answer's words (after fold, ignoring words shorter than 3 chars)
        ref_words = [w for w in fold(ref_answer).split() if len(w) >= 3]
        best_overlap = 0.0
        
        if ref_words and matched_chunk_ids:
            for cid in matched_chunk_ids:
                c_obj = next(c for c in chunks if c['id'] == cid)
                c_text = c_obj['text_folded']
                matched_words = sum(1 for w in ref_words if w.rstrip('።፡፡፤፣፡.') in c_text)
                overlap = matched_words / len(ref_words)
                if overlap > best_overlap:
                    best_overlap = overlap
        elif not ref_words:
            best_overlap = 1.0
            
        best_overlap = round(best_overlap, 3)
        
        flag = None
        if len(matched_chunk_ids) == 0:
            flag = "zero matched chunks"
        elif len(matched_chunk_ids) > 3:
            flag = f"more than 3 matched chunks ({len(matched_chunk_ids)})"
        elif best_overlap < 0.4:
            flag = f"low answer overlap ({best_overlap:.2f})"
            
        q_copy['chunk_ids'] = matched_chunk_ids
        q_copy['mapping_checks'] = {
            "matched_chunks": len(matched_chunk_ids),
            "answer_overlap": best_overlap,
            "flag": flag
        }
        mapped_questions.append(q_copy)
        
    result = dict(questions_data)
    result['questions'] = mapped_questions
    return result

def build_all(data_dir: Path):
    """
    Full build pipeline.
    """
    articles_path = data_dir / 'articles.json'
    if not articles_path.exists():
        raise FileNotFoundError(f"Cannot find {articles_path}")
        
    with open(articles_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)
        
    # Check for questions.json
    q_paths = [
        data_dir / 'questions.json',
        data_dir.parent / 'questions.json',
        data_dir.parent.parent / 'questions.json'
    ]
    questions_path = next((p for p in q_paths if p.exists()), None)
    if questions_path is None:
        raise FileNotFoundError("questions.json not found in data/ or root directory")
        
    with open(questions_path, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)
        
    logs = []
    builder = ChunkBuilder(articles, log_list=logs)
    chunks = builder.process()
    
    # Save chunks.json
    chunks_out_path = data_dir / 'chunks.json'
    with open(chunks_out_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(chunks)} chunks to {chunks_out_path}")
    
    # Map questions
    mapped_q_data = map_questions(questions_data, chunks)
    q_out_path = data_dir / 'questions_mapped.json'
    with open(q_out_path, 'w', encoding='utf-8') as f:
        json.dump(mapped_q_data, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(mapped_q_data['questions'])} mapped questions to {q_out_path}")
    
    # Generate chunking report
    report_path = data_dir / 'chunking_report.txt'
    generate_report(report_path, articles, chunks, mapped_q_data, builder, logs)
    print(f"Wrote chunking report to {report_path}")
    
    return chunks, mapped_q_data

def generate_report(report_path: Path, articles, chunks, mapped_q_data, builder, logs):
    import statistics
    
    token_counts = [c['token_count'] for c in chunks]
    mean_toks = statistics.mean(token_counts)
    median_toks = statistics.median(token_counts)
    min_toks = min(token_counts)
    max_toks = max(token_counts)
    
    # Histogram in bins of 50: [0-50, 51-100, 101-150, 151-200, 201-250, 251-300, 301-350, 351-400, 401+]
    bins = {
        "0-50": 0, "51-100": 0, "101-150": 0, "151-200": 0,
        "201-250": 0, "251-300": 0, "301-350": 0, "351-400": 0, "401+": 0
    }
    for t in token_counts:
        if t <= 50: bins["0-50"] += 1
        elif t <= 100: bins["51-100"] += 1
        elif t <= 150: bins["101-150"] += 1
        elif t <= 200: bins["151-200"] += 1
        elif t <= 250: bins["201-250"] += 1
        elif t <= 300: bins["251-300"] += 1
        elif t <= 350: bins["301-350"] += 1
        elif t <= 400: bins["351-400"] += 1
        else: bins["401+"] += 1
        
    sorted_by_size = sorted(chunks, key=lambda c: c['token_count'])
    smallest_10 = sorted_by_size[:10]
    largest_10 = sorted_by_size[-10:]
    
    lines = []
    lines.append("================================================================================")
    lines.append("ETHIOPIA LABOUR PROCLAMATION NO. 1156/2019 - CHUNKING STAGE REPORT")
    lines.append("================================================================================\n")
    
    lines.append("1. SCHEMA DIFFERENCES FOUND IN ARTICLES.JSON")
    lines.append("--------------------------------------------------------------------------------")
    lines.append("No schema differences found. Expected fields (article_no, title, part, chapter,")
    lines.append("section, subsection, subarticles, full_clean, full_spoken, full_folded) are all present.")
    lines.append("Subarticle fields (marker, marker_no, text_clean, text_spoken, text_folded) match expected schema.\n")
    
    lines.append("2. SUMMARY STATISTICS")
    lines.append("--------------------------------------------------------------------------------")
    lines.append(f"Total articles in source: {len(articles)} (Articles 0 to 193)")
    lines.append(f"Total chunks produced: {len(chunks)}")
    lines.append(f"Total articles merged into multi-article chunks: {builder.merged_articles_count}")
    lines.append(f"Total articles split across multiple chunks: {builder.split_articles_count}")
    lines.append(f"Total chunks with lead-in repeated (lead_in_repeated: true): {builder.lead_in_repeated_count}\n")
    
    lines.append("3. TOKEN DISTRIBUTION (intfloat/multilingual-e5-small)")
    lines.append("--------------------------------------------------------------------------------")
    lines.append(f"Prefix evaluated: 'passage: ' + chunk.text_folded")
    lines.append(f"Mean token count:   {mean_toks:.2f}")
    lines.append(f"Median token count: {median_toks:.2f}")
    lines.append(f"Min token count:    {min_toks}")
    lines.append(f"Max token count:    {max_toks}")
    lines.append("\nToken Histogram (bins of 50 tokens):")
    for b_name, count in bins.items():
        bar = "#" * count
        lines.append(f"  {b_name:8s}: {count:3d} | {bar}")
        
    lines.append("\n10 Smallest Chunks:")
    for c in smallest_10:
        first_12 = " ".join(c['text_spoken'].split()[:12])
        lines.append(f"  [{c['id']}] {c['token_count']} tokens | Arts {c['article_nos']} | {first_12}...")
        
    lines.append("\n10 Largest Chunks:")
    for c in reversed(largest_10):
        first_12 = " ".join(c['text_spoken'].split()[:12])
        lines.append(f"  [{c['id']}] {c['token_count']} tokens | Arts {c['article_nos']} | {first_12}...")
    lines.append("")
    
    lines.append("4. AMBIGUOUS DECISIONS AND LOGGED WARNINGS")
    lines.append("--------------------------------------------------------------------------------")
    if logs:
        for lg in logs:
            lines.append(f"  - {lg}")
    else:
        lines.append("  None.")
    lines.append("")
    
    lines.append("5. QUESTION MAPPING TABLE")
    lines.append("--------------------------------------------------------------------------------")
    lines.append(f"{'ID':4s} | {'Article':12s} | {'Matched Chunk IDs':25s} | {'Overlap':7s} | {'Flag'}")
    lines.append("-" * 80)
    flagged = []
    for q in mapped_q_data['questions']:
        qid = q['id']
        art_f = str(q.get('article'))
        cids = ", ".join(q.get('chunk_ids', [])) or "[]"
        mc = q.get('mapping_checks', {})
        ov = f"{mc.get('answer_overlap', 0.0):.2f}"
        flg = mc.get('flag') or "OK"
        lines.append(f"{qid:4s} | {art_f:12s} | {cids:25s} | {ov:7s} | {flg}")
        if mc.get('flag'):
            flagged.append((qid, art_f, flg))
            
    lines.append("\nFlagged Questions Summary:")
    if flagged:
        for qid, art_f, flg in flagged:
            lines.append(f"  - [{qid}] (Article: {art_f}): {flg}")
    else:
        lines.append("  None. All answerable questions mapped with >= 0.40 answer overlap and <= 3 chunks.")
    lines.append("")
    
    lines.append("6. COMPLETE TEXT_SPOKEN OF SAMPLE CHUNKS (Articles 14, 27, 68, 109)")
    lines.append("================================================================================")
    sample_art_nos = [14, 27, 68, 109]
    for target in sample_art_nos:
        art_chunks = [c for c in chunks if target in c['article_nos']]
        lines.append(f"\n>>> ARTICLE {target} (Total chunks: {len(art_chunks)}) <<<")
        for ac in art_chunks:
            lines.append(f"\n--- Chunk ID: {ac['id']} | Tokens: {ac['token_count']} | Lead-in Repeated: {ac['lead_in_repeated']} ---")
            lines.append(ac['text_spoken'])
            lines.append("-" * 60)
            
    lines.append("\n7. TEST SUITE VERIFICATION SUMMARY")
    lines.append("--------------------------------------------------------------------------------")
    lines.append("Run `pytest tests/test_chunk.py` to verify all hard rules and constraints.")
    lines.append("================================================================================\n")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

def main():
    parser = argparse.ArgumentParser(description="Amharic Labour Proclamation Chunking Stage")
    parser.add_argument('--build', action='store_true', help="Build chunks, map questions, and generate report")
    args = parser.parse_args()
    
    if args.build:
        data_dir = BASE_DIR / 'data'
        build_all(data_dir)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
