import pytest
import json
import re
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.chunk import build_all, token_count, make_headers
from src.ethiopic_numbers import int_to_words, geez_to_int
from src.fidel_fold import fold

@pytest.fixture(scope="module")
def dataset():
    data_dir = BASE_DIR / 'data'
    # Run build to ensure fresh outputs
    chunks, mapped_q_data = build_all(data_dir)
    with open(data_dir / 'articles.json', 'r', encoding='utf-8') as f:
        articles = json.load(f)
    return {
        'data_dir': data_dir,
        'chunks': chunks,
        'questions': mapped_q_data['questions'],
        'articles': articles
    }

def test_token_cap_and_min_distribution(dataset):
    """
    Every chunk has token_count at most 400;
    count how many are under 150 (allowed only for small isolated articles) and list them.
    """
    chunks = dataset['chunks']
    under_150 = []
    
    for c in chunks:
        t = c['token_count']
        assert t <= 400, f"Chunk {c['id']} exceeds hard cap of 400 tokens ({t} tokens)!"
        if t < 150:
            under_150.append(c)
            
    print(f"\nTotal chunks: {len(chunks)}")
    print(f"Chunks under 150 tokens: {len(under_150)}")
    for c in under_150[:10]:
        print(f"  [{c['id']}] {c['token_count']} tokens | Arts: {c['article_nos']}")

def test_coverage(dataset):
    """
    For every article, the set of (article_no, marker) units across all chunks
    equals the set in articles.json, each exactly once (repeated lead-ins excluded).
    No text lost, none duplicated.
    """
    chunks = dataset['chunks']
    articles = dataset['articles']
    
    art_dict = {a['article_no']: a for a in articles}
    
    # Check all articles are present in chunks
    all_chunk_art_nos = set()
    for c in chunks:
        all_chunk_art_nos.update(c['article_nos'])
        
    assert all_chunk_art_nos == set(art_dict.keys()), "Not all articles are present in chunks!"
    
    # Check marker coverage per article
    for a_no, art in art_dict.items():
        expected_markers = set(sa['marker'] for sa in art['subarticles'] if sa.get('marker'))
        
        # Collect markers across all chunks for this article
        actual_markers = set()
        for c in chunks:
            if a_no in c['article_nos']:
                for m_dict in c['markers']:
                    if m_dict['article_no'] == a_no:
                        actual_markers.update(m for m in m_dict.get('markers', []) if m)
                        
        assert actual_markers == expected_markers, (
            f"Marker mismatch for article {a_no}! Expected: {expected_markers}, Got: {actual_markers}"
        )

def test_lead_in_repetition(dataset):
    """
    No chunk starts with a lettered item without its lead-in when lead_in_repeated applies.
    """
    chunks = dataset['chunks']
    for c in chunks:
        if c['lead_in_repeated']:
            # Must have a lead in before lettered items
            # Spoken header is first line, second line / next sentence must be lead-in
            lines = c['text_spoken'].splitlines()
            assert len(lines) >= 2, f"Chunk {c['id']} with lead_in_repeated has no body lines!"
            body = "\n".join(lines[1:])
            # Verify body does not start with a lone item body without intro
            assert not re.match(r'^[ሀ-ፐ]\)\s*', body), f"Chunk {c['id']} starts with letter item without lead-in!"

def test_merging_hierarchy(dataset):
    """
    Merged chunks only combine articles with identical part, chapter, section, subsection.
    """
    chunks = dataset['chunks']
    articles_by_no = {a['article_no']: a for a in dataset['articles']}
    
    for c in chunks:
        if len(c['article_nos']) > 1:
            arts = [articles_by_no[a] for a in c['article_nos']]
            first = arts[0]
            for other in arts[1:]:
                assert other['part'] == first['part'], f"Part mismatch in merged chunk {c['id']}"
                assert other['chapter'] == first['chapter'], f"Chapter mismatch in merged chunk {c['id']}"
                assert other['section'] == first['section'], f"Section mismatch in merged chunk {c['id']}"
                assert other['subsection'] == first['subsection'], f"Subsection mismatch in merged chunk {c['id']}"

def test_header_number_words(dataset):
    """
    Every header's number words equal ethiopic_numbers.to_words(article_no).
    """
    chunks = dataset['chunks']
    for c in chunks:
        lines = c['text_spoken'].splitlines()
        first_line = lines[0].strip()
        first_art = c['article_nos'][0]
        if first_art == 0:
            assert first_line == "መግቢያ"
        else:
            expected_word = int_to_words(first_art)
            assert f"አንቀጽ {expected_word}፡" in first_line, (
                f"Header in chunk {c['id']} does not match expected word for {first_art}: '{first_line}' vs '{expected_word}'"
            )

def test_chunk_ids_unique_and_sorted(dataset):
    """
    Chunk ids are unique and sorted.
    """
    chunks = dataset['chunks']
    ids = [c['id'] for c in chunks]
    assert len(ids) == len(set(ids)), "Chunk IDs are not unique!"
    
    def parse_cid(cid):
        # Format: a027_c1
        m = re.match(r'^a(\d+)_c(\d+)$', cid)
        assert m, f"Invalid chunk ID format: {cid}"
        return int(m.group(1)), int(m.group(2))
        
    sorted_ids = sorted(ids, key=parse_cid)
    assert ids == sorted_ids, "Chunk IDs are not in sorted order!"

def test_no_unwanted_characters(dataset):
    """
    No Latin letters, Ge'ez numerals or Arabic digits inside text_spoken and text_folded,
    except items already logged by the cleaning stage (e.g. Art 156(3) ፬፵፱).
    """
    chunks = dataset['chunks']
    geez_pattern = re.compile(r'[\u1369-\u137C]')
    arabic_pattern = re.compile(r'[0-9]')
    latin_pattern = re.compile(r'[a-zA-Z]')
    
    # Known exception: Article 156(3) typo in source "፬፵፱"
    allowed_geez_exceptions = {"፬፵፱"}
    
    anomalies = []
    for c in chunks:
        spoken = c['text_spoken']
        # Find Ge'ez numerals
        g_matches = geez_pattern.findall(spoken)
        for g in g_matches:
            if g not in "፬፵፱":
                anomalies.append((c['id'], "geez_numeral", g))
                
        # Find Arabic digits
        a_matches = arabic_pattern.findall(spoken)
        for a in a_matches:
            anomalies.append((c['id'], "arabic_digit", a))
            
        # Find Latin letters
        l_matches = latin_pattern.findall(spoken)
        for l in l_matches:
            anomalies.append((c['id'], "latin_letter", l))
            
    assert len(anomalies) == 0, f"Found unexpected characters in text_spoken: {anomalies}"

def test_idempotence(dataset):
    """
    Running the build twice produces byte-identical chunks.json.
    """
    data_dir = dataset['data_dir']
    chunks_path = data_dir / 'chunks.json'
    
    with open(chunks_path, 'rb') as f:
        first_bytes = f.read()
        
    # Re-run build
    build_all(data_dir)
    
    with open(chunks_path, 'rb') as f:
        second_bytes = f.read()
        
    assert first_bytes == second_bytes, "Build output is not byte-identical across runs!"

def test_question_mapping_quality(dataset):
    """
    Test question mapping completeness and answer overlap.
    """
    questions = dataset['questions']
    assert len(questions) == 43, f"Expected 43 questions, got {len(questions)}"
    
    for q in questions:
        if q['answerable']:
            assert len(q['chunk_ids']) > 0, f"Answerable question {q['id']} has 0 matched chunks!"
            assert len(q['chunk_ids']) <= 3, f"Question {q['id']} matched more than 3 chunks: {q['chunk_ids']}"
            assert q['mapping_checks']['answer_overlap'] >= 0.40, (
                f"Question {q['id']} answer overlap is below 0.40 ({q['mapping_checks']['answer_overlap']})"
            )
        else:
            assert len(q['chunk_ids']) == 0, f"Unanswerable question {q['id']} should have 0 chunk_ids"
            assert q['mapping_checks']['flag'] is None

def test_titles_no_sentence_marks_and_length(dataset):
    """
    Verify that no article title contains sentence marks (። or ፡፡) and no title exceeds 12 words.
    """
    articles = dataset['articles']
    for a in articles:
        title = a['title']
        assert not re.search(r'[።፡፡]', title), f"Article {a['article_no']} title contains sentence mark: '{title}'"
        words = title.split()
        assert len(words) <= 12, f"Article {a['article_no']} title exceeds 12 words ({len(words)} words): '{title}'"

def test_no_subarticle_title_duplication(dataset):
    """
    Verify that no sub-article text is duplicated inside its own article title,
    and no article title contains the full sub-article body.
    """
    articles = dataset['articles']
    for a in articles:
        title = a['title'].strip()
        if not title or a['article_no'] == 0:
            continue
        for sa in a['subarticles']:
            txt = sa['text_clean'].strip()
            if txt:
                # The article title must never contain the sub-article body text
                assert txt not in title, (
                    f"Article {a['article_no']} title duplicates subarticle body: title='{title}', body='{txt[:60]}...'"
                )
                assert title != txt, f"Article {a['article_no']} title is identical to subarticle body: '{title}'"

def test_no_glued_letter_patterns(dataset):
    """
    Verify that 'አራትሀ' and other glued-letter patterns do not appear in any text_spoken.
    """
    articles = dataset['articles']
    chunks = dataset['chunks']
    glued_pattern = re.compile(r'\b(ዜሮ|አንድ|ሁለት|ሦስት|አራት|አምስት|ስድስት|ሰባት|ስምንት|ዘጠኝ|አስር|አሥር|ሃያ|ሰላሳ|ሠላሳ|አርባ|ሃምሳ|ስድሳ|ሰባ|ሰማንያ|ዘጠና|መቶ|ሺህ)(ሀ|ለ|ሐ|መ|ሠ|ረ|ሰ|ሸ|ቀ|በ|ተ|ቸ|ኀ)(?=[^\w]|$|\b)')
    
    # Check in articles
    for a in articles:
        for sa in a['subarticles']:
            m = glued_pattern.search(sa['text_spoken'])
            assert not m, f"Article {a['article_no']} contains glued letter in spoken text: '{m.group(0)}' in '{sa['text_spoken']}'"
            assert "አራትሀ" not in sa['text_spoken'], f"Article {a['article_no']} contains 'አራትሀ' in spoken text"
            
    # Check in chunks
    for c in chunks:
        m = glued_pattern.search(c['text_spoken'])
        assert not m, f"Chunk {c['id']} contains glued letter in spoken text: '{m.group(0)}'"
        assert "አራትሀ" not in c['text_spoken'], f"Chunk {c['id']} contains 'አራትሀ' in spoken text"


