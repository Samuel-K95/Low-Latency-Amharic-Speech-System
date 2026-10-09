"""
Unit tests for Query Intent Router and Macro Chunks.
Verifies robust handling of:
1. Greetings & Pleasantries (selam, ሰላም, hi, ጤና ይስጥልኝ)
2. Gratitude & Thank-Yous (አመሰግናለሁ, እናመሰግናለን, በጣም አመሰግናለሁ, thanks, thank you)
3. Farewells & Goodbyes (ደህና ሁን, ቻው, ባይ, መልካም ቀን, bye, goodbye, have a nice day)
4. Meta Identity & Assistant Capabilities (ማን ነህ, who are you, ምን ትሰራለህ)
5. Document Macro Overview & Scope (ስለ ምን ያወራል, ዋና ዋና ክፍሎች, ለማን አይመለከትም)
6. Specific Legal Inquiries (የሙከራ ጊዜ, የወሊድ ፈቃድ, የሥራ ስንብት)
7. Hybrid query disambiguation (thanks + legal question -> LEGAL_RETRIEVAL)
"""
import pytest
from src.router import classify_query_intent, route_query, QueryIntent
from src.macro_chunks import get_macro_chunks
from src.prompts import SYSTEM_PROMPT_AMHARIC, format_rag_prompt
from src.chunk import token_count

def test_intent_classification_greetings():
    greeting_queries = [
        "selam",
        "ሰላም",
        "salam",
        "hi",
        "hello",
        "ጤና ይስጥልኝ",
        "እንደምን ዋልክ",
        "እንደምን አደርክ",
        "እንዴት ነህ",
        "ደህና ነህ?",
        "እንደምን ዋላችሁ"
    ]
    for q in greeting_queries:
        intent = classify_query_intent(q)
        assert intent in (QueryIntent.GREETING, QueryIntent.GREETING_SMALLTALK), f"Failed for greeting query '{q}', got {intent}"

def test_intent_classification_gratitude():
    thank_queries = [
        "አመሰግናለሁ",
        "አመሰግናለው",
        "እናመሰግናለን",
        "በጣም አመሰግናለሁ",
        "እጅግ በጣም አመሰግናለሁ",
        "እግዚአብሔር ይስጥልኝ",
        "ተመስገን",
        "ጥሩ ነው ረዳኸኝ",
        "thanks",
        "thank you",
        "thank you so much",
        "thx",
        "appreciate it"
    ]
    for q in thank_queries:
        intent = classify_query_intent(q)
        assert intent == QueryIntent.THANK_YOU, f"Failed for thank-you query '{q}', got {intent}"

def test_intent_classification_farewells():
    goodbye_queries = [
        "ደህና ሁን",
        "ደህና ሁኚ",
        "ደህና ሁኑ",
        "ደህና ይሁኑ",
        "ቻው",
        "ባይ",
        "ባይ ባይ",
        "መልካም ቀን",
        "መልካም ምሽት",
        "መልካም የሥራ ጊዜ",
        "ይበቃኛል",
        "ጨርሻለሁ",
        "bye",
        "goodbye",
        "see you",
        "have a nice day",
        "take care"
    ]
    for q in goodbye_queries:
        intent = classify_query_intent(q)
        assert intent == QueryIntent.GOODBYE, f"Failed for goodbye query '{q}', got {intent}"

def test_intent_classification_meta_identity():
    meta_queries = [
        "ማን ነህ?",
        "ማን ነህ",
        "who are you",
        "ምን ትሰራለህ?",
        "እንዴት ልትረዳኝ ትችላለህ?",
        "ምን ዓይነት አገልግሎት ትሰጣለህ?",
        "ስለ ራስህ ንገረኝ",
        "ይህ ሲስተም ምን ይሰራል?",
        "ምን ልጠይቅህ እችላለሁ?"
    ]
    for q in meta_queries:
        intent = classify_query_intent(q)
        assert intent == QueryIntent.META_IDENTITY, f"Failed for meta query '{q}', got {intent}"

def test_intent_classification_doc_overview():
    overview_queries = [
        "ይህ አዋጅ ስለ ምን ያወራል?",
        "የአዋጁ አጠቃላይ ይዘት ምንድን ነው?",
        "ዋና ዋና ክፍሎቹ ምን ምን ናቸው?",
        "የአዋጁ ዓላማ ምንድን ነው?",
        "አዋጁ በማን ላይ ተፈጻሚ ይሆናል?",
        "አዋጁ ለማን አይመለከትም?",
        "ሕጉ ማንን አይመለከትም?"
    ]
    for q in overview_queries:
        intent = classify_query_intent(q)
        assert intent == QueryIntent.DOC_OVERVIEW, f"Failed for overview query '{q}', got {intent}"

def test_intent_classification_legal_retrieval_and_hybrid_disambiguation():
    pure_legal = [
        "የሙከራ ጊዜ ከስንት ቀን መብለጥ የለበትም?",
        "የዓመት ፈቃድ ስንት የሥራ ቀናት ይሰጣል?",
        "የሥራ ስንብት ክፍያ እንዴት ይሰላል?",
        "የትርፍ ሰዓት ክፍያ ተመን ምን ያህል ነው?",
        "ነፍሰጡር ሠራተኛ የወሊድ ፈቃድ ስንት ቀን ነው?"
    ]
    for q in pure_legal:
        intent = classify_query_intent(q)
        assert intent == QueryIntent.LEGAL_RETRIEVAL, f"Failed for legal query '{q}', got {intent}"

    # Hybrid queries: User says thanks or bye BUT also asks a substantive question
    hybrid_queries = [
        "አመሰግናለሁ ግን የሙከራ ጊዜ ስንት ቀን ነው?",
        "በጣም አመሰግናለሁ ስለ ዓመት ፈቃድ ንገረኝ",
        "ደህና ሁን ነገር ግን የሥራ ስንብት ክፍያ እንዴት ይሰላል?",
        "Thanks, how is overtime calculated under the proclamation?"
    ]
    for q in hybrid_queries:
        intent = classify_query_intent(q)
        assert intent == QueryIntent.LEGAL_RETRIEVAL, f"Failed to disambiguate hybrid query '{q}', got {intent}"

def test_macro_chunks_schema_and_token_cap():
    chunks = get_macro_chunks()
    assert len(chunks) == 3, f"Expected 3 macro chunks, got {len(chunks)}"
    
    expected_ids = {"a000_overview", "a000_scope", "a000_assistant"}
    actual_ids = {c['id'] for c in chunks}
    assert actual_ids == expected_ids
    
    for c in chunks:
        assert c['text_clean']
        assert c['text_spoken']
        assert c['text_folded']
        toks = token_count(c['text_folded'])
        assert toks <= 400, f"Macro chunk {c['id']} exceeds 400 token limit: {toks} tokens"
        assert toks >= 50, f"Macro chunk {c['id']} is too small: {toks} tokens"

def test_router_dispatching():
    # 1. Test Greeting Routing
    res_greet = route_query("selam")
    assert res_greet['intent'] in ("GREETING", "GREETING_SMALLTALK")
    assert "ሰላም! ጤና ይስጥልኝ" in res_greet['direct_response']
    
    # 2. Test Thank You Routing
    res_thanks = route_query("በጣም አመሰግናለሁ!")
    assert res_thanks['intent'] == "THANK_YOU"
    assert "ምንም አይደል" in res_thanks['direct_response']
    
    # 3. Test Goodbye Routing
    res_bye = route_query("ደህና ሁን፣ መልካም ቀን ይሁንልህ")
    assert res_bye['intent'] == "GOODBYE"
    assert "ደህና ይሁኑ" in res_bye['direct_response']

    # 4. Test Meta Routing
    res_meta = route_query("ማን ነህ?")
    assert res_meta['intent'] == "META_IDENTITY"
    assert res_meta['direct_response'] is not None
    assert len(res_meta['chunks']) == 1
    assert res_meta['chunks'][0]['id'] == 'a000_assistant'
    
    # 5. Test Overview Routing
    res_ov = route_query("ይህ አዋጅ ስለ ምን ያወራል?")
    assert res_ov['intent'] == "DOC_OVERVIEW"
    assert len(res_ov['chunks']) == 2
    chunk_ids = [c['id'] for c in res_ov['chunks']]
    assert 'a000_overview' in chunk_ids
    assert 'a000_scope' in chunk_ids
    
    # 6. Test Legal Retrieval Routing
    sample_chunks = [
        {"id": "a011_c1", "text_spoken": "የሙከራ ጊዜ ስልሳ ቀናት ነው", "text_folded": "የሙከራ ጊዜ ስልሳ ቀናት ነው"}
    ]
    res_legal = route_query("የሙከራ ጊዜ ስንት ቀን ነው?", chunks=sample_chunks)
    assert res_legal['intent'] == "LEGAL_RETRIEVAL"
    assert len(res_legal['chunks']) == 1
    assert "የሙከራ ጊዜ" in res_legal['prompt']

