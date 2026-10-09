#!/usr/bin/env python3
"""
Amharic Speech-to-Speech RAG Query Intent Router & Dispatcher.
Implements hierarchical, production-ready query routing:
1. GREETING_SMALLTALK -> Fast direct conversational response (zero latency, zero hallucination)
2. META_IDENTITY -> AI assistant capabilities, identity, and usage instructions
3. DOC_OVERVIEW -> Macro document summaries, structural overviews, and applicability scope
4. LEGAL_RETRIEVAL -> Dense/Hybrid retrieval over 224 fine-grained proclamation articles
"""
import re
import string
from enum import Enum
from pathlib import Path

from src.fidel_fold import fold
from src.prompts import SYSTEM_PROMPT_AMHARIC, format_rag_prompt
from src.macro_chunks import get_macro_chunks

class QueryIntent(str, Enum):
    GREETING = "GREETING"
    GREETING_SMALLTALK = "GREETING_SMALLTALK"  # Alias for backward compatibility
    THANK_YOU = "THANK_YOU"
    GOODBYE = "GOODBYE"
    META_IDENTITY = "META_IDENTITY"
    DOC_OVERVIEW = "DOC_OVERVIEW"
    LEGAL_RETRIEVAL = "LEGAL_RETRIEVAL"

# Fast lexical & transliterated token sets
GREETING_TOKENS = {
    "ሰላም", "ሰላምታ", "ጤናይስጥልኝ", "እንደምን", "እንደምንዋልክ", "እንደምንዋልሽ", "እንደምንዋላችሁ",
    "እንደምንአደርክ", "እንደምንአደርሽ", "እንደምንአደራችሁ", "እንደምንአመሸህ", "እንደምንአመሻችሁ",
    "እንዴትነህ", "እንዴትነሽ", "እንዴትናችሁ", "እንዴትኖት", "ደህናነህ", "ደህናነሽ", "ደህናናችሁ", "ደህናኖት",
    "selam", "salam", "hi", "hello", "hey", "greetings"
}

THANK_YOU_TOKENS = {
    "አመሰግናለሁ", "አመሰግናለው", "እናመሰግናለን", "አመሰገንኩ", "ተመስገን", "እግዚአብሔርይስጥልኝ",
    "እናመሰግናለን", "በርታ", "በርቱ", "በርቺ", "ጎበዝ",
    "thanks", "thank", "thx", "appreciate", "grateful", "appreciated"
}

GOODBYE_TOKENS = {
    "ቻው", "ባይ", "ባይባይ", "ደህናሁን", "ደህናሁኚ", "ደህናሁኑ", "ደህናይሁኑ", "ደህናዋል", "ደህናዋይ", "ደህናዋሉ",
    "ደህናእደር", "ደህናእደሪ", "ደህናእደሩ", "ይበቃኛል", "ጨርሻለሁ", "ጨረስኩ",
    "bye", "goodbye", "byebye", "cya", "farewell"
}

# Substantive legal topic keywords for hybrid-query disambiguation
LEGAL_KEYWORDS = {
    "ውል", "ቅጥር", "ስንብት", "ማቋረጥ", "ደመወዝ", "ደሞዝ", "ፈቃድ", "ሰዓት", "ሙከራ", "ጉዳት", "ካሣ", "ካሳ",
    "ማኅበር", "ማህበር", "ክርክር", "ቦርድ", "አንቀጽ", "አንቀፅ", "አዋጅ", "ሕግ", "ህግ", "ቅጣት", "ዕረፍት",
    "እረፍት", "ትርፍ", "ወሊድ", "አደጋ", "ተቀናሽ", "አሠሪ", "አሰሪ", "ሠራተኛ", "ሰራተኛ", "መብት", "ግዴታ",
    "ክፍያ", "ቦርድ", "ዳኝነት", "ስምምነት", "ድርድር", "መመሪያ", "አድማ", "መቆለፍ", "ደኅንነት", "ደህንነት",
    "article", "proclamation", "contract", "salary", "wage", "leave", "overtime", "probation", "severance"
}

# Regex patterns for conversational greetings & pleasantries
GREETING_PATTERNS = [
    r'^(ሰላም|ሰላምታ|ጤና\s*ይስጥልኝ|እንደምን\s*(አደርክ|አደርሽ|አደራችሁ|ዋልክ|ዋልሽ|ዋላችሁ|አመሸህ|አመሻችሁ)|እንዴት\s*(ነህ|ነሽ|ናችሁ|ኖት|አለህ|አለሽ)|ደህና\s*(ነህ|ነሽ|ናችሁ|ኖት)|ሰላም\s*(ነህ|ነሽ|ናችሁ))(\s*|\?|!|\.|$)',
    r'^(selam|salam|hi|hello|hey|good\s*(morning|afternoon|evening))(\s*|\?|!|\.|$)'
]

# Regex patterns for gratitude / thank-you
THANK_YOU_PATTERNS = [
    r'(አመሰግናለሁ|አመሰግናለው|እናመሰግናለን|አመሰገንኩ|እግዚአብሔር\s*ይስጥልኝ|ተመስገን|በጣም\s*አመሰግናለሁ|እጅግ\s*አመሰግናለሁ|ከልብ\s*አመሰግናለሁ|ምስጋናዬ\s*የላቀ\s*ነው)',
    r'(ጥሩ\s*ነው|አሪፍ\s*ነው|ረዳኸኝ|ረዳሽኝ|ረዳችሁኝ|ጠቅመኸኛል|ጥሩ\s*ረዳት\s*ነህ)',
    r'\b(thanks|thank\s*you|thx|appreciate\s*it|thank\s*you\s*so\s*much|thanks\s*a\s*lot|much\s*appreciated|grateful|great\s*job|good\s*job)\b'
]

# Regex patterns for farewell / goodbye
GOODBYE_PATTERNS = [
    r'(ደህና\s*(ሁን|ሁኚ|ሁኑ|ይሁኑ|ዋል|ዋይ|ዋሉ|እደር|እደሪ|እደሩ)|ቻው|ባይ\s*ባይ|ባይ|ስንብት|በሌላ\s*ጊዜ\s*እንገናኝ)',
    r'መልካም\s*(ቀን|ምሽት|ዕረፍት|እረፍት|ጊዜ|የሥራ\s*ጊዜ|የስራ\s*ጊዜ|ይሁንልህ|ይሁንልሽ|ይሁንላችሁ|ይሁንልዎ)',
    r'^(ይበቃኛል|ጨርሻለሁ|ጨረስኩ|ለዛሬ\s*ይበቃል|ሌላ\s*ጥያቄ\s*የለኝም)(\s*|\.|\!|$)',
    r'\b(bye|goodbye|bye\s*bye|cya|see\s*you|see\s*ya|have\s*a\s*nice\s*day|have\s*a\s*good\s*day|have\s*a\s*good\s*one|take\s*care|good\s*night|farewell|i\s*am\s*done|that\s*is\s*all)\b'
]

# Regex patterns for AI identity & capability questions
META_IDENTITY_PATTERNS = [
    r'ማን\s*(ነህ|ነሽ|ናት|ነው|ኖት|ናችሁ)',
    r'(who\s*are\s*you|what\s*are\s*you)',
    r'(ምን|እንዴት)\s*(ልትረዳኝ|ልትረጂኝ|ይረዳል|ይረዱናል|ይረዳኛል|ልትጠቅመኝ|ትጠቅማለህ|ትችላለህ|ትሰራለህ|ይሰራል)',
    r'(ምን\s*አይነት|ምን\s*ዓይነት)\s*(ጥያቄ|አገልግሎት|እርዳታ)',
    r'(ስለ\s*ራስህ|ስለ\s*ራስሽ|ስለ\s*ሲስተሙ|ስለ\s*ቦቱ|ስለ\s*ረዳቱ|ስለዚህ\s*ሲስተም)',
    r'(አገልግሎትህ|ተግባርህ|አላማህ|ዓላማህ|የአገልግሎት\s*መመሪያ|ማንነት)',
    r'(ሲስተም|ቦት|ረዳት).*(ምን\s*(ይሰራል|ያደርጋል|ይጠቅማል))',
    r'ምን\s*መስራት\s*ትችላለህ',
    r'ምን\s*ልጠይቅህ',
    r'ምን\s*ልትመልስ'
]

# Regex patterns for Macro Document Overview and Scope
OVERVIEW_PATTERNS = [
    r'(ስለ\s*ምን\s*ያወራል|ስለ\s*ምን\s*ይናገራል|ስለ\s*ምን\s*ነው|ምን\s*ይዟል)',
    r'(አጠቃላይ\s*ይዘት|አጠቃላይ\s*መዋቅር|አጠቃላይ\s*ማጠቃለያ|ዋና\s*ዋና)',
    r'(የአዋጁ\s*ዓላማ|የአዋጁ\s*አላማ|የአዋጁ\s*ይዘት|የአዋጁ\s*ርዕስ)',
    r'(ስንት\s*ክፍል|ስንት\s*አንቀጽ|ስንት\s*ምዕራፍ|ዋና\s*ክፍል|ክፍሎቹ|ምዕራፎቹ)',
    r'(የተፈጻሚነት\s*ወሰን|አዋጁ\s*በማን\s*ላይ|አዋጁ\s*ለማን\s*አይመለከትም|የማይመለከታቸው|ተፈጻሚ\s*አይሆንም)',
    r'(ሕጉ\s*የሚሸፍናቸው|አዋጁ\s*የሚሸፍናቸው|ማንን\s*አይመለከትም|ማን\s*ላይ\s*ይሰራል)'
]

GREETING_REGEX = re.compile("|".join(GREETING_PATTERNS), re.IGNORECASE)
THANK_YOU_REGEX = re.compile("|".join(THANK_YOU_PATTERNS), re.IGNORECASE)
GOODBYE_REGEX = re.compile("|".join(GOODBYE_PATTERNS), re.IGNORECASE)
META_IDENTITY_REGEX = re.compile("|".join(META_IDENTITY_PATTERNS), re.IGNORECASE)
OVERVIEW_REGEX = re.compile("|".join(OVERVIEW_PATTERNS), re.IGNORECASE)

# Substantive legal subject nouns (with optional Amharic grammatical prefixes)
LEGAL_NOUNS_PATTERN = [
    r'(የ|በ|ከ|ለ|ስለ|እንደ)?(ሙከራ|ውል|ቅጥር|ስንብት|ማቋረጥ|ደመወዝ|ደሞዝ|ፈቃድ|ሰዓት|ጉዳት|ካሣ|ካሳ|ማኅበር|ማህበር|ክርክር|ቦርድ|አንቀጽ|አንቀፅ|አዋጅ|ሕግ|ህግ|ቅጣት|ዕረፍት|እረፍት|ትርፍ|ወሊድ|አደጋ|ተቀናሽ|አሠሪ|አሰሪ|ሠራተኛ|ሰራተኛ|መብት|ግዴታ|ክፍያ|ዳኝነት|ስምምነት|ድርድር|መመሪያ|አድማ|መቆለፍ|ደኅንነት|ደህንነት)',
    r'\b(article|proclamation|contract|salary|wage|leave|overtime|probation|severance|penalty|dismissal)\b'
]
LEGAL_NOUNS_REGEX = re.compile("|".join(LEGAL_NOUNS_PATTERN), re.IGNORECASE)

def normalize_text(text: str) -> str:
    """Normalizes whitespace and removes punctuation for resilient classification."""
    if not text:
        return ""
    # Remove standard punctuation and Ethiopic punctuation
    cleaned = re.sub(r'[\?!\.,;:፣፤፦፨።\(\)\[\]\"\'\-]', ' ', text)
    return " ".join(cleaned.strip().split())

def has_substantive_legal_query(text: str, folded_text: str) -> bool:
    """
    Checks if a query contains substantive legal topics,
    ensuring hybrid queries (e.g. 'አመሰግናለሁ ግን የሙከራ ጊዜ ስንት ቀን ነው?') route to retrieval.
    """
    if LEGAL_NOUNS_REGEX.search(text) or LEGAL_NOUNS_REGEX.search(folded_text):
        words = text.split()
        if len(words) <= 4 and any(g in text for g in ["መልካም ቀን", "መልካም ምሽት", "መልካም ዕረፍት", "መልካም እረፍት", "መልካም ጊዜ", "መልካም የሥራ ጊዜ"]):
            return False
        return True
    return False

def classify_query_intent(query_text: str) -> QueryIntent:
    """
    Robust multi-layer intent classifier:
    1. Disambiguates hybrid utterances (e.g. thanks/greeting + substantive legal question).
    2. Identifies Thanks & Gratitude (አመሰግናለሁ, thank you).
    3. Identifies Farewells & Goodbyes (ደህና ሁን, ቻው, bye, have a nice day).
    4. Identifies Greetings & Pleasantries (ሰላም, hi, ጤና ይስጥልኝ).
    5. Identifies Meta Assistant & Identity (ማን ነህ, who are you).
    6. Identifies Macro Document Overview & Scope (ስለ ምን ያወራል, የተፈጻሚነት ወሰን).
    7. Defaults to fine-grained Legal Retrieval.
    """
    if not query_text:
        return QueryIntent.LEGAL_RETRIEVAL
        
    raw_clean = normalize_text(query_text)
    folded_q = fold(raw_clean)
    words_raw = set(raw_clean.lower().split())
    words_folded = set(folded_q.split())
    all_words = words_raw | words_folded

    # Check if this query contains substantive legal questions or overview topics
    is_legal_substantive = has_substantive_legal_query(raw_clean, folded_q)
    is_doc_overview = bool(OVERVIEW_REGEX.search(raw_clean) or OVERVIEW_REGEX.search(folded_q))
    is_meta_identity = bool(META_IDENTITY_REGEX.search(raw_clean) or META_IDENTITY_REGEX.search(folded_q))

    # If the user asks an explicit document overview or identity question, prioritize it
    if is_doc_overview:
        return QueryIntent.DOC_OVERVIEW
    if is_meta_identity:
        return QueryIntent.META_IDENTITY

    # If there are substantive legal keywords and length > 3 words, route to Legal Retrieval
    if is_legal_substantive and len(raw_clean.split()) > 3:
        return QueryIntent.LEGAL_RETRIEVAL

    # 1. Gratitude / Thanks check
    if (all_words & THANK_YOU_TOKENS) or THANK_YOU_REGEX.search(raw_clean) or THANK_YOU_REGEX.search(folded_q):
        if not is_legal_substantive:
            return QueryIntent.THANK_YOU

    # 2. Farewell / Goodbye check
    if (all_words & GOODBYE_TOKENS) or GOODBYE_REGEX.search(raw_clean) or GOODBYE_REGEX.search(folded_q):
        if not is_legal_substantive:
            return QueryIntent.GOODBYE

    # 3. Pure Greeting / Pleasantry check
    if (all_words & GREETING_TOKENS) or GREETING_REGEX.search(raw_clean) or GREETING_REGEX.search(folded_q):
        if not is_legal_substantive:
            return QueryIntent.GREETING_SMALLTALK

    # 4. Fallback check for legal keywords vs retrieval
    return QueryIntent.LEGAL_RETRIEVAL

def route_query(query_text: str, chunks: list = None, retriever_fn = None, top_k: int = 3) -> dict:
    """
    Production query router and dispatcher:
    - GREETING / GREETING_SMALLTALK -> Returns welcoming spoken Amharic greeting.
    - THANK_YOU -> Returns warm, appreciative spoken Amharic acknowledgment.
    - GOODBYE -> Returns courteous spoken Amharic farewell and best wishes.
    - META_IDENTITY -> Injects assistant capability card and provides direct explanation.
    - DOC_OVERVIEW -> Injects macro summary and scope chunks.
    - LEGAL_RETRIEVAL -> Executes hybrid/dense retriever for fine-grained articles.
    """
    intent = classify_query_intent(query_text)
    macro_chunks = get_macro_chunks()
    
    if intent in (QueryIntent.GREETING, QueryIntent.GREETING_SMALLTALK):
        direct_response = (
            "ሰላም! ጤና ይስጥልኝ። እኔ ስለ ኢትዮጵያ የአሠሪና ሠራተኛ ጉዳይ አዋጅ ቁጥር ፩ሺ፩፻፶፮/፪ሺ፲፩ "
            "መረጃ የምሰጥ የሕግ ረዳት AI ነኝ። ስለ የሥራ ውል፣ ደመወዝ፣ የሙከራ ጊዜ፣ ፈቃድ ወይም "
            "ሌሎች የሥራ ሕግ ድንጋጌዎች ምን ማወቅ ይፈልጋሉ?"
        )
        selected_chunks = [c for c in macro_chunks if c['id'] == 'a000_assistant']
        return {
            "intent": "GREETING_SMALLTALK",
            "query": query_text,
            "chunks": selected_chunks,
            "context_passages": [c['text_spoken'] for c in selected_chunks],
            "direct_response": direct_response,
            "prompt": format_rag_prompt(query_text, [c['text_spoken'] for c in selected_chunks], "GREETING_SMALLTALK")
        }

    elif intent == QueryIntent.THANK_YOU:
        direct_response = (
            "ምንም አይደል! ስለተገለገሉ እናመሰግናለን። ሌላ ተጨማሪ የሥራ ሕግ ጥያቄ ወይም "
            "የማብራሪያ ፍላጎት ካለዎት በማንኛውም ጊዜ ለመርዳት ዝግጁ ነኝ።"
        )
        selected_chunks = [c for c in macro_chunks if c['id'] == 'a000_assistant']
        return {
            "intent": QueryIntent.THANK_YOU.value,
            "query": query_text,
            "chunks": selected_chunks,
            "context_passages": [c['text_spoken'] for c in selected_chunks],
            "direct_response": direct_response,
            "prompt": format_rag_prompt(query_text, [c['text_spoken'] for c in selected_chunks], QueryIntent.THANK_YOU.value)
        }

    elif intent == QueryIntent.GOODBYE:
        direct_response = (
            "ደህና ይሁኑ! መልካም ቀንና የተሳካ የሥራ ጊዜ ይሁንልዎ። "
            "ስለ ኢትዮጵያ የሥራ ሕግ ማወቅ የሚፈልጉት ነገር ሲኖር ሁልጊዜም እዚህ አለሁ።"
        )
        selected_chunks = [c for c in macro_chunks if c['id'] == 'a000_assistant']
        return {
            "intent": QueryIntent.GOODBYE.value,
            "query": query_text,
            "chunks": selected_chunks,
            "context_passages": [c['text_spoken'] for c in selected_chunks],
            "direct_response": direct_response,
            "prompt": format_rag_prompt(query_text, [c['text_spoken'] for c in selected_chunks], QueryIntent.GOODBYE.value)
        }
        
    elif intent == QueryIntent.META_IDENTITY:
        selected_chunks = [c for c in macro_chunks if c['id'] == 'a000_assistant']
        direct_response = (
            "እኔ የኢትዮጵያ የአሠሪና ሠራተኛ ጉዳይ አዋጅ ቁጥር ፩ሺ፩፻፶፮/፪ሺ፲፩ ላይ የተመሰረተ የሕግ ረዳት AI ነኝ። "
            "ስለ ቅጥር ውል፣ የሙከራ ጊዜ፣ ደመወዝ፣ የሥራ ሰዓት፣ የዓመትና የወሊድ ፈቃድ፣ የሥራ ላይ አደጋ ካሣ፣ "
            "የሠራተኛ ማኅበራት እና የሥራ ክርክር አፈታት ሕጋዊ ድንጋጌዎችን በማብራራት ልረዳዎት እችላለሁ።"
        )
        return {
            "intent": intent.value,
            "query": query_text,
            "chunks": selected_chunks,
            "context_passages": [c['text_spoken'] for c in selected_chunks],
            "direct_response": direct_response,
            "prompt": format_rag_prompt(query_text, [c['text_spoken'] for c in selected_chunks], intent.value)
        }
        
    elif intent == QueryIntent.DOC_OVERVIEW:
        selected_chunks = [c for c in macro_chunks if c['id'] in ['a000_overview', 'a000_scope']]
        return {
            "intent": intent.value,
            "query": query_text,
            "chunks": selected_chunks,
            "context_passages": [c['text_spoken'] for c in selected_chunks],
            "direct_response": None,
            "prompt": format_rag_prompt(query_text, [c['text_spoken'] for c in selected_chunks], intent.value)
        }
        
    else: # LEGAL_RETRIEVAL
        if retriever_fn is not None:
            retrieved = retriever_fn(query_text, top_k=top_k)
        elif chunks is not None:
            # Fallback simple keyword / token overlap if retriever not provided
            q_terms = set(fold(normalize_text(query_text)).split())
            scored = []
            for c in chunks:
                c_terms = set(c.get('text_folded', '').split())
                overlap = len(q_terms & c_terms)
                scored.append((overlap, c))
            scored.sort(key=lambda x: x[0], reverse=True)
            retrieved = [item[1] for item in scored[:top_k]]
        else:
            retrieved = []
            
        passages = [c['text_spoken'] for c in retrieved]
        return {
            "intent": intent.value,
            "query": query_text,
            "chunks": retrieved,
            "context_passages": passages,
            "direct_response": None,
            "prompt": format_rag_prompt(query_text, passages, intent.value)
        }
