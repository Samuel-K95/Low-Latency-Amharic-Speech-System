"""Shared Amharic test prompts/responses covering the assignment's stress cases."""
# Response-like sentences for TTS: short -> long, names, places, numbers, dates, code-switching.
TTS_SENTENCES = {
    "greeting":      "ሰላም! እንዴት ልረዳዎት እችላለሁ?",
    "capital":       "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት።",
    "weather":       "ዛሬ የአየር ሁኔታው ፀሐያማ ነው፣ ነገር ግን ምሽት ላይ ዝናብ ሊኖር ይችላል።",
    "museum_hours":  "ብሔራዊ ሙዚየም ከጠዋቱ ሶስት ሰዓት እስከ ማታ አስራ አንድ ሰዓት ክፍት ነው።",
    "names_places":  "ዶክተር አበበ በቀለ ነገ ከሰዓት በኋላ በባህር ዳር ዩኒቨርሲቲ ስለ ጤና አጠባበቅ ንግግር ያደርጋሉ። ስብሰባው የፕሮጀክቱ አካል ነው።",
    "digits_date":   "ስብሰባው መስከረም 15 ቀን 2019 ዓ.ም. በ3 ሰዓት ይጀምራል።",
    "code_switch":   "የእርስዎ meeting ነገ በ Zoom ላይ ይካሄዳል፣ link ን በ email እልክልዎታለሁ።",
}
# Conversational user questions used as LLM prompts.
USER_QUESTIONS = [
    "የኢትዮጵያ ዋና ከተማ ማን ናት?",
    "ነገ ወደ ባህር ዳር ለመሄድ ምን ማድረግ አለብኝ?",
    "ለጉንፋን ምን ማድረግ ይሻላል?",
    "ስለ ላሊበላ አብያተ ክርስቲያናት በአጭሩ ንገረኝ።",
    "ቡና እንዴት ይፈላል?",
]
