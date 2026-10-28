import re

# Comprehensive Common English vocabulary list for post-processing disambiguation
COMMON_ENGLISH_WORDS = {
    # Pronouns & Auxiliaries
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "their", "our", "its", "myself", "yourself",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "done", "doing",
    "can", "could", "should", "would", "may", "might", "must", "shall", "will",

    # Conjunctions & Prepositions
    "and", "or", "but", "if", "because", "so", "then", "for", "with", "about",
    "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",

    # Common Daily Nouns & Verbs & Adjectives
    "weather", "today", "tomorrow", "yesterday", "tired", "happy", "sad", "good", "bad",
    "great", "movie", "film", "cinema", "ending", "boring", "interesting", "nice",
    "college", "school", "university", "assignment", "submit", "complete", "completed",
    "presentation", "meeting", "lecture", "class", "exam", "test", "course", "project",
    "work", "job", "time", "day", "night", "morning", "evening", "name", "live", "living",
    "go", "going", "gone", "went", "come", "coming", "came", "see", "look", "watch",
    "think", "know", "want", "like", "love", "need", "make", "take", "get", "got",
    "first", "last", "next", "best", "better", "report", "file", "document",
    "online", "offline", "system", "code", "data", "deep", "learning", "nlp",
    "model", "detection", "language", "mixed", "text", "analysis", "user", "input"
}

# Standard Hindi lexemes (Devanagari + Romanized)
HINDI_PRONOUNS = {
    "मेरा", "मेरी", "मेरे", "मैं", "तुम", "आप", "वह", "यह", "हम", "वे", "मुझे", "तुझे",
    "उसका", "उसकी", "उसके", "इनका", "इनकी", "इनके", "हमारा", "हमारी", "हमारे",
    "mera", "meri", "mere", "main", "mai", "mein", "tum", "aap", "woh", "wo", "yeh", "ye", "hum", "ham", "ve"
}

HINDI_VERBS = {
    "है", "हूँ", "हैं", "हो", "था", "थी", "थे", "रहता", "रहती", "रहते", "रहना",
    "जाना", "जाता", "जाती", "जाते", "करना", "करता", "करती", "करते", "किया", "की", "किया",
    "हुआ", "हुई", "हुए", "होगा", "होगी", "होंगे",
    "hai", "hoon", "hain", "ho", "tha", "thi", "the", "rehta", "rehti", "rehte",
    "jaana", "jana", "jaate", "karna", "karo", "karta", "karti", "kya", "hua", "hui"
}

HINDI_POSTPOSITIONS = {
    "में", "पर", "से", "को", "का", "की", "के", "ने", "तक", "द्वारा", "लिए",
    "mein", "par", "se", "ko", "ka", "ki", "ke", "ne", "tak", "liye"
}

ENGLISH_PRONOUNS = {
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "their", "our"
}

ENGLISH_AUX_VERBS = {
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "can", "could", "would", "should", "will", "may", "might", "must"
}


def split_words_keep_punct(text: str) -> list[str]:
    """
    Split text into tokens keeping Unicode Devanagari words intact while isolating punctuation,
    including Devanagari Danda (।) and Double Danda (॥).
    """
    if not text:
        return []
    pattern = r'[\u0900-\u0963\u0966-\u097F]+|[a-zA-Z0-9_]+|[^\w\s\u0900-\u097F]|\u0964|\u0965'
    tokens = re.findall(pattern, text, re.UNICODE)
    return [t.strip() for t in tokens if t.strip()]


def detect_script(token: str) -> str:
    """
    Categorize Unicode script of a token.
    """
    if bool(re.match(r'^[^\w\s\u0900-\u097F]+$', token) or token in ['।', '॥']):
        return "Punctuation/Numeric"
    elif bool(re.match(r'^\d+$', token)):
        return "Number"
    elif bool(re.search(r'[\u0900-\u0963\u0966-\u097F]', token)):
        return "Devanagari"
    elif bool(re.search(r'[a-zA-Z]', token)):
        return "Latin"
    return "Other"


def get_script_distribution(tokens: list[str]) -> dict:
    """
    Calculate character-level script distribution in text.
    """
    dev_chars = 0
    lat_chars = 0
    other_chars = 0

    full_text = " ".join(tokens)
    for char in full_text:
        if char.isspace():
            continue
        code = ord(char)
        if (0x0900 <= code <= 0x0963) or (0x0966 <= code <= 0x097F):
            dev_chars += 1
        elif (0x0041 <= code <= 0x005A) or (0x0061 <= code <= 0x007A):
            lat_chars += 1
        else:
            other_chars += 1

    total = dev_chars + lat_chars + other_chars
    if total == 0:
        return {"devanagari": 0.0, "latin": 0.0, "other": 0.0}

    return {
        "devanagari": round((dev_chars / total) * 100, 1),
        "latin": round((lat_chars / total) * 100, 1),
        "other": round((other_chars / total) * 100, 1)
    }


def annotate_pos_and_morphology(token: str, lang: str, script: str) -> dict:
    """
    Perform lightweight POS tagging & morphological analysis.
    """
    t_lower = token.lower()

    if script in ["Punctuation/Numeric", "Number"]:
        pos = "PUNCT / NUM"
        morph = "Non-Linguistic Token"
    elif lang == "hi":
        if t_lower in HINDI_PRONOUNS:
            pos = "PRON (Pronoun)"
            morph = "Personal / Possessive Pronoun"
        elif t_lower in HINDI_VERBS:
            pos = "VERB (Verb / Aux)"
            morph = "Finite Verb / Auxiliary"
        elif t_lower in HINDI_POSTPOSITIONS:
            pos = "ADP (Postposition)"
            morph = "Case Marker / Postposition"
        else:
            pos = "NOUN / ADJ"
            morph = "Devanagari / Romanized Hindi Lexeme"
    elif lang == "en":
        if t_lower in ENGLISH_PRONOUNS:
            pos = "PRON (Pronoun)"
            morph = "Personal / Possessive Pronoun"
        elif t_lower in ENGLISH_AUX_VERBS:
            pos = "AUX (Auxiliary Verb)"
            morph = "Auxiliary / Modal Verb"
        elif t_lower in COMMON_ENGLISH_WORDS:
            pos = "NOUN / VERB / ADJ"
            morph = "English Content Word"
        else:
            pos = "NOUN / ADJ"
            morph = "Latin English Lexeme"
    else:
        pos = "SYM / OTHER"
        morph = "Other Symbol / Token"

    return {"pos": pos, "morphology": morph}


def determine_input_type(word_tokens: list[dict], switches_count: int) -> str:
    """
    Classify sentence high-level input type:
    - PURE HINDI
    - PURE ENGLISH
    - HINDI-ENGLISH CODE-MIXED
    - PREDOMINANTLY HINDI
    - PREDOMINANTLY ENGLISH
    - OTHER / NON-LINGUISTIC
    """
    if not word_tokens:
        return "OTHER / NON-LINGUISTIC"

    hi_count = sum(1 for w in word_tokens if w["language"] == "hi")
    en_count = sum(1 for w in word_tokens if w["language"] == "en")
    ot_count = sum(1 for w in word_tokens if w["language"] == "ot")
    total = len(word_tokens)

    if total == 0:
        return "OTHER / NON-LINGUISTIC"

    # Check script breakdown of original tokens
    dev_tokens = sum(1 for w in word_tokens if w.get("script") == "Devanagari")
    lat_tokens = sum(1 for w in word_tokens if w.get("script") == "Latin")

    if hi_count == total and lat_tokens == 0:
        return "PURE HINDI"
    if en_count == total and dev_tokens == 0:
        return "PURE ENGLISH"
    if (hi_count > 0 and en_count > 0) or (dev_tokens > 0 and lat_tokens > 0):
        return "HINDI-ENGLISH CODE-MIXED"
    if hi_count > en_count:
        return "PREDOMINANTLY HINDI"
    if en_count > hi_count:
        return "PREDOMINANTLY ENGLISH"
    return "OTHER / MIXED"
