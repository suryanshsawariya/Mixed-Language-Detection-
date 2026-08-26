import re

# Common English vocabulary list for post-processing disambiguation
COMMON_ENGLISH_WORDS = {
    "college", "assignment", "submit", "complete", "presentation", "today", "meeting",
    "lecture", "class", "exam", "test", "course", "project", "work", "job", "time",
    "but", "and", "or", "if", "because", "so", "then", "have", "has", "had", "is", "are",
    "was", "were", "been", "being", "be", "do", "does", "did", "done", "doing", "can",
    "could", "should", "would", "may", "might", "must", "shall", "will", "good", "bad",
    "great", "late", "early", "new", "old", "first", "last", "next", "best", "better",
    "presentation", "report", "file", "document", "online", "offline", "system", "code"
}

# Basic POS tag rules for Hindi and English tokens for morphological analysis
HINDI_PRONOUNS = {"मेरा", "मैं", "तुम", "आप", "वह", "यह", "हम", "वे", "मेरा", "मेरी", "मेरे", "मुझे", "तुझे", "mera", "main", "tum", "aap", "woh", "yeh", "hum", "ve"}
HINDI_VERBS = {"है", "हूँ", "हैं", "था", "थी", "थे", "होगा", "रहता", "रहती", "रहते", "जाना", "करना", "हुआ", "गया", "hai", "hoon", "hain", "tha", "thi", "the", "rehta", "jaana", "karna", "hua"}
HINDI_POSTPOSITIONS = {"में", "पर", "से", "को", "का", "की", "के", "ने", "तक", "mein", "par", "se", "ko", "ka", "ki", "ke", "ne", "tak"}
ENGLISH_PRONOUNS = {"i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "their", "our"}
ENGLISH_AUX_VERBS = {"is", "am", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "can", "could", "would", "should", "will", "may", "might", "must"}

def split_words_keep_punct(text: str) -> list[str]:
    """
    Split text into tokens keeping Unicode Devanagari words intact while isolating punctuation,
    including Devanagari Danda (।) and Double Danda (॥).
    """
    if not text:
        return []
    # Devanagari range excluding danda \u0964 & \u0965
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
    if hi_count == total:
        return "PURE HINDI"
    if en_count == total:
        return "PURE ENGLISH"
    if hi_count > 0 and en_count > 0 and switches_count > 0:
        return "HINDI-ENGLISH CODE-MIXED"
    if hi_count > en_count:
        return "PREDOMINANTLY HINDI"
    if en_count > hi_count:
        return "PREDOMINANTLY ENGLISH"
    return "OTHER / MIXED"
