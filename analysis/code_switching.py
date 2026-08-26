def find_switching_points(results: list[dict]) -> list[dict]:
    """
    Identify language switching points (Hindi <-> English), ignoring punctuation and numbers.

    Returns list of switch transition objects:
    - position (1-indexed token index)
    - from (previous language label)
    - to (current language label)
    - token (current token word)
    """
    switches = []
    
    # Filter out punctuation and number tokens
    valid_tokens = [
        (idx, r) for idx, r in enumerate(results)
        if r.get("script") not in ["Punctuation/Numeric", "Number"] and r.get("language") in ["hi", "en"]
    ]
    
    for i in range(1, len(valid_tokens)):
        prev_idx, prev_item = valid_tokens[i - 1]
        curr_idx, curr_item = valid_tokens[i]
        
        prev_lang = prev_item["language"]
        curr_lang = curr_item["language"]
        
        if prev_lang != curr_lang:
            switches.append({
                "position": curr_idx + 1,
                "from": prev_lang,
                "to": curr_lang,
                "token": curr_item["token"]
            })
            
    return switches

def code_switching_metrics(results: list[dict]) -> dict:
    """
    Calculate code switching metrics: total switches and switching rate.
    """
    switches = find_switching_points(results)
    valid_langs = [
        r["language"] for r in results
        if r.get("script") not in ["Punctuation/Numeric", "Number"] and r.get("language") in ["hi", "en"]
    ]
    
    if len(valid_langs) < 2:
        return {"switches": 0, "switching_rate": 0.0, "details": []}
        
    rate = len(switches) / (len(valid_langs) - 1)
    return {
        "switches": len(switches),
        "switching_rate": round(rate, 4),
        "details": switches
    }

def language_distribution(results: list[dict]) -> dict:
    """
    Calculate distribution percentages of language tokens (excluding punctuation).
    """
    words = [
        r["language"] for r in results
        if r.get("script") not in ["Punctuation/Numeric", "Number"]
    ]
    total = len(words)
    if not total:
        return {"hi": 0.0, "en": 0.0, "ot": 0.0, "counts": {"hi": 0, "en": 0, "ot": 0}, "total_words": 0}
        
    counts = {
        "hi": words.count("hi"),
        "en": words.count("en"),
        "ot": words.count("ot")
    }
    
    return {
        "hi": round(counts["hi"] / total * 100, 1),
        "en": round(counts["en"] / total * 100, 1),
        "ot": round(counts["ot"] / total * 100, 1),
        "counts": counts,
        "total_words": total
    }
