import re
from pathlib import Path
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification

from analysis.script_utils import (
    split_words_keep_punct,
    detect_script,
    get_script_distribution,
    annotate_pos_and_morphology,
    determine_input_type,
    COMMON_ENGLISH_WORDS,
)
from analysis.code_switching import find_switching_points, language_distribution

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "saved" / "xlmr"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_xlmr_model():
    """
    Load XLM-RoBERTa tokenizer and model cached in Streamlit resource cache.
    """
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Trained XLM-R model directory not found at: {MODEL_DIR}")
        
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForTokenClassification.from_pretrained(str(MODEL_DIR))
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

def analyze_text(text: str, tokenizer, model) -> dict:
    """
    Perform full tokenization, XLM-R inference, script-aware postprocessing,
    code-switching analysis, and linguistic feature extraction.
    """
    tokens = split_words_keep_punct(text)
    if not tokens:
        return {
            "tokens": [],
            "summary": {
                "total_words": 0, "hi_count": 0, "en_count": 0, "ot_count": 0,
                "hi_pct": 0.0, "en_pct": 0.0, "ot_pct": 0.0,
                "dominant_lang": "None", "input_type": "OTHER / NON-LINGUISTIC",
                "code_switching": False, "switch_count": 0, "avg_confidence": 0.0
            },
            "switches": [],
            "script_dist": {"devanagari": 0.0, "latin": 0.0, "other": 0.0},
            "sequence": []
        }

    # Encode with HuggingFace BatchEncoding object
    encoded = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    
    # Extract word_ids BEFORE dictionary conversion to prevent AttributeError
    word_ids = encoded.word_ids(batch_index=0)
    inputs = {k: v.to(DEVICE) for k, v in encoded.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits = outputs.logits[0]
    probabilities = torch.softmax(logits, dim=-1)
    
    results = []
    conf_scores = []
    
    for word_idx, token in enumerate(tokens):
        subtoken_indices = [i for i, w_id in enumerate(word_ids) if w_id == word_idx]
        if subtoken_indices:
            word_probs = probabilities[subtoken_indices].mean(dim=0)
            pred_id = int(torch.argmax(word_probs).item())
            model_conf = float(word_probs[pred_id].item())
            model_label = model.config.id2label[pred_id]
            hi_prob = float(word_probs[0].item())
            en_prob = float(word_probs[1].item())
            ot_prob = float(word_probs[2].item())
        else:
            model_label = "ot"
            model_conf = 0.5
            hi_prob, en_prob, ot_prob = 0.0, 0.0, 1.0
            
        script_type = detect_script(token)
        has_devanagari = (script_type == "Devanagari")
        is_punct = (script_type == "Punctuation/Numeric")
        is_num = (script_type == "Number")
        is_latin = (script_type == "Latin")
        
        # Script-aware Decision Engine
        if is_punct or is_num:
            final_label = "ot"
            confidence = 1.0
        elif has_devanagari:
            final_label = "hi"
            confidence = 0.999
        elif is_latin:
            token_lower = token.lower()
            if model_label in ["en", "hi"]:
                final_label = model_label
                confidence = model_conf
            elif model_label == "ot":
                if token_lower in COMMON_ENGLISH_WORDS or en_prob > hi_prob:
                    final_label = "en"
                    confidence = max(en_prob, 0.85)
                elif hi_prob > en_prob:
                    final_label = "hi"
                    confidence = max(hi_prob, 0.85)
                else:
                    final_label = "en"
                    confidence = 0.80
        else:
            final_label = model_label
            confidence = model_conf
            
        pos_info = annotate_pos_and_morphology(token, final_label, script_type)
        conf_scores.append(confidence)
        
        results.append({
            "index": word_idx + 1,
            "token": token,
            "language": final_label,
            "confidence": confidence,
            "script": script_type,
            "model_pred": model_label,
            "pos": pos_info["pos"],
            "morphology": pos_info["morphology"]
        })

    # Calculations
    dist = language_distribution(results)
    switches = find_switching_points(results)
    script_dist = get_script_distribution(tokens)
    
    word_tokens = [r for r in results if r["script"] not in ["Punctuation/Numeric", "Number"]]
    input_type = determine_input_type(word_tokens, len(switches))
    
    # Dominant language calculation
    counts = dist["counts"]
    if counts["hi"] > counts["en"] and counts["hi"] >= counts["ot"]:
        dominant = "Hindi"
    elif counts["en"] > counts["hi"] and counts["en"] >= counts["ot"]:
        dominant = "English"
    elif counts["hi"] == counts["en"] and counts["hi"] > 0:
        dominant = "Hindi / English (Equal)"
    else:
        dominant = "Other / Non-Linguistic"
        
    avg_conf = sum(conf_scores) / len(conf_scores) if conf_scores else 0.0
    
    sequence = [f"{r['token']} ({r['language'].upper()})" for r in word_tokens]
    
    return {
        "tokens": results,
        "summary": {
            "total_words": dist["total_words"],
            "total_tokens": len(results),
            "hi_count": counts["hi"],
            "en_count": counts["en"],
            "ot_count": counts["ot"],
            "hi_pct": dist["hi"],
            "en_pct": dist["en"],
            "ot_pct": dist["ot"],
            "dominant_lang": dominant,
            "input_type": input_type,
            "code_switching": len(switches) > 0,
            "switch_count": len(switches),
            "avg_confidence": round(avg_conf * 100, 2)
        },
        "switches": switches,
        "script_dist": script_dist,
        "sequence": sequence
    }
