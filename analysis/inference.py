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
    HINDI_PRONOUNS,
    HINDI_VERBS,
    HINDI_POSTPOSITIONS,
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
    Returns JSON-serializable Python data structures.
    """
    tokens = split_words_keep_punct(text)
    if not tokens:
        return {
            "text": text,
            "tokens": [],
            "summary": {
                "total_words": 0,
                "total_tokens": 0,
                "hi_count": 0,
                "en_count": 0,
                "ot_count": 0,
                "hi_pct": 0.0,
                "en_pct": 0.0,
                "ot_pct": 0.0,
                "dominant_lang": "None",
                "input_type": "OTHER / NON-LINGUISTIC",
                "code_switching": False,
                "switch_count": 0,
                "avg_confidence": 0.0
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

    # Extract word_ids BEFORE dictionary conversion
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

        # Hybrid Decision Engine combining XLM-R Transformer & Script/Lexical Rules
        if is_punct or is_num:
            final_label = "ot"
            confidence = 1.0
        elif has_devanagari:
            final_label = "hi"
            confidence = 0.999
        elif is_latin:
            token_lower = token.lower()
            if token_lower in COMMON_ENGLISH_WORDS:
                final_label = "en"
                confidence = max(en_prob, 0.92)
            elif token_lower in HINDI_PRONOUNS or token_lower in HINDI_VERBS or token_lower in HINDI_POSTPOSITIONS:
                final_label = "hi"
                confidence = max(hi_prob, 0.92)
            elif en_prob >= hi_prob:
                final_label = "en"
                confidence = float(en_prob)
            elif hi_prob > en_prob:
                final_label = "hi"
                confidence = float(hi_prob)
            else:
                final_label = model_label
                confidence = float(model_conf)
        else:
            final_label = model_label
            confidence = float(model_conf)

        pos_info = annotate_pos_and_morphology(token, final_label, script_type)
        conf_scores.append(confidence)

        results.append({
            "index": int(word_idx + 1),
            "token": str(token),
            "language": str(final_label),
            "confidence": float(round(confidence, 4)),
            "script": str(script_type),
            "model_pred": str(model_label),
            "pos": str(pos_info["pos"]),
            "morphology": str(pos_info["morphology"])
        })

    # Calculations
    dist = language_distribution(results)
    switches = find_switching_points(results)
    script_dist = get_script_distribution(tokens)

    word_tokens = [r for r in results if r["script"] not in ["Punctuation/Numeric", "Number"]]
    input_type = determine_input_type(word_tokens, len(switches))

    # Dominant language calculation
    counts = dist["counts"]
    hi_c, en_c, ot_c = counts["hi"], counts["en"], counts["ot"]
    if hi_c > en_c and hi_c >= ot_c:
        dominant = "Hindi"
    elif en_c > hi_c and en_c >= ot_c:
        dominant = "English"
    elif hi_c > 0 and en_c > 0 and hi_c == en_c:
        dominant = "Hindi / English (Equal)"
    else:
        dominant = "Other / Non-Linguistic"

    avg_conf = float(sum(conf_scores) / len(conf_scores)) if conf_scores else 0.0

    sequence = [f"{r['token']} ({r['language'].upper()})" for r in word_tokens]

    return {
        "text": str(text),
        "tokens": results,
        "summary": {
            "total_words": int(dist["total_words"]),
            "total_tokens": int(len(results)),
            "hi_count": int(hi_c),
            "en_count": int(en_c),
            "ot_count": int(ot_c),
            "hi_pct": float(dist["hi"]),
            "en_pct": float(dist["en"]),
            "ot_pct": float(dist["ot"]),
            "dominant_lang": str(dominant),
            "input_type": str(input_type),
            "code_switching": bool(len(switches) > 0),
            "switch_count": int(len(switches)),
            "avg_confidence": float(round(avg_conf * 100, 2))
        },
        "switches": switches,
        "script_dist": script_dist,
        "sequence": sequence
    }
