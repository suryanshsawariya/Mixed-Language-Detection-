import sys
from pathlib import Path
import json
import traceback

import numpy as np
import pandas as pd
import streamlit as st
import torch
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PROJECT PATH & IMPORTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.inference import load_xlmr_model, analyze_text
from analysis.script_utils import COMMON_ENGLISH_WORDS

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Code-Mixed Language Detection & Linguistic Analyzer",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS FOR STUNNING AESTHETICS
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%);
        padding: 30px 40px;
        border-radius: 16px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.3);
        border: 1px solid #3730a3;
    }

    .hero-title {
        font-size: 34px;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(90deg, #ffffff, #c7d2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        font-size: 16px;
        color: #e0e7ff;
        margin-top: 8px;
        font-weight: 400;
    }

    .token-badge {
        display: inline-flex;
        align-items: center;
        padding: 6px 14px;
        margin: 5px 6px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.15);
    }

    .token-hi {
        background: #15803d;
        color: #ffffff;
        border: 1px solid #22c55e;
    }

    .token-en {
        background: #1d4ed8;
        color: #ffffff;
        border: 1px solid #3b82f6;
    }

    .token-ot {
        background: #475569;
        color: #ffffff;
        border: 1px solid #64748b;
    }

    .token-sub {
        font-size: 11px;
        opacity: 0.9;
        margin-left: 7px;
        padding: 2px 5px;
        border-radius: 4px;
        background: rgba(0,0,0,0.3);
    }

    .section-header {
        font-size: 20px;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 25px;
        margin-bottom: 15px;
        border-bottom: 2px solid #334155;
        padding-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# LOAD MODEL IN CACHE
# ============================================================

try:
    tokenizer, model = load_xlmr_model()
    model_loaded = True
    model_error = None
except Exception as e:
    model_loaded = False
    model_error = str(e)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/language.png", width=64)
    st.title("Navigation")

    page = st.radio(
        "Select Page:",
        ["🔍 Live Text Analyzer", "📊 Model Comparison", "ℹ️ About Project"],
        index=0
    )

    st.divider()
    st.markdown("### ⚙️ System & Hardware")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        cuda_ver = torch.version.cuda
        st.success(f"🚀 **GPU Active**\n\n{gpu_name}\n\nCUDA: {cuda_ver}")
    else:
        st.info("💻 **CPU Mode Active**")

    st.divider()
    st.markdown("### 📊 Model Metrics Summary")
    st.markdown("""
    - **XLM-R Transformer**: Macro F1 = **95.20%**
    - **Baseline (TF-IDF + LR)**: Accuracy = **96.05%**
    - **Training Dataset**: 90,000 Balanced Tokens
    - **Supported Languages**: Hindi (`hi`), English (`en`), Other (`ot`)
    """)

    st.divider()
    st.caption("Mixed Language Detection & Computational Linguistics Project")

# ============================================================
# HEADER HERO
# ============================================================

st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">🌐 Code-Mixed Language Detection & Linguistic Analyzer</div>
        <div class="hero-subtitle">Real-Time Transformer-Based Language Identification (LID) for Hindi, English, and Code-Mixed Sentences</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# PAGE 1: LIVE TEXT ANALYZER
# ============================================================

if page == "🔍 Live Text Analyzer":

    if not model_loaded:
        st.error(f"❌ Could not load XLM-R Model: {model_error}")
        st.stop()

    st.markdown('<div class="section-header">✍️ Live Input Sentence & Preset Examples</div>', unsafe_allow_html=True)

    if "user_text" not in st.session_state:
        st.session_state["user_text"] = "मेरा नाम Suryansh है और मैं Karnataka में रहता हूँ।"

    st.markdown("**Click any preset example below to load into the analyzer:**")
    ex1, ex2, ex3, ex4, ex5, ex6 = st.columns(6)

    if ex1.button("Pure Devanagari", use_container_width=True):
        st.session_state["user_text"] = "मेरा नाम सूर्यांश है, मैं कर्नाटक में रहता हूँ।"
    if ex2.button("Pure English", use_container_width=True):
        st.session_state["user_text"] = "My name is Suryansh and I live in Karnataka."
    if ex3.button("Devanagari+Latin", use_container_width=True):
        st.session_state["user_text"] = "मेरा नाम Suryansh है और मैं Karnataka में रहता हूँ।"
    if ex4.button("Weather Mixed", use_container_width=True):
        st.session_state["user_text"] = "आज weather बहुत अच्छा है।"
    if ex5.button("Tired Mixed", use_container_width=True):
        st.session_state["user_text"] = "I am आज बहुत tired."
    if ex6.button("Unseen Sentence", use_container_width=True):
        st.session_state["user_text"] = "This project uses deep learning and NLP for language detection."

    text_input = st.text_area(
        "Enter any Hindi, English, or Code-Mixed sentence live:",
        value=st.session_state["user_text"],
        height=110,
        help="Type or paste any sentence in Hindi (Devanagari), English, or Hindi-English mixed text."
    )
    st.session_state["user_text"] = text_input

    c_btn1, c_btn2, c_space = st.columns([2, 2, 6])
    analyze_clicked = c_btn1.button("⚡ Analyze / Predict", type="primary", use_container_width=True)
    clear_clicked = c_btn2.button("🗑️ Clear Input", use_container_width=True)

    if clear_clicked:
        st.session_state["user_text"] = ""
        st.rerun()

    if analyze_clicked or st.session_state.get("auto_analyze", True):
        st.session_state["auto_analyze"] = False

        if not text_input.strip():
            st.warning("⚠️ Please enter a sentence to analyze.")
        else:
            with st.spinner("Processing sentence through XLM-RoBERTa NLP pipeline..."):
                try:
                    analysis = analyze_text(text_input.strip(), tokenizer, model)
                    summary = analysis["summary"]
                    tokens_data = analysis["tokens"]

                    st.markdown('<div class="section-header">🎯 Language Identification Results</div>', unsafe_allow_html=True)

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Dominant Language", summary["dominant_lang"])
                    m2.metric("Input Classification", summary["input_type"])
                    m3.metric("Code-Switching Transitions", summary["switch_count"])
                    m4.metric("Avg Model Confidence", f"{summary['avg_confidence']:.2f}%")

                    st.markdown("### 🏷️ Visual Token-Level Language Map")
                    badge_html = ""
                    for item in tokens_data:
                        lang_code = item["language"]
                        css_class = f"token-{lang_code}"
                        lang_title = {"hi": "Hindi", "en": "English", "ot": "Other"}.get(lang_code, lang_code.upper())
                        badge_html += f'<span class="token-badge {css_class}">{item["token"]} <span class="token-sub">{lang_title} ({item["confidence"]*100:.0f}%)</span></span> '
                    st.markdown(badge_html, unsafe_allow_html=True)

                    st.divider()

                    col_left, col_right = st.columns([6, 6])

                    with col_left:
                        st.markdown("### 📋 Token Classification Table")
                        table_rows = []
                        for item in tokens_data:
                            table_rows.append({
                                "Index": item["index"],
                                "Token": item["token"],
                                "Language": {"hi": "Hindi", "en": "English", "ot": "Other"}[item["language"]],
                                "Script": item["script"],
                                "Confidence": f"{item['confidence']*100:.1f}%",
                                "POS": item["pos"]
                            })
                        df_table = pd.DataFrame(table_rows)
                        st.dataframe(df_table, use_container_width=True, hide_index=True)

                    with col_right:
                        st.markdown("### 📊 Token Confidence Scores")
                        df_chart = pd.DataFrame({
                            "Token": [item["token"] for item in tokens_data],
                            "Confidence (%)": [item["confidence"] * 100 for item in tokens_data],
                            "Language": [{"hi": "Hindi", "en": "English", "ot": "Other"}[item["language"]] for item in tokens_data]
                        })
                        fig_conf = px.bar(
                            df_chart,
                            x="Token",
                            y="Confidence (%)",
                            color="Language",
                            color_discrete_map={"Hindi": "#16a34a", "English": "#2563eb", "Other": "#64748b"},
                            text_auto=".1f",
                            title="Prediction Confidence per Token"
                        )
                        fig_conf.update_layout(yaxis_range=[0, 105], height=340, margin=dict(t=40, b=10, l=10, r=10))
                        st.plotly_chart(fig_conf, use_container_width=True)

                    st.markdown("### 🔄 Code-Switching Analysis")
                    if summary["code_switching"]:
                        st.success(f"Detected **{summary['switch_count']}** language transition(s) in this sentence.")
                        for switch in analysis["switches"]:
                            st.write(f"• Transition from **{switch['from_token']}** (`{switch['from_lang'].upper()}`) ➔ **{switch['to_token']}** (`{switch['to_lang'].upper()}`)")
                    else:
                        st.info("No Hindi-English language code-switching transitions detected.")

                    st.markdown('<div class="section-header">💾 Export Results</div>', unsafe_allow_html=True)
                    e1, e2 = st.columns(2)

                    csv_data = df_table.to_csv(index=False).encode('utf-8')
                    json_data = json.dumps(analysis, indent=2, ensure_ascii=False).encode('utf-8')

                    e1.download_button("⬇️ Export Analysis (CSV)", data=csv_data, file_name="language_detection_analysis.csv", mime="text/csv", use_container_width=True)
                    e2.download_button("⬇️ Export Analysis (JSON)", data=json_data, file_name="language_detection_analysis.json", mime="application/json", use_container_width=True)

                except Exception as err:
                    st.error("❌ Unable to analyze the text.")
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())

elif page == "📊 Model Comparison":
    st.markdown('<div class="section-header">📊 Model Performance Comparison</div>', unsafe_allow_html=True)

    st.markdown("""
    This section evaluates the baseline **TF-IDF + Logistic Regression** model against the fine-tuned **XLM-RoBERTa** Transformer model.
    """)

    comp_df = pd.DataFrame({
        "Model": ["TF-IDF + Logistic Regression (Baseline)", "XLM-RoBERTa (Transformer)"],
        "Architecture": ["Char N-Gram (2-5) + Logistic Regression", "Pre-trained Multilingual Transformer"],
        "Accuracy": ["96.05%", "95.14%"],
        "Macro Precision": ["N/A", "94.52%"],
        "Macro Recall": ["N/A", "95.98%"],
        "Macro F1-Score": ["N/A", "95.20%"],
        "Context Awareness": ["N-gram matches", "Deep Subword Self-Attention"]
    })

    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    df_chart = pd.DataFrame({
        "Metric": ["Accuracy", "Accuracy", "Macro F1"],
        "Model": ["Logistic Regression", "XLM-RoBERTa", "XLM-RoBERTa"],
        "Percentage": [96.05, 95.14, 95.20]
    })

    fig_comp = px.bar(
        df_chart, x="Metric", y="Percentage", color="Model", barmode="group",
        text_auto=".2f",
        title="Accuracy and F1-Score Comparison (%)",
        color_discrete_sequence=["#6366f1", "#10b981"]
    )
    fig_comp.update_layout(yaxis_range=[85, 100], height=350)
    st.plotly_chart(fig_comp, use_container_width=True)

elif page == "ℹ️ About Project":
    st.markdown('<div class="section-header">ℹ️ Project Overview & NLP Pipeline</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🎯 Task & Objectives
    Code-mixing is common in Indian multilingual communities where speakers alternate between Hindi and English within the same sentence or conversation.
    This project provides **Token-Level Language Identification (LID)** and **Sentence-Level Code-Switching Detection**.

    ### 🏗️ End-to-End Pipeline Architecture
    User Input Text ➔ Normalization ➔ Tokenization ➔ XLM-RoBERTa Inference ➔ Code-Switching Analysis ➔ Visual Results
    """)

    st.markdown("""
    ### 📁 Dataset & Models
    - **Dataset**: COMI-LINGUA Hindi-English LID corpus (~596,810 annotated tokens).
    - **Fine-Tuned Model**: XLM-RoBERTa token classification head (`FacebookAI/xlm-roberta-base`).
    - **Baseline Model**: Character 2-5 Gram TF-IDF + Logistic Regression.
    """)

st.divider()
st.caption("Mixed Language Detection & Analysis Project | Computational Linguistics NLP System")
