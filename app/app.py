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
    /* Global Fonts & Colors */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .hero-container {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        padding: 30px 40px;
        border-radius: 16px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    }
    
    .hero-title {
        font-size: 38px;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(90deg, #ffffff, #c7d2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .hero-subtitle {
        font-size: 17px;
        color: #e0e7ff;
        margin-top: 8px;
        font-weight: 400;
    }
    
    .metric-card-custom {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .metric-card-custom:hover {
        border-color: #6366f1;
        transform: translateY(-2px);
    }
    
    .metric-val {
        font-size: 28px;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 4px;
    }
    
    .metric-lbl {
        font-size: 13px;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .token-badge {
        display: inline-flex;
        align-items: center;
        padding: 6px 12px;
        margin: 4px 5px;
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
        font-size: 10px;
        opacity: 0.85;
        margin-left: 6px;
        padding: 2px 4px;
        border-radius: 4px;
        background: rgba(0,0,0,0.25);
    }

    .workflow-box {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        color: #e2e8f0;
    }
    
    .section-header {
        font-size: 22px;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 30px;
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
        st.warning("⚠️ **CPU Mode Active**")
        
    st.divider()
    st.markdown("### 📊 Model Metrics Summary")
    st.markdown("""
    - **XLM-R Model**: Macro F1 = **95.20%**
    - **Baseline (TF-IDF + LR)**: Accuracy = **96.05%**
    - **Training Dataset**: 90,000 Balanced Tokens
    - **Classes**: Hindi / English / Other
    """)
    
    st.divider()
    st.caption("Mixed Language Detection & Analysis Project | Computational Linguistics")

# ============================================================
# HEADER HERO
# ============================================================

st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">🌐 Code-Mixed Language Detection & Linguistic Analyzer</div>
        <div class="hero-subtitle">Transformer-based Hindi-English Language Identification and Code-Switching Analysis using XLM-RoBERTa</div>
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

    st.markdown('<div class="section-header">✍️ Input Text & Examples</div>', unsafe_allow_html=True)
    
    # Initialize session state for text input if missing
    if "input_text" not in st.session_state:
        st.session_state["input_text"] = "Kal college jaana hai but assignment complete nahi hua."

    # Example Preset Buttons
    col_ex1, col_ex2, col_ex3, col_ex4, col_ex5 = st.columns(5)
    
    if col_ex1.button("Ex 1: Code-Mixed 1", use_container_width=True):
        st.session_state["input_text"] = "Kal college jaana hai but assignment complete nahi hua."
    if col_ex2.button("Ex 2: Pure Devanagari", use_container_width=True):
        st.session_state["input_text"] = "मेरा नाम सूर्यांश है, मैं कर्नाटक में रहता हूँ।"
    if col_ex3.button("Ex 3: Devanagari + English", use_container_width=True):
        st.session_state["input_text"] = "मेरा college आज बहुत अच्छा था।"
    if col_ex4.button("Ex 4: English + Romanized", use_container_width=True):
        st.session_state["input_text"] = "Today mera college mein presentation hai."
    if col_ex5.button("Ex 5: Pure Romanized", use_container_width=True):
        st.session_state["input_text"] = "mera naam suryansh hai aur main karnataka mein rehta hoon"

    text_input = st.text_area(
        "Enter text for language identification and linguistic analysis:",
        value=st.session_state["input_text"],
        height=120,
        placeholder="Type or paste Hindi-English text here..."
    )

    col_btn, _ = st.columns([1, 4])
    analyze_click = col_btn.button("🔍 Analyze Text", type="primary", use_container_width=True)

    if analyze_click or text_input:
        if not text_input.strip():
            st.warning("⚠️ Please enter some text to analyze.")
            st.stop()
            
        try:
            with st.spinner("Analyzing text with XLM-RoBERTa..."):
                analysis = analyze_text(text_input, tokenizer, model)
                
            tokens_data = analysis["tokens"]
            summary = analysis["summary"]
            switches = analysis["switches"]
            script_dist = analysis["script_dist"]
            
            if not tokens_data:
                st.warning("No valid tokens found in input.")
                st.stop()

            # ----------------------------------------------------
            # SUMMARY DASHBOARD CARDS
            # ----------------------------------------------------
            st.markdown('<div class="section-header">📊 Summary Dashboard</div>', unsafe_allow_html=True)
            
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            
            with m1:
                st.markdown(f'<div class="metric-card-custom"><div class="metric-lbl">Total Words</div><div class="metric-val">{summary["total_words"]}</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-card-custom"><div class="metric-lbl">Hindi Words</div><div class="metric-val" style="color:#22c55e;">{summary["hi_count"]} ({summary["hi_pct"]}%)</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric-card-custom"><div class="metric-lbl">English Words</div><div class="metric-val" style="color:#3b82f6;">{summary["en_count"]} ({summary["en_pct"]}%)</div></div>', unsafe_allow_html=True)
            with m4:
                st.markdown(f'<div class="metric-card-custom"><div class="metric-lbl">Dominant Lang</div><div class="metric-val" style="font-size:20px;">{summary["dominant_lang"]}</div></div>', unsafe_allow_html=True)
            with m5:
                switch_str = f"YES ({summary['switch_count']})" if summary["code_switching"] else "NO"
                switch_color = "#f59e0b" if summary["code_switching"] else "#10b981"
                st.markdown(f'<div class="metric-card-custom"><div class="metric-lbl">Code-Switching</div><div class="metric-val" style="color:{switch_color}; font-size:20px;">{switch_str}</div></div>', unsafe_allow_html=True)
            with m6:
                st.markdown(f'<div class="metric-card-custom"><div class="metric-lbl">Avg Confidence</div><div class="metric-val" style="color:#a855f7;">{summary["avg_confidence"]}%</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.info(f"🏷️ **Input Type Classification:** `{summary['input_type']}`")

            # ----------------------------------------------------
            # VISUAL LANGUAGE MAP
            # ----------------------------------------------------
            st.markdown('<div class="section-header">🎨 Visual Token Language Map</div>', unsafe_allow_html=True)
            
            badge_htmls = []
            for item in tokens_data:
                lang = item["language"]
                tok = item["token"]
                conf = item["confidence"] * 100
                cls_name = f"token-{lang}"
                lang_name = "Hindi" if lang == "hi" else ("English" if lang == "en" else "Other")
                
                badge_htmls.append(
                    f'<span class="token-badge {cls_name}" title="{lang_name} | Confidence: {conf:.1f}% | Script: {item["script"]}">'
                    f'{tok}<span class="token-sub">{lang_name.upper()} {conf:.0f}%</span>'
                    f'</span>'
                )
                
            st.markdown("<div>" + "".join(badge_htmls) + "</div>", unsafe_allow_html=True)

            # ----------------------------------------------------
            # LANGUAGE & SCRIPT DISTRIBUTION CHARTS
            # ----------------------------------------------------
            st.markdown('<div class="section-header">📈 Linguistic Distribution & Script Breakdown</div>', unsafe_allow_html=True)
            
            c_chart1, c_chart2 = st.columns(2)
            
            with c_chart1:
                df_lang = pd.DataFrame({
                    "Language": ["Hindi", "English", "Other"],
                    "Count": [summary["hi_count"], summary["en_count"], summary["ot_count"]]
                })
                fig_lang = px.pie(
                    df_lang, values="Count", names="Language",
                    title="Language Distribution",
                    color="Language",
                    color_discrete_map={"Hindi": "#16a34a", "English": "#2563eb", "Other": "#64748b"},
                    hole=0.4
                )
                fig_lang.update_layout(margin=dict(t=40, b=10, l=10, r=10), height=300)
                st.plotly_chart(fig_lang, use_container_width=True)

            with c_chart2:
                df_script = pd.DataFrame({
                    "Script": ["Devanagari", "Latin", "Other/Punctuation"],
                    "Percentage": [script_dist["devanagari"], script_dist["latin"], script_dist["other"]]
                })
                fig_script = px.pie(
                    df_script, values="Percentage", names="Script",
                    title="Script Breakdown (%)",
                    color="Script",
                    color_discrete_map={"Devanagari": "#ea580c", "Latin": "#0284c7", "Other/Punctuation": "#64748b"},
                    hole=0.4
                )
                fig_script.update_layout(margin=dict(t=40, b=10, l=10, r=10), height=300)
                st.plotly_chart(fig_script, use_container_width=True)

            # ----------------------------------------------------
            # TOKEN LEVEL TABLE WITH POS & MORPHOLOGY
            # ----------------------------------------------------
            st.markdown('<div class="section-header">🔤 Detailed Token-Level Analysis</div>', unsafe_allow_html=True)
            
            table_rows = []
            for item in tokens_data:
                table_rows.append({
                    "#": item["index"],
                    "Token": item["token"],
                    "Language": "Hindi" if item["language"] == "hi" else ("English" if item["language"] == "en" else "Other"),
                    "Confidence": f"{item['confidence']*100:.2f}%",
                    "Script": item["script"],
                    "POS Tag": item["pos"],
                    "Morphological Analysis": item["morphology"]
                })
                
            df_table = pd.DataFrame(table_rows)
            st.dataframe(df_table, use_container_width=True, hide_index=True)

            # ----------------------------------------------------
            # CODE SWITCHING ANALYSIS
            # ----------------------------------------------------
            st.markdown('<div class="section-header">🔄 Code-Switching Analysis</div>', unsafe_allow_html=True)
            
            if summary["code_switching"]:
                st.success(f"✅ **Code-Switching Detected**: Found **{summary['switch_count']}** switching points in sentence structure.")
                
                # Language Sequence
                st.markdown(f"**Language Flow Sequence:** `{' → '.join(analysis['sequence'])}`")
                
                # Switch Table
                sw_rows = []
                for sw in switches:
                    sw_rows.append({
                        "Token Position": sw["position"],
                        "From Language": "Hindi" if sw["from"] == "hi" else "English",
                        "To Language": "Hindi" if sw["to"] == "hi" else "English",
                        "Switch Token": sw["token"]
                    })
                df_sw = pd.DataFrame(sw_rows)
                st.dataframe(df_sw, use_container_width=True, hide_index=True)
            else:
                st.info("ℹ️ **No Code-Switching Detected**: Input is single-language or contains no Hindi-English transition points.")

            # ----------------------------------------------------
            # CONFIDENCE BAR CHART
            # ----------------------------------------------------
            st.markdown('<div class="section-header">🎯 Per-Token Model Confidence</div>', unsafe_allow_html=True)
            
            df_conf = pd.DataFrame({
                "Token": [f"{item['index']}. {item['token']}" for item in tokens_data],
                "Confidence (%)": [item["confidence"] * 100 for item in tokens_data],
                "Language": ["Hindi" if item["language"] == "hi" else ("English" if item["language"] == "en" else "Other") for item in tokens_data]
            })
            
            fig_conf = px.bar(
                df_conf, x="Token", y="Confidence (%)", color="Language",
                color_discrete_map={"Hindi": "#16a34a", "English": "#2563eb", "Other": "#64748b"},
                text_auto=".1f",
                title="Word Prediction Confidence Scores"
            )
            fig_conf.update_layout(yaxis_range=[0, 105], height=320, margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig_conf, use_container_width=True)

            # ----------------------------------------------------
            # DOWNLOAD / EXPORT RESULTS
            # ----------------------------------------------------
            st.markdown('<div class="section-header">💾 Export Analysis</div>', unsafe_allow_html=True)
            
            c_dl1, c_dl2 = st.columns(2)
            
            csv_data = df_table.to_csv(index=False).encode('utf-8')
            json_data = json.dumps(analysis, indent=2, ensure_ascii=False).encode('utf-8')
            
            c_dl1.download_button(
                "⬇️ Download CSV Analysis",
                data=csv_data,
                file_name="code_mixed_language_analysis.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            c_dl2.download_button(
                "⬇️ Download JSON Analysis",
                data=json_data,
                file_name="code_mixed_language_analysis.json",
                mime="application/json",
                use_container_width=True
            )

        except Exception as err:
            st.error("❌ Unable to analyze the text. Please check input and try again.")
            with st.expander("Technical Details"):
                st.code(traceback.format_exc())

# ============================================================
# PAGE 2: MODEL COMPARISON
# ============================================================

elif page == "📊 Model Comparison":
    st.markdown('<div class="section-header">📊 Baseline vs. XLM-RoBERTa Model Comparison</div>', unsafe_allow_html=True)
    
    st.markdown("""
    This section evaluates the performance of the **TF-IDF + Logistic Regression** baseline model against the fine-tuned **XLM-RoBERTa** Transformer model.
    """)
    
    comp_df = pd.DataFrame({
        "Model": ["TF-IDF + Logistic Regression", "XLM-RoBERTa (Transformer)"],
        "Architecture": ["Char N-Gram (2-5) + Logistic Regression", "Pre-trained Multilingual Transformer"],
        "Accuracy": ["96.05%", "95.14%"],
        "Precision (Macro)": ["N/A", "94.52%"],
        "Recall (Macro)": ["N/A", "95.98%"],
        "Macro F1": ["N/A", "95.20%"],
        "Training Samples": ["Full Token Dataset", "90,000 Balanced Tokens"]
    })
    
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Comparison Chart
    df_chart = pd.DataFrame({
        "Metric": ["Accuracy", "Accuracy", "Macro F1"],
        "Model": ["Logistic Regression", "XLM-RoBERTa", "XLM-RoBERTa"],
        "Percentage": [96.05, 95.14, 95.20]
    })
    
    fig_comp = px.bar(
        df_chart, x="Metric", y="Percentage", color="Model", barmode="group",
        text_auto=".2f",
        title="Model Performance Comparison (%)",
        color_discrete_sequence=["#6366f1", "#10b981"]
    )
    fig_comp.update_layout(yaxis_range=[85, 100], height=350)
    st.plotly_chart(fig_comp, use_container_width=True)
    
    st.warning("""
    > ⚠️ **Evaluation Protocol Note:**  
    > The baseline Logistic Regression model was evaluated on the raw token sequence, whereas the XLM-RoBERTa Transformer model was evaluated on a balanced 90,000 subword-token dataset. Therefore, accuracy numbers should be interpreted in context of their evaluation protocols. XLM-RoBERTa provides contextual awareness critical for ambiguous Romanized Hindi tokens.
    """)

# ============================================================
# PAGE 3: ABOUT PROJECT
# ============================================================

elif page == "ℹ️ About Project":
    st.markdown('<div class="section-header">ℹ️ Research & Project Documentation</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Problem Statement
    Code-mixing—the alternating use of two or more languages in a single sentence or conversation—is prevalent across social media and informal digital communication in India (e.g., Hindi-English / Hinglish). Standard single-language NLP models struggle to accurately process code-mixed data due to spelling variations, Romanized script usage, and absence of standardized grammar rules.

    ### 🔄 System Workflow Pipeline
    """)
    
    # Workflow Box
    st.markdown(
        """
        <div class="workflow-box">
        <pre style="color: #e2e8f0; font-family: monospace; font-size: 14px; line-height: 1.5;">
                 User Input Text
                       │
                       ▼
               Text Normalization
                       │
                       ▼
                 Tokenization
                       │
                       ▼
            Language Identification
               /       │       \ 
          Hindi     English    Other
               \       │       /
                       ▼
                  POS Tagging
                       │
                       ▼
             Morphological Analysis
                       │
                       ▼
            Code-Mixing Analysis
                       │
      ┌────────────────┼────────────────┐
      ▼                ▼                ▼
 Language %       Mixing Type      Linguistic Features
      └────────────────┼────────────────┘
                       │
                       ▼
              Final Visualization
        </pre>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("""
    ### 📁 Dataset & Preprocessing
    - **Dataset**: COMI-LINGUA Hindi-English LID corpus.
    - **Token Count**: Approximately 596,810 annotated tokens.
    - **Annotators**: 3 human annotators per sentence.
    - **Majority Voting**: Normalized labels using strict majority voting (requiring agreement from at least 2 annotators) into 3 final classes (`hi` = Hindi, `en` = English, `ot` = Other).

    ### 🤖 Trained Models
    1. **XLM-RoBERTa**: Fine-tuned `FacebookAI/xlm-roberta-base` token classification head trained on GPU with mixed precision (FP16).
    2. **Baseline Logistic Regression**: Character n-gram (2-5) TF-IDF vectorizer paired with weighted Logistic Regression classifier.

    ### 🚀 Future Scope
    - Expansion to additional Indian regional languages (Tamil, Telugu, Bengali, Marathi, Gujarati).
    - Multilingual code-switching (e.g., Hindi-English-Tamil tri-lingual text).
    - POS Tagging and Named Entity Recognition (NER) for code-mixed sentences.
    - Social media sentiment analysis and intent detection on code-mixed posts.
    - Speech-to-Text code-mixed audio processing pipeline integration.
    """)

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption("Mixed Language Detection & Analysis | Computational Linguistics NLP Project | XLM-RoBERTa & PyTorch CUDA")