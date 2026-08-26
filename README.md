# 🌐 Code-Mixed Hindi-English Language Detection & Linguistic Analyzer

A Transformer-based Computational Linguistics & NLP system for token-level Hindi-English Language Identification (LID), script awareness, code-switching analysis, and morphological feature extraction.

---

## 📌 Project Overview
This project addresses the challenge of **Hindi-English Code-Mixed (Hinglish)** language processing in informal digital communications. Standard single-language NLP systems fail on code-mixed text due to spelling variations, Romanized script usage, and absence of standardized grammar rules.

### Key Capabilities:
- **Hindi-English Code-Mixed Analysis**: `"Kal college jaana hai but assignment complete nahi hua."`
- **Pure Devanagari Hindi Analysis**: `"मेरा नाम सूर्यांश है, मैं कर्नाटक में रहता हूँ。"`
- **Pure English Analysis**: `"Today I have a college presentation."`
- **Romanized Hindi Analysis**: `"mera naam suryansh hai aur main karnataka mein rehta hoon"`
- **Mixed Script Analysis**: `"मेरा college आज बहुत अच्छा था。"`
- **Non-Linguistic Token Handling**: Safe handling of numbers, punctuation, emojis, URLs, and usernames without false code-switch triggers.

---

## 🔄 System Workflow Diagram

```
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
```

---

## 🏗️ Project Architecture & Structure

```
mixed_language_detection/
│
├── app/
│   └── app.py                             # Professional Streamlit UI Dashboard
│
├── analysis/
│   ├── __init__.py
│   ├── script_utils.py                    # Script awareness, Devanagari regex, POS tagging
│   ├── code_switching.py                  # Code-switching transition metrics
│   └── inference.py                       # Subword aggregation & XLM-R inference engine
│
├── preprocessing/
│   ├── __init__.py
│   └── prepare_lid_dataset.py             # COMI-LINGUA 3-annotator majority voting
│
├── training/
│   ├── __init__.py
│   ├── explore_dataset.py                 # Dataset EDA
│   ├── train_baseline.py                  # TF-IDF + Logistic Regression Baseline
│   └── train_xlmr.py                      # XLM-RoBERTa Token Classifier PyTorch training
│
├── data/
│   ├── raw/                               # LID_train.csv, LID_test.csv
│   └── processed/                         # LID_train_processed.csv, LID_test_processed.csv
│
├── models/
│   └── saved/
│       ├── baseline_logistic_regression.joblib
│       └── xlmr/                          # Fine-tuned XLM-RoBERTa PyTorch safetensors
│
├── run_app.bat                            # Quick launch batch file
├── requirements.txt                       # Python dependencies
└── README.md                              # Documentation
```

---

## 📊 Models & Benchmarks

| Model Architecture | Features | Accuracy | Macro F1 | Status |
| :--- | :--- | :--- | :--- | :--- |
| **TF-IDF + Logistic Regression** | Char N-gram (2-5), Sublinear TF | **96.05%** | Baseline | Saved in `models/saved/` |
| **XLM-RoBERTa (Transformer)** | Multilingual Contextual Embeddings | **95.14%** | **95.20%** | Saved in `models/saved/xlmr/` |

---

## 🚀 Running the Application

### 1. Activate Environment & Run Streamlit UI
```powershell
..\venv\Scripts\streamlit.exe run app/app.py
```
Or execute:
```powershell
run_app.bat
```

---

## 🔬 Testing & Verification

Run all test cases:
1. `Kal college jaana hai but assignment complete nahi hua.` → Code-Mixed (YES)
2. `मेरा नाम सूर्यांश है, मैं कर्नाटक में रहता हूँ。` → Pure Hindi (NO)
3. `Today I have a college presentation.` → Pure English (NO)
4. `मेरा college आज बहुत अच्छा था。` → Code-Mixed (YES)
5. `mera naam suryansh hai aur main karnataka mein rehta hoon` → Pure Hindi / Romanized (NO)
6. `12345 !!! ,,,` → Other / Non-Linguistic (NO)
7. `आज college में assignment submit करना है。` → Code-Mixed (YES)
