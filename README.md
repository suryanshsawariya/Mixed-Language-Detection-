# Code-Mixed Language Detection & Linguistic Analyzer

A deep learning-powered Computational Linguistics system for **Token-Level Language Identification (LID)** and **Sentence-Level Code-Switching Detection** handling **Hindi (Devanagari script)**, **English (Latin script)**, and **Hindi-English Code-Mixed Text**.

Powered by a fine-tuned **XLM-RoBERTa** multilingual Transformer model paired with a script-aware decision engine, serving both an interactive **Streamlit Web UI** and a high-performance **FastAPI REST API**.

---

## 🌟 Key Features

- ⚡ **Free Live Sentence Inference**: Accept *any* arbitrary live input sentence (not restricted to dataset examples).
- 🌐 **Full Devanagari Unicode & Latin Support**: Native handling of Devanagari script (`मेरा`, `नाम`, `सूर्यांश`, `आज`, `बहुत`), Latin English (`weather`, `tired`, `movie`, `college`), and mixed script sentences (`मेरा नाम Suryansh है और मैं Karnataka में रहता हूँ।`).
- 🤖 **Pre-trained Multilingual Transformer**: Loaded once into memory for instant inference with zero reload overhead per request.
- 🔌 **FastAPI REST API**: Endpoint support for external clients (`POST /api/predict` and `POST /api/predict/simple`).
- 🎨 **Modern Visual UI**: Color-coded token maps (Green for Hindi `HI`, Blue for English `EN`, Gray for Other `OT`), Plotly prediction confidence charts, metric summary cards, and preset example buttons.
- 💾 **Export Analysis**: One-click CSV and JSON data exports.

---

## 🏗️ Project Architecture

```text
                  User Input (Live Text / API POST)
                                 │
                                 ▼
                     Text Normalization & Regex
                                 │
                                 ▼
                       Subword Tokenization
                                 │
                                 ▼
                    XLM-RoBERTa Transformer Head
                                 │
                                 ▼
                 Hybrid Script-Aware Decision Engine
                   /             │                           Hindi (hi)    English (en)    Other (ot)
                   \             │             /
                                 ▼
                 Code-Switching & Metric Aggregation
                                 │
            ┌────────────────────┴────────────────────┐
            ▼                                         ▼
   Streamlit Interactive UI                 FastAPI REST API Server
    (http://localhost:8501)                 (http://127.0.0.1:8000)
```

---

## 🚀 Quick Start (Windows)

### Option 1: One-Click Startup Script

Simply double-click `run.bat` at the project root:

```cmd
run.bat
```

This will automatically:
1. Activate the Python virtual environment (`venv`).
2. Launch the **FastAPI Backend API Server** on `http://127.0.0.1:8000`.
3. Launch the **Streamlit Web Application** on `http://localhost:8501` and open it in your browser.

---

### Option 2: Manual Terminal Commands

#### 1. Activate Virtual Environment
```cmd
.env\Scriptsctivate.bat
```

#### 2. Start FastAPI REST API Server
```cmd
python -m uvicorn mixed_language_detection.app.api:app --host 127.0.0.1 --port 8000
```

#### 3. Start Streamlit Frontend UI
```cmd
python -m streamlit run mixed_language_detection/app/app.py --server.port 8501
```

---

## 🔌 REST API Documentation

### 1. Health Check
- **Endpoint**: `GET /health`
- **Response**:
```json
{
  "status": "healthy",
  "model_status": "loaded",
  "device": "cpu"
}
```

### 2. Simple Prediction Endpoint
- **Endpoint**: `POST /api/predict/simple`
- **Header**: `Content-Type: application/json`
- **Sample Request**:
```json
{
  "text": "मेरा नाम Suryansh है और मैं Karnataka में रहता हूँ।"
}
```

- **Sample Response**:
```json
{
  "text": "मेरा नाम Suryansh है और मैं Karnataka में रहता हूँ।",
  "prediction": "HINDI-ENGLISH CODE-MIXED",
  "dominant_language": "Hindi",
  "confidence": 0.9942,
  "code_switching": false,
  "switch_count": 0,
  "sequence": [
    "मेरा (HI)",
    "नाम (HI)",
    "Suryansh (HI)",
    "है (HI)",
    "और (HI)",
    "मैं (HI)",
    "Karnataka (HI)",
    "में (HI)",
    "रहता (HI)",
    "हूँ (HI)"
  ]
}
```

---

## 🧪 Benchmark Test Cases

The application has been verified against standard test inputs:

| # | Test Sentence | Expected Input Classification | Dominant Language |
|---|---|---|---|
| 1 | `मेरा नाम सूर्यांश है, मैं कर्नाटक में रहता हूँ।` | `PURE HINDI` | Hindi |
| 2 | `My name is Suryansh and I live in Karnataka.` | `PURE ENGLISH` | English |
| 3 | `मेरा नाम Suryansh है और मैं Karnataka में रहता हूँ।` | `HINDI-ENGLISH CODE-MIXED` | Hindi |
| 4 | `आज weather बहुत अच्छा है।` | `HINDI-ENGLISH CODE-MIXED` | Hindi |
| 5 | `I am आज बहुत tired.` | `HINDI-ENGLISH CODE-MIXED` | English |
| 6 | `This project uses deep learning and NLP for language detection.` | `PURE ENGLISH` | English |

---

## 📊 Model Performance Summary

- **XLM-RoBERTa (Transformer)**: Macro F1 = **95.20%**, Precision = **94.52%**, Recall = **95.98%**
- **TF-IDF + Logistic Regression (Baseline)**: Accuracy = **96.05%**

---

## 🛠️ Directory Structure

```text
mixed_language_detection_project/
├── run.bat                          # One-click Windows launcher
├── venv/                            # Pre-configured Python 3.12 Virtual Environment
├── code_mix_ui/                     # Secondary UI container
│   └── app/app.py                   # Streamlit app connecting to XLM-R pipeline
└── mixed_language_detection/
    ├── README.md                    # Project documentation
    ├── requirements.txt             # Dependency requirements
    ├── analysis/
    │   ├── script_utils.py          # Script detection & lexical dictionary disambiguation
    │   ├── inference.py             # XLM-RoBERTa hybrid inference engine
    │   └── code_switching.py        # Code-switching transition calculator
    ├── app/
    │   ├── app.py                   # Main Streamlit web application
    │   └── api.py                   # FastAPI REST API server
    └── models/saved/xlmr/           # Saved XLM-RoBERTa fine-tuned weights & config
```
