import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is in Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.inference import load_xlmr_model, analyze_text, DEVICE

# Global variables for cached model & tokenizer
TOKENIZER = None
MODEL = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load XLM-RoBERTa model once during application startup.
    """
    global TOKENIZER, MODEL
    print("Initializing XLM-RoBERTa Model & Tokenizer for FastAPI backend...")
    try:
        TOKENIZER, MODEL = load_xlmr_model.__wrapped__()
        print(f"XLM-RoBERTa Model successfully loaded on device: {DEVICE}")
    except Exception as e:
        print(f"Error loading model: {e}")
        TOKENIZER, MODEL = None, None
    yield
    print("Shutting down FastAPI backend server...")


app = FastAPI(
    title="Mixed Language Detection API",
    description="REST API for Token-level Language Identification and Code-Switching Analysis using XLM-RoBERTa",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str = Field(..., description="Text sentence to analyze", example="मेरा नाम Suryansh है और मैं Karnataka में रहता हूँ।")


@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "Mixed Language Detection & Linguistic Analysis API",
        "version": "1.0.0",
        "docs_url": "/docs"
    }


@app.get("/health")
async def health_check():
    model_status = "loaded" if MODEL is not None else "not_loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "device": str(DEVICE)
    }


@app.post("/api/predict")
async def predict_full(request: PredictRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    if MODEL is None or TOKENIZER is None:
        raise HTTPException(status_code=500, detail="XLM-R model is not initialized.")

    try:
        analysis = analyze_text(request.text.strip(), TOKENIZER, MODEL)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/api/predict/simple")
async def predict_simple(request: PredictRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    if MODEL is None or TOKENIZER is None:
        raise HTTPException(status_code=500, detail="XLM-R model is not initialized.")

    try:
        analysis = analyze_text(request.text.strip(), TOKENIZER, MODEL)
        summary = analysis["summary"]
        return {
            "text": request.text.strip(),
            "prediction": summary["input_type"],
            "dominant_language": summary["dominant_lang"],
            "confidence": round(summary["avg_confidence"] / 100.0, 4),
            "code_switching": summary["code_switching"],
            "switch_count": summary["switch_count"],
            "sequence": analysis["sequence"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host="127.0.0.1", port=8000, reload=False)
