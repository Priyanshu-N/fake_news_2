from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 1. Initialize FastAPI app
app = FastAPI(
    title="TruthLens AI API 🔍",
    description="Backend API service for Fake News Detection using NLP & Machine Learning.",
    version="1.0.0"
)

# 2. Add CORS Middleware (allows React frontend on port 5173 to communicate with API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Path setup for saved model artifact
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "truthlens_pipeline.joblib"

pipeline = None


@app.on_event("startup")
def load_model_artifact():
    global pipeline
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. "
            "Please run `python -m src.models.train` first!"
        )
    
    print(f"Loading trained pipeline from: {MODEL_PATH}")
    pipeline = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")


# 4. Pydantic schema for request & response validation
class ArticleInput(BaseModel):
    title: str = Field(default="", description="Headline of the news article")
    text: str = Field(..., min_length=10, description="Body content of the news article")


class PredictionOutput(BaseModel):
    label: int
    label_name: str
    confidence: float
    disclaimer: str


# 5. Define API Endpoints
@app.get("/")
def home():
    return {
        "message": "Welcome to TruthLens AI API 🚀. Visit /docs for interactive documentation."
    }


@app.post("/predict", response_model=PredictionOutput)
def predict_news(article: ArticleInput):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model pipeline is not loaded.")

    full_text = f"{article.title} {article.text}".lower().strip()

    # Predict
    prediction = int(pipeline.predict([full_text])[0])
    probabilities = pipeline.predict_proba([full_text])[0]

    # PRINT DEBUG LOGS TO TERMINAL
    print("\n" + "="*50)
    print("DEBUG TEST INPUT:", full_text[:100])
    print("DEBUG PREDICTION INTEGER:", prediction)
    print("DEBUG PROBABILITIES [Class 0, Class 1]:", probabilities)
    print("="*50 + "\n")

    label_names = {1: "Fake News", 0: "Real News"}
    confidence_score = float(probabilities[prediction]) * 100

    return {
        "label": prediction,
        "label_name": label_names.get(prediction, "Unknown"),
        "confidence": round(confidence_score, 2),
        "disclaimer": "Predictions identify learned text patterns and are not independent factual verifications."
    }