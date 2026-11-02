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

# 2. Add CORS Middleware (Allows our React frontend to make web requests to FastAPI)
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

# Load model during server startup
@app.on_event("startup")
def load_model_artifact():
    global pipeline
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found at {MODEL_PATH}. Please run `python -m src.models.train` first!")
    
    print(f"Loading trained pipeline from: {MODEL_PATH}")
    pipeline = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")


# 4. Define data structures (Pydantic models for request validation)
class ArticleInput(BaseModel):
    title: str = Field(default="", description="Headline of the article")
    text: str = Field(..., min_length=10, description="Body content of the article")


class PredictionOutput(BaseModel):
    label: int
    label_name: str
    confidence: float
    disclaimer: str


# 5. Define API Endpoints
@app.get("/")
def home():
    return {"message": "Welcome to TruthLens AI API 🚀. Visit /docs to test endpoints!"}


@app.post("/predict", response_model=PredictionOutput)
def predict_news(article: ArticleInput):
    """Takes headline and article text, returns prediction with confidence score."""
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model pipeline is not loaded.")

    # Combine title and text exactly as done during training
    full_text = f"{article.title} {article.text}".strip()

    # Get binary prediction (0 = Fake, 1 = Real)
    prediction = int(pipeline.predict([full_text])[0])

    # Get probability score (confidence percentage)
    probabilities = pipeline.predict_proba([full_text])[0]
    confidence_score = float(probabilities[prediction]) * 100

    label_names = {0: "Fake News", 1: "Real News"}

    return {
        "label": prediction,
        "label_name": label_names.get(prediction, "Unknown"),
        "confidence": round(confidence_score, 2),
        "disclaimer": "Predictions identify learned text patterns and are not independent factual verifications."
    }