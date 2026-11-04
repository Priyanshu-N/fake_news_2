from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Define project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "preprocessed_news.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_SAVE_PATH = MODEL_DIR / "truthlens_pipeline.joblib"


def train_model() -> None:
    """Train TF-IDF + Logistic Regression pipeline and export model artifact."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing dataset at {DATA_PATH}. "
            "Please run `python -m src.data.preprocess_text` first!"
        )

    print("=" * 60)
    print("TRUTHLENS AI: MODEL TRAINING & EVALUATION")
    print("=" * 60)

    # 1. Load data
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["combined_text", "label"])

    X = df["combined_text"].astype(str)
    y = df["label"].astype(int)

    # 2. Train/Test split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set: {len(X_train):,} samples")
    print(f"Testing set:  {len(X_test):,} samples\n")

    # 3. Create pipeline (TF-IDF + Logistic Regression)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=2.0,
            random_state=42
        ))
    ])

    # 4. Train pipeline
    print("Training model pipeline...")
    pipeline.fit(X_train, y_train)

    # 5. Evaluate on unseen test data
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=["Fake (0)", "Real (1)"]
    )
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 60)
    print(f"Model Accuracy: {acc * 100:.2f}%")
    print("=" * 60)
    print("\nClassification Report:\n", report)
    print("Confusion Matrix:\n", cm)

    # 6. Save model pipeline
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_SAVE_PATH)

    print("\n" + "=" * 60)
    print(f"Saved trained pipeline artifact to: {MODEL_SAVE_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    train_model()