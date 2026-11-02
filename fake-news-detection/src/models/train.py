import os
from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Locate project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "preprocessed_news.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_SAVE_PATH = MODEL_DIR / "truthlens_pipeline.joblib"


def train_model():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset at {DATA_PATH}. Run preprocess_text.py first!")

    print("=" * 60)
    print("TRUTHLENS AI: MODEL TRAINING & EVALUATION")
    print("=" * 60)

    # 1. Load data
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["combined_text", "label"])

    X = df["combined_text"]
    y = df["label"].astype(int)

    # 2. Split data: 80% Training, 20% Testing
    # 'stratify=y' ensures equal fake/real proportions in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training on: {len(X_train):,} samples")
    print(f"Testing on:  {len(X_test):,} samples\n")

    # 3. Define classifiers to benchmark
    candidate_models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=2.0, random_state=42)
    }

    best_pipeline = None
    best_f1 = 0.0
    best_name = ""

    # 4. Train and evaluate each model inside a Scikit-Learn Pipeline
    for name, classifier in candidate_models.items():
        print(f"--- Training {name} ---")

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=50000,
                ngram_range=(1, 2),  # Extract unigrams + bigrams
                sublinear_tf=True
            )),
            ("clf", classifier)
        ])

        # Fit vectorizer + model ONLY on training data to prevent data leakage
        pipeline.fit(X_train, y_train)

        # Make predictions on test data
        y_pred = pipeline.predict(X_test)

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=["Fake (0)", "Real (1)"], output_dict=True)
        macro_f1 = report["macro avg"]["f1-score"]

        print(f"Accuracy: {acc * 100:.2f}%")
        print(f"Macro F1-Score: {macro_f1:.4f}")
        print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Fake (0)", "Real (1)"]))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("-" * 50)

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_pipeline = pipeline
            best_name = name

    # 5. Save winning pipeline artifact for FastAPI backend
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_SAVE_PATH)

    print("\n" + "=" * 60)
    print(f"🏆 BEST MODEL: {best_name} (F1: {best_f1:.4f})")
    print(f"Saved model pipeline artifact to: {MODEL_SAVE_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    train_model()