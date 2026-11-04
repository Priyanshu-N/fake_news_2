from pathlib import Path
import pandas as pd

# Define paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEANED_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_news.csv"
PREPROCESSED_DATASET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "preprocessed_news.csv"
)


def preprocess_text() -> None:
    """Normalize text data and combine headline and body content."""

    if not CLEANED_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset missing at: {CLEANED_DATASET_PATH}. "
            "Please run `python -m src.data.clean_dataset` first!"
        )

    print("=" * 60)
    print("TRUTHLENS AI: TEXT PREPROCESSING")
    print("=" * 60)

    # 1. Load cleaned data
    df = pd.read_csv(CLEANED_DATASET_PATH)
    print(f"Loaded {len(df):,} cleaned records.")

    # 2. Fill missing strings safely
    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)

    # 3. Create unified normalized text field
    # Convert to lowercase and strip whitespace
    df["title_norm"] = df["title"].str.lower().str.strip()
    df["text_norm"] = df["text"].str.lower().str.strip()

    # Concatenate title and text into a single combined feature column
    df["combined_text"] = (df["title_norm"] + " " + df["text_norm"]).str.strip()

    # 4. Filter out any empty combined strings
    df = df[df["combined_text"] != ""].copy()

    # 5. Select final output columns
    output_df = df[["title", "text", "combined_text", "label"]]

    # 6. Save output file
    PREPROCESSED_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(PREPROCESSED_DATASET_PATH, index=False)

    print(f"Preprocessed records count: {len(output_df):,}")
    print(f"Saved preprocessed file to: {PREPROCESSED_DATASET_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    preprocess_text()