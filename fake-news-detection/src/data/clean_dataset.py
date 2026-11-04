from pathlib import Path
import pandas as pd

# Define paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "WELFake_Dataset.csv"
PROCESSED_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_news.csv"

REQUIRED_COLUMNS = {"title", "text", "label"}


def clean_dataset() -> None:
    """Clean the raw WELFake dataset and save sanitized output."""

    # 1. Check raw file exists
    if not RAW_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset missing at: {RAW_DATASET_PATH}. "
            "Please ensure WELFake_Dataset.csv exists in data/raw/"
        )

    print("=" * 60)
    print("TRUTHLENS AI: DATA CLEANING")
    print("=" * 60)

    # 2. Read raw CSV
    df = pd.read_csv(RAW_DATASET_PATH)
    initial_rows = len(df)
    print(f"Original row count: {initial_rows:,}")

    # 3. Validate columns
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required dataset columns: {missing_cols}")

    # 4. Keep required columns only
    df = df[["title", "text", "label"]].copy()

    # 5. Trim whitespace and replace empty strings with NA
    for col in ["title", "text"]:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace("", pd.NA)

    # 6. Drop missing entries
    df = df.dropna(subset=["title", "text", "label"])

    # 7. Deduplicate exact matching articles
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["title", "text", "label"]).copy()
    removed_dups = before_dedup - len(df)

    # 8. Validate binary labels
    valid_labels = {0, 1}
    observed_labels = set(df["label"].unique())
    if not observed_labels.issubset(valid_labels):
        raise ValueError(f"Unexpected label values found: {observed_labels - valid_labels}")

    df["label"] = df["label"].astype(int)

    # 9. Save clean output
    PROCESSED_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATASET_PATH, index=False)

    # 10. Output cleaning summary
    print(f"Duplicates removed: {removed_dups:,}")
    print(f"Total rows removed overall: {initial_rows - len(df):,}")
    print(f"Final clean row count: {len(df):,}")
    print("\nClean Label Distribution:")
    print(df["label"].value_counts().sort_index())
    print(f"\nSaved clean file to: {PROCESSED_DATASET_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    clean_dataset()