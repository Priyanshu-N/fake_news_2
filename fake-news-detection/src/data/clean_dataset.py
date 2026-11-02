from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATASET_PATH = (
    PROJECT_ROOT / "data" / "raw" / "WELFake_Dataset.csv"
)

PROCESSED_DATASET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "cleaned_news.csv"
)

REQUIRED_COLUMNS = {"title", "text", "label"}


def clean_dataset() -> None:
    """Clean the dataset and save the resulting CSV."""

    # Step 1: Check that the raw dataset exists.
    if not RAW_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_DATASET_PATH}"
        )

    # Step 2: Load the raw dataset.
    df = pd.read_csv(RAW_DATASET_PATH)

    print("=" * 60)
    print("TRUTHLENS AI: DATA CLEANING")
    print("=" * 60)

    print(f"\nOriginal number of rows: {len(df)}")

    # Step 3: Check required columns.
    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Step 4: Keep only the columns required for this pipeline.
    df = df[["title", "text", "label"]].copy()

    # Step 5: Normalize missing values and remove empty text.
    for column in ["title", "text"]:
        df[column] = df[column].astype("string").str.strip()
        df[column] = df[column].replace("", pd.NA)

    df = df.dropna(subset=["title", "text", "label"])

    # Step 6: Remove exact duplicate records.
    before_deduplication = len(df)

    df = df.drop_duplicates(
        subset=["title", "text", "label"]
    ).copy()

    removed_duplicates = before_deduplication - len(df)

    # Step 7: Validate the labels.
    valid_labels = {0, 1}

    observed_labels = set(df["label"].unique())

    if not observed_labels.issubset(valid_labels):
        raise ValueError(
            f"Unexpected labels found: {observed_labels - valid_labels}"
        )

    # Step 8: Convert labels to integers.
    df["label"] = df["label"].astype(int)

    # Step 9: Save the cleaned dataset.
    PROCESSED_DATASET_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(PROCESSED_DATASET_PATH, index=False)

    # Step 10: Print a cleaning summary.
    print(f"Rows after cleaning: {len(df)}")
    print(f"Duplicate records removed: {removed_duplicates}")
    print(
        f"Rows removed overall: "
        f"{len(pd.read_csv(RAW_DATASET_PATH)) - len(df)}"
    )

    print("\nFinal label distribution:")
    print(df["label"].value_counts().sort_index())

    print(f"\nCleaned dataset saved to:")
    print(PROCESSED_DATASET_PATH)

    print("\nData cleaning completed.")


if __name__ == "__main__":
    clean_dataset()