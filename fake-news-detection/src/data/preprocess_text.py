
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DATASET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "cleaned_news.csv"
)

OUTPUT_DATASET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "preprocessed_news.csv"
)


def preprocess_text() -> None:
    """Prepare cleaned news text for NLP feature extraction."""

    # Step 1: Check whether the input file exists.
    if not INPUT_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {INPUT_DATASET_PATH}"
        )

    # Step 2: Load the cleaned dataset.
    df = pd.read_csv(INPUT_DATASET_PATH)

    print("=" * 60)
    print("TRUTHLENS AI: NLP TEXT PREPROCESSING")
    print("=" * 60)

    print(f"\nOriginal number of rows: {len(df)}")

    # Step 3: Validate the required columns.
    required_columns = {"title", "text", "label"}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Step 4: Normalize the title and article text.
    for column in ["title", "text"]:
        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    # Step 5: Combine the title and article text.
    df["combined_text"] = (
        df["title"] + " " + df["text"]
    ).str.strip()

    # Step 6: Remove records with no usable text.
    df = df[df["combined_text"] != ""].copy()

    # Step 7: Save the preprocessed dataset.
    OUTPUT_DATASET_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(OUTPUT_DATASET_PATH, index=False)

    # Step 8: Print the summary.
    print(f"Rows after preprocessing: {len(df)}")

    print("\nExample preprocessed text:")

    if not df.empty:
        print(df["combined_text"].iloc[0][:500])

    print(f"\nSaved preprocessed dataset to:")
    print(OUTPUT_DATASET_PATH)

    print("\nText preprocessing completed.")


if __name__ == "__main__":
    preprocess_text()
