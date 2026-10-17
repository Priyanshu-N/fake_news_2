from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT / "data" / "raw" / "WELFake_Dataset.csv"
)


def validate_dataset() -> None:
    """Validate the news dataset before preprocessing."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    print("\n--- Dataset Validation ---")

    # Check the dataset dimensions.
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    # Inspect the column names.
    print(f"\nColumns: {df.columns.tolist()}")

    # Check required columns.
    required_columns = {"title", "text", "label"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        print(f"\nMissing required columns: {missing_columns}")
        return

    # Check missing values.
    print("\nMissing values:")
    print(df[["title", "text", "label"]].isnull().sum())

    # Check duplicate records.
    print(f"\nExact duplicate rows: {df.duplicated().sum()}")

    # Check class distribution.
    print("\nLabel distribution:")
    print(df["label"].value_counts(dropna=False))

    print("\nLabel percentages:")
    print(
        (df["label"].value_counts(normalize=True) * 100)
        .round(2)
    )

    # Check unexpected labels.
    print("\nUnique labels:")
    print(df["label"].unique())

    print("\nValidation complete.")


if __name__ == "__main__":
    validate_dataset()