from pathlib import Path

import pandas as pd


# Locate the dataset relative to the project directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT / "data" / "raw" / "WELFake_Dataset.csv"
)


def inspect_dataset(file_path: Path) -> None:
    """Inspect the dataset without modifying the original file."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {file_path}"
        )

    print("=" * 60)
    print("FAKE NEWS DATASET INSPECTION")
    print("=" * 60)

    # Load the CSV file.
    df = pd.read_csv(file_path)

    # 1. Dataset dimensions
    print("\n1. Dataset Shape")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    # 2. Column names
    print("\n2. Column Names")
    print(df.columns.tolist())

    # 3. First five records
    print("\n3. First Five Records")
    print(df.head().to_string())

    # 4. Data types
    print("\n4. Column Data Types")
    print(df.dtypes)

    # 5. Missing values
    print("\n5. Missing Values")
    print(df.isnull().sum())

    # 6. Duplicate records
    print("\n6. Duplicate Records")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    # 7. Label distribution
    print("\n7. Label Distribution")

    if "label" in df.columns:
        print(df["label"].value_counts(dropna=False))
        print("\nLabel Distribution (Percentages)")
        print(
            (df["label"].value_counts(normalize=True) * 100)
            .round(2)
        )
    else:
        print("No column named 'label' was found.")

    print("\nInspection complete.")


if __name__ == "__main__":
    inspect_dataset(DATASET_PATH)