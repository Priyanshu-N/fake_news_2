from pathlib import Path
import pandas as pd

# Define paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "WELFake_Dataset.csv"


def inspect_dataset(file_path: Path) -> None:
    """Inspect the raw dataset without modifying original data."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {file_path}. "
            "Please ensure WELFake_Dataset.csv is placed inside data/raw/"
        )

    print("=" * 60)
    print("TRUTHLENS AI: DATASET INSPECTION")
    print("=" * 60)

    # Load dataset
    df = pd.read_csv(file_path)

    print("\n1. Dataset Dimensions")
    print(f"Total Rows:    {df.shape[0]:,}")
    print(f"Total Columns: {df.shape[1]}")

    print("\n2. Column Names")
    print(df.columns.tolist())

    print("\n3. First 5 Records")
    print(df.head().to_string())

    print("\n4. Column Data Types")
    print(df.dtypes)

    print("\n5. Missing Values per Column")
    print(df.isnull().sum())

    print("\n6. Duplicate Records")
    print(f"Exact Duplicate Rows: {df.duplicated().sum():,}")

    print("\n7. Label Distribution")
    if "label" in df.columns:
        print(df["label"].value_counts(dropna=False))
        print("\nLabel Proportions (%):")
        print((df["label"].value_counts(normalize=True) * 100).round(2))
    else:
        print("❌ Warning: No 'label' column found in dataset.")

    print("\nInspection complete.")


if __name__ == "__main__":
    inspect_dataset(DATASET_PATH)