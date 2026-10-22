from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "WELFake_Dataset.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"


def explore_dataset() -> None:
    """Explore the WELFake dataset and save basic visualizations."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATASET_PATH}"
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load the dataset
    df = pd.read_csv(DATASET_PATH)

    required_columns = {"title", "text", "label"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Dataset must contain these columns: {required_columns}"
        )

    print("=" * 60)
    print("TRUTHLENS AI: EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # 1. Dataset overview
    print("\n1. Dataset Overview")
    print(f"Total articles: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    # 2. Missing values
    print("\n2. Missing Values")
    print(df[["title", "text", "label"]].isnull().sum())

    # 3. Duplicate rows
    print("\n3. Duplicate Rows")
    print(f"Exact duplicates: {df.duplicated().sum()}")

    # 4. Label distribution
    print("\n4. Label Distribution")
    print(df["label"].value_counts(dropna=False))

    print("\nLabel Percentages")
    print(
        (
            df["label"].value_counts(normalize=True, dropna=False)
            * 100
        ).round(2)
    )

    # 5. Article length analysis
    print("\n5. Article Length Statistics")

    text_lengths = df["text"].fillna("").astype(str).str.split().str.len()

    print(text_lengths.describe())

    # 6. Plot label distribution
    label_counts = df["label"].value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    label_counts.plot(kind="bar")

    plt.title("Distribution of Fake and Real News")
    plt.xlabel("Label")
    plt.ylabel("Number of Articles")
    plt.xticks(rotation=0)
    plt.tight_layout()

    chart_path = REPORTS_DIR / "label_distribution.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()

    print(f"\nLabel distribution chart saved to: {chart_path}")

    # 7. Plot article lengths
    plt.figure(figsize=(8, 5))
    text_lengths.clip(upper=text_lengths.quantile(0.99)).hist(bins=50)

    plt.title("Distribution of Article Lengths")
    plt.xlabel("Number of Words")
    plt.ylabel("Number of Articles")
    plt.tight_layout()

    length_chart_path = REPORTS_DIR / "article_lengths.png"
    plt.savefig(length_chart_path, dpi=150)
    plt.close()

    print(f"Article length chart saved to: {length_chart_path}")

    print("\nEDA completed successfully.")


if __name__ == "__main__":
    explore_dataset()