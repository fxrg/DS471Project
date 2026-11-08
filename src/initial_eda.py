"""Initial exploratory data analysis for the DS471 project."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_TARGET = "target"


def load_dataset(data_path: str | None) -> tuple[pd.DataFrame, str, str]:
    """Load a local CSV or the scikit-learn copy of the UCI dataset."""
    if data_path:
        path = Path(data_path)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file was not found: {path}")
        return pd.read_csv(path), DEFAULT_TARGET, f"local CSV: {path}"

    from sklearn.datasets import load_breast_cancer

    dataset = load_breast_cancer(as_frame=True)
    frame = dataset.frame.copy()
    frame[DEFAULT_TARGET] = dataset.target
    return frame, DEFAULT_TARGET, "scikit-learn copy of UCI Breast Cancer Wisconsin (Diagnostic)"


def print_initial_eda(frame: pd.DataFrame, target: str, source: str) -> None:
    """Print the required Day 1 dataset inspection and initial observations."""
    print("=" * 72)
    print("DS471 - Initial EDA")
    print(f"Source: {source}")
    print("=" * 72)

    print("\n[1] Shape")
    print(f"Rows: {frame.shape[0]:,}")
    print(f"Columns: {frame.shape[1]:,}")

    print("\n[2] Info")
    frame.info()

    print("\n[3] Missing values")
    missing = frame.isnull().sum()
    print(missing.to_string())
    print(f"Total missing cells: {int(missing.sum()):,}")

    print("\n[4] Duplicate rows")
    print(f"Duplicate rows: {int(frame.duplicated().sum()):,}")

    print("\n[5] Descriptive statistics")
    print(frame.describe(include="all").transpose().to_string())

    print("\n[6] Key variables")
    print(f"Target variable: {target}")
    print(f"Numeric variables: {frame.select_dtypes(include="number").shape[1]:,}")
    print(f"Categorical variables: {frame.select_dtypes(exclude="number").shape[1]:,}")
    if target in frame.columns:
        print("Target distribution:")
        print(frame[target].value_counts(dropna=False).to_string())

    print("\n[7] Initial observations")
    print(f"- The dataset contains {frame.shape[0]:,} observations and "
          f"{frame.shape[1] - 1:,} candidate predictors.")
    print(f"- Missing cells detected: {int(missing.sum()):,}.")
    print(f"- Duplicate rows detected: {int(frame.duplicated().sum()):,}.")
    if target in frame.columns:
        print("- The target distribution should be checked for class imbalance "
              "before selecting evaluation metrics and a split strategy.")
    print("- Outliers and feature scales require visual and statistical checks "
          "before preprocessing decisions are finalized.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        help="Path to a local CSV. If omitted, use the UCI dataset from scikit-learn.",
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help=f"Target column for a local CSV (default: {DEFAULT_TARGET}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame, default_target, source = load_dataset(args.data_path)
    target = args.target if args.data_path else default_target
    if target not in frame.columns:
        raise ValueError(f"Target column '{target}' was not found in the dataset.")
    print_initial_eda(frame, target, source)


if __name__ == "__main__":
    main()
