"""Build the leakage-safe preprocessing datasets for the DS471 project."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "breast_cancer.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
TARGET_COLUMN = "diagnosis"
ID_COLUMN = "id"
RANDOM_STATE = 42


def load_and_validate(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load WDBC CSV and reject missing, malformed, or anomalous records."""
    if not path.is_file():
        raise FileNotFoundError(f"Input dataset was not found: {path}")

    frame = pd.read_csv(path)
    required = {ID_COLUMN, TARGET_COLUMN}
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    feature_columns = [
        column for column in frame.columns if column not in {ID_COLUMN, TARGET_COLUMN}
    ]
    if len(feature_columns) != 30:
        raise ValueError(f"Expected 30 feature columns, found {len(feature_columns)}")
    if frame.empty:
        raise ValueError("The input dataset is empty.")
    if frame[ID_COLUMN].isna().any() or not frame[ID_COLUMN].is_unique:
        raise ValueError("The identifier column must be complete and unique.")
    if frame.isna().any().any():
        null_counts = frame.isna().sum()
        raise ValueError(
            f"Missing values detected: {null_counts[null_counts > 0].to_dict()}"
        )

    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype(str).str.strip()
    labels = set(frame[TARGET_COLUMN].unique())
    if not labels <= {"M", "B"} or labels != {"M", "B"}:
        raise ValueError(f"Expected target labels {{'M', 'B'}}, found {sorted(labels)}")

    numeric_features = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    if numeric_features.isna().any().any():
        raise ValueError("One or more feature values are not numeric.")
    if not np.isfinite(numeric_features.to_numpy()).all():
        raise ValueError("One or more feature values are not finite.")
    if frame.duplicated().any():
        raise ValueError("Duplicate rows detected in the input dataset.")

    frame[feature_columns] = numeric_features
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].map({"M": 1, "B": 0}).astype("int64")
    return frame, feature_columns


def split_and_scale(
    frame: pd.DataFrame, feature_columns: list[str]
) -> dict[str, pd.DataFrame]:
    """Create stratified partitions and scale using training data only."""
    x = frame[feature_columns]
    y = frame[TARGET_COLUMN]
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    def assemble(features, labels) -> pd.DataFrame:
        result = pd.DataFrame(features, columns=feature_columns, index=labels.index)
        result[TARGET_COLUMN] = labels
        return result.sort_index()

    return {
        "train": assemble(x_train_scaled, y_train),
        "val": assemble(x_val_scaled, y_val),
        "test": assemble(x_test_scaled, y_test),
    }


def save_partitions(partitions: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Persist each processed partition as a CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, partition in partitions.items():
        destination = output_dir / f"{name}.csv"
        partition.to_csv(destination, index=False)


def print_summary(
    frame: pd.DataFrame, partitions: dict[str, pd.DataFrame], output_dir: Path
) -> None:
    """Print inspection, partition, balance, and persistence results."""
    print("DS471 preprocessing pipeline")
    print(f"Input shape: {frame.shape}")
    print(f"Input dtypes: {frame.dtypes.value_counts().to_dict()}")
    print(f"Null cells: {int(frame.isna().sum().sum())}")
    print(f"Target encoding: M -> 1, B -> 0")
    print("Identifier column excluded from model features: id")
    print("Partitions (minor rounding is necessary because n=569):")
    for name, partition in partitions.items():
        balance = partition[TARGET_COLUMN].value_counts(normalize=True).sort_index()
        print(
            f"  {name}: shape={partition.shape}, "
            f"class_balance={balance.round(4).to_dict()}"
        )
        print(f"  generated: {output_dir / f'{name}.csv'}")
    print("Scaling: StandardScaler fitted on train only; val/test were transformed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame, feature_columns = load_and_validate(args.input)
    partitions = split_and_scale(frame, feature_columns)
    save_partitions(partitions, args.output_dir)
    print_summary(frame, partitions, args.output_dir)


if __name__ == "__main__":
    main()
