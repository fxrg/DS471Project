"""Train and evaluate traditional DS471 classification baselines."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
TARGET_COLUMN = "diagnosis"
RANDOM_STATE = 42
PARTITIONS = ("train", "val", "test")
METRIC_NAMES = ("Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC")


def load_partitions(data_dir: Path) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Load and validate the already scaled train, validation, and test files."""
    partitions: dict[str, pd.DataFrame] = {}
    for name in PARTITIONS:
        path = data_dir / f"{name}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Processed partition was not found: {path}")
        frame = pd.read_csv(path)
        if TARGET_COLUMN not in frame.columns:
            raise ValueError(f"{path} does not contain '{TARGET_COLUMN}'.")
        if frame.empty or frame.isna().any().any():
            raise ValueError(f"{path} is empty or contains missing values.")
        partitions[name] = frame

    feature_columns = [
        column for column in partitions["train"].columns if column != TARGET_COLUMN
    ]
    if len(feature_columns) != 30:
        raise ValueError(f"Expected 30 numerical features, found {len(feature_columns)}.")
    if any(
        list(partitions[name].columns) != feature_columns + [TARGET_COLUMN]
        for name in PARTITIONS
    ):
        raise ValueError("Partition columns do not match in the same order.")
    if any(
        not np.isfinite(partitions[name][feature_columns].to_numpy()).all()
        for name in PARTITIONS
    ):
        raise ValueError("Non-finite feature values found in processed data.")
    if set(partitions["train"][TARGET_COLUMN].unique()) != {0, 1}:
        raise ValueError("The training target must contain both classes encoded as 0/1.")
    return partitions, feature_columns


def build_models() -> dict[str, Any]:
    """Create the required baseline estimators with fixed reproducible settings."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE
        ),
    }


def evaluate_model(model: Any, frame: pd.DataFrame, feature_columns: list[str]) -> dict[str, float]:
    """Evaluate a fitted classifier using labels and positive-class probabilities."""
    x = frame[feature_columns]
    y = frame[TARGET_COLUMN]
    predictions = model.predict(x)
    probabilities = model.predict_proba(x)[:, 1]
    return {
        "Accuracy": accuracy_score(y, predictions),
        "Precision": precision_score(y, predictions, zero_division=0),
        "Recall": recall_score(y, predictions, zero_division=0),
        "F1-Score": f1_score(y, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y, probabilities),
    }


def train_and_evaluate(
    partitions: dict[str, pd.DataFrame], feature_columns: list[str]
) -> pd.DataFrame:
    """Fit on train only, then evaluate independently on validation and test."""
    train = partitions["train"]
    results: list[dict[str, Any]] = []
    for model_name, model in build_models().items():
        model.fit(train[feature_columns], train[TARGET_COLUMN])
        for split_name in ("val", "test"):
            metrics = evaluate_model(model, partitions[split_name], feature_columns)
            results.append({"Model": model_name, "Split": split_name, **metrics})
    return pd.DataFrame(results, columns=["Model", "Split", *METRIC_NAMES])


def print_summary(results: pd.DataFrame) -> None:
    """Print a compact comparison table for validation monitoring and test checks."""
    display = results.copy()
    for metric in METRIC_NAMES:
        display[metric] = display[metric].map(lambda value: f"{value:.4f}")
    print("\nTraditional ML baseline results")
    print(display.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing train.csv, val.csv, and test.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    partitions, feature_columns = load_partitions(args.data_dir)
    results = train_and_evaluate(partitions, feature_columns)
    print_summary(results)


if __name__ == "__main__":
    main()
