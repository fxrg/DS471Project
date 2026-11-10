"""Tune DS471 classifiers with stratified cross-validation and evaluate on test data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "confusion_matrices.png"
TARGET_COLUMN = "diagnosis"
PARTITIONS = ("train", "val", "test")
RANDOM_STATE = 42
METRIC_NAMES = ("Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC")


def load_partitions(
    data_dir: Path,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Load processed partitions and validate their shared feature schema."""
    partitions: dict[str, pd.DataFrame] = {}
    feature_columns: list[str] | None = None

    for split_name in PARTITIONS:
        path = data_dir / f"{split_name}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Processed partition was not found: {path}")
        frame = pd.read_csv(path)
        if frame.empty or TARGET_COLUMN not in frame.columns:
            raise ValueError(f"{path} is empty or missing '{TARGET_COLUMN}'.")
        if frame.isna().any().any():
            raise ValueError(f"{path} contains missing values.")

        current_features = [column for column in frame.columns if column != TARGET_COLUMN]
        if feature_columns is None:
            feature_columns = current_features
        elif current_features != feature_columns:
            raise ValueError("Partition feature columns do not match in the same order.")

        if not np.isfinite(frame[current_features].to_numpy(dtype=float)).all():
            raise ValueError(f"{path} contains non-finite feature values.")
        if not set(frame[TARGET_COLUMN].unique()) <= {0, 1}:
            raise ValueError(f"{path} target must be encoded as 0/1.")
        partitions[split_name] = frame

    if feature_columns is None or len(feature_columns) != 30:
        raise ValueError("Expected exactly 30 numerical feature columns.")
    if set(partitions["train"][TARGET_COLUMN].unique()) != {0, 1}:
        raise ValueError("The training target must contain both classes.")
    return partitions, feature_columns


def build_searches(cv: StratifiedKFold) -> dict[str, GridSearchCV]:
    """Create the required model searches and their bounded tuning grids."""
    searches: dict[str, GridSearchCV] = {
        "Logistic Regression": GridSearchCV(
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            {
                "C": [0.01, 0.1, 1.0, 10.0],
                "penalty": ["l2"],
            },
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
        ),
        "Random Forest": GridSearchCV(
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5],
            },
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
        ),
        "MLP": GridSearchCV(
            MLPClassifier(
                max_iter=1000,
                early_stopping=True,
                random_state=RANDOM_STATE,
            ),
            {
                "hidden_layer_sizes": [(32,), (64, 32)],
                "alpha": [0.0001, 0.001],
                "learning_rate_init": [0.0005, 0.001],
            },
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
        ),
    }
    return searches


def evaluate_model(
    model: Any, features: pd.DataFrame, labels: pd.Series
) -> dict[str, float]:
    """Compute classification metrics from a fitted probabilistic classifier."""
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    return {
        "Accuracy": accuracy_score(labels, predictions),
        "Precision": precision_score(labels, predictions, zero_division=0),
        "Recall": recall_score(labels, predictions, zero_division=0),
        "F1-Score": f1_score(labels, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(labels, probabilities),
    }


def save_confusion_matrices(
    models: dict[str, Any],
    features: pd.DataFrame,
    labels: pd.Series,
    output_path: Path,
) -> None:
    """Save test-set confusion matrices for all tuned candidate models."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, len(models), figsize=(15, 4.5))
    for axis, (model_name, model) in zip(np.atleast_1d(axes), models.items()):
        matrix = confusion_matrix(labels, model.predict(features), labels=[0, 1])
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Benign (0)", "Malignant (1)"],
            yticklabels=["Benign (0)", "Malignant (1)"],
            ax=axis,
        )
        axis.set_title(model_name)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
    figure.suptitle("Tuned model confusion matrices (hold-out test set)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def tune_and_evaluate(
    partitions: dict[str, pd.DataFrame],
    feature_columns: list[str],
    figure_path: Path,
) -> pd.DataFrame:
    """Tune on train+validation folds and evaluate each refit model on test."""
    train_validation = pd.concat(
        [partitions["train"], partitions["val"]], ignore_index=True
    )
    test = partitions["test"]
    x_train_validation = train_validation[feature_columns]
    y_train_validation = train_validation[TARGET_COLUMN]
    x_test = test[feature_columns]
    y_test = test[TARGET_COLUMN]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    tuned_models: dict[str, Any] = {}
    results: list[dict[str, Any]] = []
    for model_name, search in build_searches(cv).items():
        search.fit(x_train_validation, y_train_validation)
        tuned_models[model_name] = search.best_estimator_
        results.append(
            {
                "Model": model_name,
                "Best Parameters": search.best_params_,
                **evaluate_model(search.best_estimator_, x_test, y_test),
            }
        )

    save_confusion_matrices(tuned_models, x_test, y_test, figure_path)
    return pd.DataFrame(results, columns=["Model", "Best Parameters", *METRIC_NAMES])


def print_summary(results: pd.DataFrame, figure_path: Path) -> None:
    """Print tuned parameters and hold-out test metrics."""
    display = results.copy()
    for metric in METRIC_NAMES:
        display[metric] = display[metric].map(lambda value: f"{value:.4f}")
    print("\nTuned model results (5-fold stratified CV on train + validation)")
    print(display.to_string(index=False))
    print(f"\nConfusion matrices saved to: {figure_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_FIGURE_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    partitions, feature_columns = load_partitions(args.data_dir)
    results = tune_and_evaluate(partitions, feature_columns, args.figure_path)
    print_summary(results, args.figure_path)


if __name__ == "__main__":
    main()
