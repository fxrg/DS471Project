"""Train and evaluate the DS471 multilayer perceptron baseline."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_CURVE_PATH = PROJECT_ROOT / "reports" / "figures" / "mlp_learning_curves.png"
TARGET_COLUMN = "diagnosis"
RANDOM_STATE = 42


class BreastCancerMLP(nn.Module):
    """Two-hidden-layer MLP for the 30 standardized WDBC features."""

    def __init__(self, input_features: int = 30) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(1)


def set_seed(seed: int) -> None:
    """Make CPU training reproducible for baseline comparisons."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_partitions(
    data_dir: Path,
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], list[str]]:
    """Load processed partitions and convert their 30 features to tensors."""
    partitions: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    feature_columns: list[str] | None = None
    for split_name in ("train", "val", "test"):
        path = data_dir / f"{split_name}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Processed partition was not found: {path}")
        frame = pd.read_csv(path)
        if TARGET_COLUMN not in frame.columns or frame.empty:
            raise ValueError(f"{path} is empty or missing '{TARGET_COLUMN}'.")
        if frame.isna().any().any():
            raise ValueError(f"{path} contains missing values.")
        current_features = [column for column in frame.columns if column != TARGET_COLUMN]
        if feature_columns is None:
            feature_columns = current_features
        if current_features != feature_columns:
            raise ValueError("Partition feature columns do not match.")
        labels = set(frame[TARGET_COLUMN].unique())
        if not labels <= {0, 1}:
            raise ValueError(f"{path} target must be encoded as 0/1, found {labels}.")
        features = torch.tensor(
            frame[feature_columns].to_numpy(dtype=np.float32), dtype=torch.float32
        )
        targets = torch.tensor(
            frame[TARGET_COLUMN].to_numpy(dtype=np.float32), dtype=torch.float32
        )
        partitions[split_name] = (features, targets)

    if feature_columns is None or len(feature_columns) != 30:
        raise ValueError("Expected exactly 30 numerical feature columns.")
    return partitions, feature_columns


def evaluate(
    model: BreastCancerMLP,
    features: torch.Tensor,
    targets: torch.Tensor,
    loss_function: nn.Module,
) -> tuple[float, dict[str, float]]:
    """Evaluate without gradient tracking and return loss plus classification metrics."""
    model.eval()
    with torch.no_grad():
        logits = model(features)
        loss = loss_function(logits, targets).item()
        probabilities = torch.sigmoid(logits).cpu().numpy()
    predictions = (probabilities >= 0.5).astype(int)
    labels = targets.cpu().numpy().astype(int)
    metrics = {
        "Accuracy": accuracy_score(labels, predictions),
        "Precision": precision_score(labels, predictions, zero_division=0),
        "Recall": recall_score(labels, predictions, zero_division=0),
        "F1-Score": f1_score(labels, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(labels, probabilities),
    }
    return loss, metrics


def train_model(
    partitions: dict[str, tuple[torch.Tensor, torch.Tensor]],
    epochs: int,
    learning_rate: float,
    patience: int,
) -> tuple[BreastCancerMLP, pd.DataFrame, int]:
    """Train with Adam/BCE and restore the best validation-loss checkpoint."""
    train_features, train_targets = partitions["train"]
    val_features, val_targets = partitions["val"]
    model = BreastCancerMLP(input_features=train_features.shape[1])
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_loss = loss_function(model(train_features), train_targets)
        train_loss.backward()
        optimizer.step()

        validation_loss, validation_metrics = evaluate(
            model, val_features, val_targets, loss_function
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss.item(),
                "val_loss": validation_loss,
                "val_accuracy": validation_metrics["Accuracy"],
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                name: parameter.detach().clone()
                for name, parameter in model.state_dict().items()
            }
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a validation checkpoint.")
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), best_epoch


def save_learning_curve(history: pd.DataFrame, output_path: Path) -> None:
    """Save training and validation loss curves for the project report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(history["epoch"], history["train_loss"], label="Training loss")
    axis.plot(history["epoch"], history["val_loss"], label="Validation loss")
    axis.set_title("MLP Training and Validation Loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Binary cross-entropy loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--curve-path", type=Path, default=DEFAULT_CURVE_PATH)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.learning_rate <= 0 or args.patience < 1:
        raise ValueError("epochs and patience must be positive; learning rate must be > 0.")
    set_seed(RANDOM_STATE)
    partitions, feature_columns = load_partitions(args.data_dir)
    model, history, best_epoch = train_model(
        partitions, args.epochs, args.learning_rate, args.patience
    )
    save_learning_curve(history, args.curve_path)
    loss_function = nn.BCEWithLogitsLoss()
    _, test_metrics = evaluate(
        model, partitions["test"][0], partitions["test"][1], loss_function
    )
    print("Neural Network (MLP) baseline")
    print("Architecture: 30 -> 64 (ReLU) -> Dropout(0.2) -> 32 (ReLU) -> 1")
    print(
        f"Optimizer: Adam (learning_rate={args.learning_rate}); "
        f"loss: BCEWithLogitsLoss; best epoch: {best_epoch}"
    )
    print(f"Epochs completed: {len(history)}")
    print(f"Learning curve: {args.curve_path}")
    print("\nFinal test metrics")
    for metric_name, value in test_metrics.items():
        print(f"{metric_name:>10}: {value:.4f}")
    if len(feature_columns) != 30:
        raise RuntimeError("Unexpected feature count after training.")


if __name__ == "__main__":
    main()
