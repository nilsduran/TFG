from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class TorchTrainConfig:
    epochs: int = 80
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 12
    n_splits: int = 5
    val_size: float = 0.2
    random_seed: int = 42
    device: str | None = None


class NumpyClassificationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_item = torch.from_numpy(self.X[index])
        y_item = torch.tensor(self.y[index], dtype=torch.long)
        return x_item, y_item


def build_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    dataset = NumpyClassificationDataset(X, y)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )


def standardize_train_test(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)
    return ((X_train - mean) / std).astype(np.float32), ((X_test - mean) / std).astype(np.float32)


def _evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    losses: list[float] = []
    y_true: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            losses.append(float(loss_fn(logits, yb).item()))
            y_true.append(yb.cpu().numpy())
            probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    y_arr = np.concatenate(y_true)
    p_arr = np.concatenate(probs)
    pred = (p_arr >= 0.5).astype(int)
    return float(np.mean(losses)), float(accuracy_score(y_arr, pred)), y_arr, p_arr


def fit_one_split(
    make_model: Callable[[], nn.Module],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: TorchTrainConfig,
) -> tuple[dict[str, float], pd.DataFrame]:
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("TorchTrainConfig.device='cuda' was requested, but CUDA is not available.")
    tr_idx, val_idx = train_test_split(
        np.arange(len(y_train)),
        test_size=config.val_size,
        stratify=y_train,
        random_state=config.random_seed,
    )
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]

    # Build lazy datasets so each split is batched on demand instead of being
    # copied into a second full tensor upfront.
    train_loader = build_dataloader(X_tr, y_tr, batch_size=config.batch_size, shuffle=True, device=device)
    val_loader = build_dataloader(X_val, y_val, batch_size=config.batch_size, shuffle=False, device=device)
    test_loader = build_dataloader(X_test, y_test, batch_size=config.batch_size, shuffle=False, device=device)

    model = make_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    best_state = None
    best_val = -np.inf
    wait = 0
    history = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_loss, val_acc, _, _ = _evaluate(model, val_loader, device)
        history.append(
            {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": val_loss, "val_acc": val_acc}
        )
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_acc, y_true, p_test = _evaluate(model, test_loader, device)
    pred = (p_test >= 0.5).astype(int)
    try:
        auc = float(roc_auc_score(y_true, p_test))
    except ValueError:
        auc = float("nan")
    metrics = {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_balanced_acc": float(balanced_accuracy_score(y_true, pred)),
        "test_auc": auc,
        "best_val_acc": float(best_val),
        "epochs_ran": int(len(history)),
    }
    # Release the fold-local model and dataloaders promptly; on Colab CPU this
    # avoids cumulative memory growth when the backbone is rebuilt multiple times.
    del model, optimizer, loss_fn, train_loader, val_loader, test_loader, best_state
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics, pd.DataFrame(history)


def cross_validate_torch_model(
    X: np.ndarray,
    y: np.ndarray,
    make_model: Callable[[], nn.Module],
    config: TorchTrainConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("TorchTrainConfig.device='cuda' was requested, but CUDA is not available.")
    print(f"Using torch device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")

    splitter = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.random_seed)
    rows = []
    histories = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        X_train, X_test = standardize_train_test(X[train_idx], X[test_idx])
        y_train, y_test = y[train_idx], y[test_idx]
        metrics, history = fit_one_split(make_model, X_train, y_train, X_test, y_test, config)
        metrics["fold"] = fold
        rows.append(metrics)
        history["fold"] = fold
        histories.append(history)
        print(
            f"Fold {fold}: acc={metrics['test_acc']:.3f}, bal_acc={metrics['test_balanced_acc']:.3f}, auc={metrics['test_auc']:.3f}"
        )
    return pd.DataFrame(rows), pd.concat(histories, ignore_index=True)


def plot_torch_history(history: pd.DataFrame, *, output_file: Path | None = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for fold, d in history.groupby("fold"):
        axes[0].plot(d["epoch"], d["train_loss"], alpha=0.75, label=f"fold {fold}")
        axes[1].plot(d["epoch"], d["val_acc"], alpha=0.75, label=f"fold {fold}")
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Validation accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(title="Fold", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    if output_file is not None:
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.show()


def plot_fold_metrics(metrics: pd.DataFrame, *, output_file: Path | None = None) -> None:
    cols = ["test_acc", "test_balanced_acc", "test_auc"]
    d = metrics.melt(
        id_vars="fold", value_vars=[c for c in cols if c in metrics.columns], var_name="metric", value_name="value"
    )
    plt.figure(figsize=(8, 4.5))
    import seaborn as sns

    sns.barplot(data=d, x="metric", y="value", errorbar="sd")
    sns.stripplot(data=d, x="metric", y="value", color="black", alpha=0.65)
    plt.ylim(0.0, 1.0)
    plt.title("Cross-validation metrics")
    plt.tight_layout()
    if output_file is not None:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.show()
