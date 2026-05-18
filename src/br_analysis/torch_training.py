from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .experiments import temporal_stratified_splits


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
    progress: bool = False


def temporal_train_val_split(y_train: np.ndarray, val_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Split train/validation chronologically, with validation after training."""
    n_samples = len(y_train)
    min_val = max(2, int(round(n_samples * val_size)))
    for val_start in range(n_samples - min_val, 1, -1):
        tr_idx = np.arange(val_start)
        val_idx = np.arange(val_start, n_samples)
        if len(np.unique(y_train[tr_idx])) >= 2 and len(np.unique(y_train[val_idx])) >= 2:
            return tr_idx, val_idx
    raise ValueError("No valid temporal train/validation split with both classes in each subset.")


def _format_seconds(seconds: float) -> str:
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


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
        # Optionally show a progress bar for evaluation
        try:
            from tqdm.auto import tqdm as _tqdm  # lazy import
        except Exception:
            _tqdm = None
        iter_loader = loader
        if getattr(loader, "__len__", None) is not None and _tqdm is not None and getattr(loader, "progress", False):
            iter_loader = _tqdm(loader, desc="Eval", leave=False)
        for xb, yb in iter_loader:
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
    split_start = time.perf_counter()
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("TorchTrainConfig.device='cuda' was requested, but CUDA is not available.")
    tr_idx, val_idx = temporal_train_val_split(y_train, config.val_size)
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
    epoch_durations: list[float] = []

    # Optionally use tqdm for epoch and batch progress
    try:
        from tqdm.auto import tqdm as _tqdm
    except Exception:
        _tqdm = None

    epoch_iter = range(1, config.epochs + 1)
    if config.progress and _tqdm is not None:
        epoch_iter = _tqdm(epoch_iter, desc="Epochs")

    for epoch in epoch_iter:
        epoch_start = time.perf_counter()
        model.train()
        train_losses: list[float] = []
        batch_iter = train_loader
        if config.progress and _tqdm is not None:
            batch_iter = _tqdm(train_loader, desc=f"Train ep {epoch}", leave=False)
        for xb, yb in batch_iter:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        # run evaluation (no progress for val by default)
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

        epoch_seconds = time.perf_counter() - epoch_start
        epoch_durations.append(epoch_seconds)
        if config.progress:
            avg_epoch = float(np.mean(epoch_durations))
            epochs_left = max(0, config.epochs - epoch)
            split_elapsed = time.perf_counter() - split_start
            split_eta = avg_epoch * epochs_left
            print(
                f"Epoch {epoch}/{config.epochs} | "
                f"train_loss={history[-1]['train_loss']:.4f} val_acc={val_acc:.4f} | "
                f"epoch={_format_seconds(epoch_seconds)} elapsed={_format_seconds(split_elapsed)} "
                f"eta_split={_format_seconds(split_eta)}"
            )

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
    *,
    subjects: np.ndarray | None = None,
    subject_id: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if subject_id is not None:
        if subjects is None:
            raise ValueError("subjects must be provided when subject_id is set.")
        mask = np.asarray(subjects) == subject_id
        if not mask.any():
            raise ValueError(f"Subject not found: {subject_id}")
        X = X[mask]
        y = y[mask]
        print(f"Filtering to subject {subject_id}: {len(y)} epochs")

    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("TorchTrainConfig.device='cuda' was requested, but CUDA is not available.")
    print(f"Using torch device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")

    rows = []
    histories = []
    fold_times: list[float] = []
    cv_start = time.perf_counter()
    # Optionally show fold progress
    try:
        from tqdm.auto import tqdm as _tqdm
    except Exception:
        _tqdm = None

    splits = temporal_stratified_splits(y, config.n_splits)
    fold_iter = enumerate(splits, start=1)
    if config.progress and _tqdm is not None:
        fold_iter = _tqdm(list(fold_iter), desc="Folds")

    for fold, (train_idx, test_idx) in fold_iter:
        fold_start = time.perf_counter()
        X_train, X_test = standardize_train_test(X[train_idx], X[test_idx])
        y_train, y_test = y[train_idx], y[test_idx]
        metrics, history = fit_one_split(make_model, X_train, y_train, X_test, y_test, config)
        metrics["fold"] = fold
        rows.append(metrics)
        history["fold"] = fold
        histories.append(history)
        fold_seconds = time.perf_counter() - fold_start
        fold_times.append(fold_seconds)
        avg_fold = float(np.mean(fold_times))
        folds_left = max(0, len(splits) - fold)
        eta_cv = avg_fold * folds_left
        elapsed_cv = time.perf_counter() - cv_start
        print(
            f"Fold {fold}: acc={metrics['test_acc']:.3f}, bal_acc={metrics['test_balanced_acc']:.3f}, auc={metrics['test_auc']:.3f} "
            f"| fold_time={_format_seconds(fold_seconds)} elapsed_cv={_format_seconds(elapsed_cv)} eta_cv={_format_seconds(eta_cv)}"
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
