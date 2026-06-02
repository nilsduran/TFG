from __future__ import annotations

import gc
import time
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    auc,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .experiments import temporal_stratified_splits, balanced_sample


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
    l1_lambda: float = 0.0


def temporal_train_val_split(y_train: np.ndarray, val_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Split train/validation using a chronological stratified split to guarantee
    class balance while ensuring that all validation trials are chronologically
    later than training trials of the same class, avoiding temporal leakage.
    """
    n_samples = len(y_train)
    indices = np.arange(n_samples)

    val_indices = []
    tr_indices = []

    # Split chronologically within each class to ensure stratification
    for c in np.unique(y_train):
        c_indices = indices[y_train == c]
        n_c = len(c_indices)
        n_val_c = int(round(n_c * val_size))
        # Ensure we have at least 1 train sample per class
        if n_c - n_val_c < 1:
            n_val_c = n_c - 1
        # If possible, have at least 1 val sample per class
        if n_val_c < 1 and n_c >= 2:
            n_val_c = 1

        n_train_c = n_c - n_val_c
        tr_indices.extend(c_indices[:n_train_c])
        val_indices.extend(c_indices[n_train_c:])

    tr_indices = np.sort(tr_indices)
    val_indices = np.sort(val_indices)

    # Fallback to randomized stratified split if validation or train is empty
    if len(tr_indices) == 0 or len(val_indices) == 0:
        from sklearn.model_selection import train_test_split

        try:
            tr_indices, val_indices = train_test_split(indices, test_size=val_size, stratify=y_train, random_state=42)
        except ValueError:
            tr_indices, val_indices = train_test_split(indices, test_size=val_size, random_state=42)
        tr_indices, val_indices = np.sort(tr_indices), np.sort(val_indices)

    return tr_indices, val_indices


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
    def _standardize_individual(X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            # Raw EEG: (trials, channels, times)
            # Z-score per trial and per channel over the time axis
            mean = X.mean(axis=-1, keepdims=True)
            std = X.std(axis=-1, keepdims=True)
        elif X.ndim == 4:
            # Graph features: (trials, windows, nodes, features)
            # Z-score per trial, node, and feature over the windows (temporal) axis
            mean = X.mean(axis=1, keepdims=True)
            std = X.std(axis=1, keepdims=True)
        else:
            mean = X.mean(axis=-1, keepdims=True)
            std = X.std(axis=-1, keepdims=True)
        std = np.maximum(std, 1e-6)
        return ((X - mean) / std).astype(np.float32)

    return _standardize_individual(X_train), _standardize_individual(X_test)


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


def _instantiate_model(make_model: Callable, X: np.ndarray) -> nn.Module:
    sig = inspect.signature(make_model)
    if len(sig.parameters) > 0:
        return make_model(X.shape[1], X.shape[2])
    return make_model()


def fit_one_split(
    make_model: Callable[[], nn.Module],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: TorchTrainConfig,
    *,
    save_path: Path | None = None,
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray, np.ndarray]:
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

    model = _instantiate_model(make_model, X_train).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Calculate class weights for training set to handle chronological imbalance
    class_counts = np.bincount(y_tr, minlength=2)
    class_counts = np.maximum(class_counts, 1)
    weights = len(y_tr) / (2.0 * class_counts)
    class_weights = torch.tensor(weights, dtype=torch.float, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_state = None
    best_loss = np.inf
    best_val_bal_acc = 0.0
    wait = 0
    history = []
    epoch_durations: list[float] = []

    # Optionally use tqdm for epoch and batch progress
    try:
        from tqdm.auto import tqdm as _tqdm
    except Exception:
        _tqdm = None

    epoch_iter = range(1, config.epochs + 1)
    pbar = None
    if config.progress and _tqdm is not None:
        pbar = _tqdm(epoch_iter, desc="Epochs", leave=False)
        epoch_iter = pbar

    for epoch in epoch_iter:
        epoch_start = time.perf_counter()
        model.train()
        train_losses: list[float] = []
        batch_iter = train_loader
        for xb, yb in batch_iter:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.amp.autocast(device_type="cuda"):
                    loss = loss_fn(model(xb), yb)
                    if getattr(config, "l1_lambda", 0.0) > 0.0:
                        l1_penalty = 0.0
                        for name, param in model.named_parameters():
                            if "classifier" in name and "weight" in name:
                                l1_penalty = l1_penalty + torch.sum(torch.abs(param))
                        loss = loss + config.l1_lambda * l1_penalty
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = loss_fn(model(xb), yb)
                if getattr(config, "l1_lambda", 0.0) > 0.0:
                    l1_penalty = 0.0
                    for name, param in model.named_parameters():
                        if "classifier" in name and "weight" in name:
                            l1_penalty = l1_penalty + torch.sum(torch.abs(param))
                    loss = loss + config.l1_lambda * l1_penalty
                loss.backward()
                optimizer.step()

            train_losses.append(float(loss.item()))

        # run evaluation (no progress for val by default)
        val_loss, val_acc, y_val_true, p_val = _evaluate(model, val_loader, device)
        pred_val = (p_val >= 0.5).astype(int)
        val_bal_acc = float(balanced_accuracy_score(y_val_true, pred_val))
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_balanced_acc": val_bal_acc,
            }
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_val_bal_acc = val_bal_acc
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
            if pbar is not None:
                pbar.set_postfix(
                    {
                        "loss": f"{history[-1]['train_loss']:.4f}",
                        "val_acc": f"{val_acc:.4f}",
                        "elapsed": _format_seconds(split_elapsed),
                        "eta": _format_seconds(split_eta),
                    }
                )
            else:
                print(
                    f"Epoch {epoch}/{config.epochs} | "
                    f"train_loss={history[-1]['train_loss']:.4f} val_acc={val_acc:.4f} | "
                    f"epoch={_format_seconds(epoch_seconds)} elapsed={_format_seconds(split_elapsed)} "
                    f"eta_split={_format_seconds(split_eta)}"
                )

    if pbar is not None:
        pbar.close()

    if best_state is not None:
        model.load_state_dict(best_state)
        if save_path is not None:
            torch.save(best_state, save_path)

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
        "best_val_acc": float(best_val_bal_acc),
        "epochs_ran": int(len(history)),
    }
    # Release the fold-local model and dataloaders promptly; on Colab CPU this
    # avoids cumulative memory growth when the backbone is rebuilt multiple times.
    del model, optimizer, loss_fn, train_loader, val_loader, test_loader, best_state
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics, pd.DataFrame(history), y_true, p_test


def cross_validate_torch_model(
    X: np.ndarray,
    y: np.ndarray,
    make_model: Callable[[], nn.Module],
    config: TorchTrainConfig,
    *,
    subjects: np.ndarray | None = None,
    subject_id: str | None = None,
    save_weights_dir: Path | None = None,
    weights_prefix: str | None = None,
    load_weights: bool = False,
    plot_roc_pr: bool = False,
    figures_dir: Path | None = None,
    task_tag: str | None = None,
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

    if save_weights_dir is not None:
        save_weights_dir = Path(save_weights_dir)
        save_weights_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    histories = []
    fold_times: list[float] = []
    all_fold_y_true: list[np.ndarray] = []
    all_fold_p_test: list[np.ndarray] = []
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

        fold_weights_path = None
        if save_weights_dir is not None and weights_prefix is not None:
            fold_weights_path = save_weights_dir / f"{weights_prefix}_fold{fold}.pt"

        if load_weights and fold_weights_path is not None and fold_weights_path.exists():
            print(f"Fold {fold}: Loading weights from {fold_weights_path}")
            model = _instantiate_model(make_model, X).to(device)
            model.load_state_dict(torch.load(fold_weights_path, map_location=device))

            _, X_test = standardize_train_test(X[train_idx], X[test_idx])
            y_test = y[test_idx]
            test_loader = build_dataloader(X_test, y_test, batch_size=config.batch_size, shuffle=False, device=device)

            test_loss, test_acc, y_true_arr, p_test = _evaluate(model, test_loader, device)
            pred = (p_test >= 0.5).astype(int)
            try:
                auc = float(roc_auc_score(y_true_arr, p_test))
            except ValueError:
                auc = float("nan")

            metrics = {
                "test_loss": test_loss,
                "test_acc": test_acc,
                "test_balanced_acc": balanced_accuracy_score(y_true_arr, pred),
                "test_auc": auc,
                "best_val_acc": float("nan"),
                "epochs_ran": 0,
                "fold": fold,
            }
            rows.append(metrics)
            all_fold_y_true.append(y_true_arr)
            all_fold_p_test.append(p_test)

            history = pd.DataFrame(
                [
                    {
                        "epoch": 1,
                        "train_loss": float("nan"),
                        "val_loss": float("nan"),
                        "val_acc": float("nan"),
                        "fold": fold,
                    }
                ]
            )
            histories.append(history)

            fold_seconds = time.perf_counter() - fold_start
            fold_times.append(fold_seconds)
            del model, test_loader
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(
                f"Fold {fold} (LOADED): acc={metrics['test_acc']:.3f}, bal_acc={metrics['test_balanced_acc']:.3f}, auc={metrics['test_auc']:.3f} "
                f"| fold_time={_format_seconds(fold_seconds)}"
            )
            continue

        X_train, X_test = standardize_train_test(X[train_idx], X[test_idx])
        y_train, y_test = y[train_idx], y[test_idx]
        metrics, history, fold_y_true, fold_p_test = fit_one_split(
            make_model, X_train, y_train, X_test, y_test, config, save_path=fold_weights_path
        )
        metrics["fold"] = fold
        rows.append(metrics)
        all_fold_y_true.append(fold_y_true)
        all_fold_p_test.append(fold_p_test)
        history["fold"] = fold
        histories.append(history)
        fold_seconds = time.perf_counter() - fold_start
        fold_times.append(fold_seconds)
        avg_fold = float(np.mean(fold_times))
        folds_left = max(0, len(splits) - fold)
        eta_cv = avg_fold * folds_left
        elapsed_cv = time.perf_counter() - cv_start
        print(
            f"Fold {fold}: acc={metrics['test_acc']:.4f}, bal_acc={metrics['test_balanced_acc']:.4f}, auc={metrics['test_auc']:.4f} "
            f"| fold_time={_format_seconds(fold_seconds)} elapsed_cv={_format_seconds(elapsed_cv)} eta_cv={_format_seconds(eta_cv)}"
        )

    if all_fold_y_true:
        df_rows = pd.DataFrame(rows)
        mean_acc = df_rows["test_acc"].mean()
        std_acc = df_rows["test_acc"].std() if len(df_rows) > 1 else 0.0
        fold5_acc = (
            df_rows[df_rows["fold"] == len(df_rows)]["test_acc"].values[0]
            if len(df_rows) >= 5
            else df_rows["test_acc"].values[-1]
        )
        print("\n" + "=" * 60)
        print("SUMMARY METRICS ACROSS FOLDS")
        print("=" * 60)
        print(f"Fold 5 Accuracy: {fold5_acc:.4f}")
        print(f"Average CV Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        print("=" * 60 + "\n")

        y_true_all = np.concatenate(all_fold_y_true)
        p_test_all = np.concatenate(all_fold_p_test)
        y_pred_all = (p_test_all >= 0.5).astype(int)

        from sklearn.metrics import classification_report

        print("\n" + "=" * 60)
        print("OVERALL CROSS-VALIDATION CLASSIFICATION REPORT (AGGREGATED)")
        print("=" * 60)
        print(classification_report(y_true_all, y_pred_all))
        print("=" * 60 + "\n")

    # ── ROC-AUC & PR-AUC plots ──────────────────────────────────────────
    if plot_roc_pr and all_fold_y_true:
        _plot_roc_pr_curves(
            all_fold_y_true,
            all_fold_p_test,
            figures_dir=figures_dir,
            task_tag=task_tag,
        )

    return pd.DataFrame(rows), pd.concat(histories, ignore_index=True)


def _plot_roc_pr_curves(
    all_fold_y_true: list[np.ndarray],
    all_fold_p_test: list[np.ndarray],
    *,
    figures_dir: Path | None = None,
    task_tag: str | None = None,
) -> None:
    """Generate and save ROC-AUC and PR-AUC curve plots across folds."""
    cmap = plt.cm.get_cmap("tab10")
    n_folds = len(all_fold_y_true)
    common_grid = np.linspace(0.0, 1.0, 100)

    # ── ROC Curve ────────────────────────────────────────────────────────
    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
    tpr_interps: list[np.ndarray] = []
    roc_aucs: list[float] = []

    for i in range(n_folds):
        fpr_i, tpr_i, _ = roc_curve(all_fold_y_true[i], all_fold_p_test[i])
        roc_auc_i = auc(fpr_i, tpr_i)
        roc_aucs.append(roc_auc_i)
        ax_roc.plot(
            fpr_i,
            tpr_i,
            color=cmap(i),
            alpha=0.3,
            linewidth=1,
            label=f"Fold {i + 1} (AUC = {roc_auc_i:.3f})",
        )
        tpr_interp = np.interp(common_grid, fpr_i, tpr_i)
        tpr_interp[0] = 0.0
        tpr_interps.append(tpr_interp)

    mean_tpr = np.mean(tpr_interps, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tpr_interps, axis=0)
    mean_auc = float(np.mean(roc_aucs))
    std_auc = float(np.std(roc_aucs))

    ax_roc.plot(
        common_grid,
        mean_tpr,
        color="navy",
        linewidth=2,
        label=f"Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})",
    )
    ax_roc.fill_between(
        common_grid,
        np.clip(mean_tpr - std_tpr, 0, 1),
        np.clip(mean_tpr + std_tpr, 0, 1),
        color="navy",
        alpha=0.15,
    )
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="Chance")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve (Cross-Validation)")
    ax_roc.legend(loc="lower right", fontsize=8)
    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.02])
    ax_roc.grid(True, alpha=0.3)
    fig_roc.tight_layout()

    if figures_dir is not None:
        tag = task_tag or "model"
        fig_roc.savefig(Path(figures_dir) / f"roc_auc_{tag}.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ── PR Curve ─────────────────────────────────────────────────────────
    fig_pr, ax_pr = plt.subplots(figsize=(7, 6))
    prec_interps: list[np.ndarray] = []
    pr_aucs: list[float] = []
    baselines: list[float] = []

    for i in range(n_folds):
        prec_i, rec_i, _ = precision_recall_curve(all_fold_y_true[i], all_fold_p_test[i])
        ap_i = average_precision_score(all_fold_y_true[i], all_fold_p_test[i])
        pr_aucs.append(ap_i)
        baselines.append(float(np.mean(all_fold_y_true[i])))
        ax_pr.plot(
            rec_i,
            prec_i,
            color=cmap(i),
            alpha=0.3,
            linewidth=1,
            label=f"Fold {i + 1} (μP = {ap_i:.3f})",
        )
        # Reverse so recall is monotonically increasing for interp
        rec_rev = rec_i[::-1]
        prec_rev = prec_i[::-1]
        prec_interp = np.interp(common_grid, rec_rev, prec_rev)
        prec_interps.append(prec_interp)

    mean_prec = np.mean(prec_interps, axis=0)
    std_prec = np.std(prec_interps, axis=0)
    mean_ap = float(np.mean(pr_aucs))
    std_ap = float(np.std(pr_aucs))
    baseline = float(np.mean(baselines))

    ax_pr.plot(
        common_grid,
        mean_prec,
        color="darkgreen",
        linewidth=2,
        label=f"Mean PR (μP = {mean_ap:.3f} ± {std_ap:.3f})",
    )
    ax_pr.fill_between(
        common_grid,
        np.clip(mean_prec - std_prec, 0, 1),
        np.clip(mean_prec + std_prec, 0, 1),
        color="darkgreen",
        alpha=0.15,
    )
    ax_pr.axhline(y=baseline, linestyle="--", color="grey", linewidth=1, label=f"Baseline ({baseline:.2f})")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve (Cross-Validation)")
    ax_pr.legend(loc="lower left", fontsize=8)
    ax_pr.set_xlim([-0.02, 1.02])
    ax_pr.set_ylim([-0.02, 1.05])
    ax_pr.grid(True, alpha=0.3)
    fig_pr.tight_layout()

    if figures_dir is not None:
        tag = task_tag or "model"
        fig_pr.savefig(Path(figures_dir) / f"pr_auc_{tag}.png", dpi=300, bbox_inches="tight")
    plt.show()


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


def run_torch_epochs_grid(
    X: np.ndarray,
    y: np.ndarray,
    make_model: Callable[[], nn.Module],
    config: TorchTrainConfig,
    epochs_grid: Sequence[int],
    n_repeats: int,
    *,
    subjects: np.ndarray | None = None,
    subject_id: str | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Run cross-validation on PyTorch models for subsets of epochs to draw learning curves.

    Uses analytical binomial chance levels to save 100% of permutation cost.
    """
    import scipy.stats as stats
    from .experiments import choose_n_splits

    if subject_id is not None:
        if subjects is None:
            raise ValueError("subjects must be provided when subject_id is set.")
        mask = np.asarray(subjects) == subject_id
        X = X[mask]
        y = y[mask]
        print(f"[Epochs Grid] Filtering to subject {subject_id}: {len(y)} epochs")

    rng = np.random.default_rng(random_seed)
    rows = []

    n_splits = config.n_splits

    for n_total in epochs_grid:
        if n_total % 2 != 0:
            continue

        # Test size per fold in chronological CV: approximately n_total / (n_splits + 1)
        test_size_est = max(1, n_total // (n_splits + 1))

        # Free analytical chance levels
        chance_mean = 0.5
        chance_q95 = float(stats.binom.ppf(0.95, test_size_est, 0.5) / test_size_est)

        real_accs = []
        for rep in range(n_repeats):
            idx_sub = balanced_sample(y, n_total, rng)
            if idx_sub is None:
                continue
            X_sub = X[idx_sub]
            y_sub = y[idx_sub]

            silent_config = TorchTrainConfig(
                epochs=config.epochs,
                batch_size=config.batch_size,
                lr=config.lr,
                weight_decay=config.weight_decay,
                patience=config.patience,
                n_splits=config.n_splits,
                val_size=config.val_size,
                random_seed=random_seed + rep,
                device=config.device,
                progress=False,
                l1_lambda=config.l1_lambda,
            )
            cv_metrics, _ = cross_validate_torch_model(X_sub, y_sub, make_model, silent_config)
            real_accs.append(cv_metrics["test_acc"].mean())

        if not real_accs:
            continue

        real_mean = float(np.mean(real_accs))
        real_std = float(np.std(real_accs))

        rows.append(
            {
                "subject": subject_id or "all",
                "n_epochs_total": int(n_total),
                "acc_real_mean": real_mean,
                "acc_real_std": real_std,
                "acc_chance_mean": chance_mean,
                "acc_chance_q95": chance_q95,
                "delta_acc": real_mean - chance_mean,
                "delta_acc_q95": real_mean - chance_q95,
            }
        )
        print(f"Grid point {n_total} epochs: acc={real_mean:.3f} ± {real_std:.3f} | chance Q95={chance_q95:.3f}")

    return pd.DataFrame(rows)


def run_torch_time_map(
    X: np.ndarray,
    y: np.ndarray,
    make_model: Callable[[], nn.Module],
    config: TorchTrainConfig,
    windows: Any,
    *,
    subjects: np.ndarray | None = None,
    subject_id: str | None = None,
) -> dict[str, np.ndarray]:
    """Train PyTorch model on sliding time windows to draw accuracy map over time."""
    import scipy.stats as stats

    if subject_id is not None:
        if subjects is None:
            raise ValueError("subjects must be provided when subject_id is set.")
        mask = np.asarray(subjects) == subject_id
        X = X[mask]
        y = y[mask]
        print(f"[Time Map] Filtering to subject {subject_id}: {len(y)} epochs")

    n_windows = len(windows.starts)
    real_map = np.empty(n_windows, dtype=np.float32)
    surr_mean = np.empty(n_windows, dtype=np.float32)
    surr_q95 = np.empty(n_windows, dtype=np.float32)

    # Calculate test size for analytical Q95
    n_splits = config.n_splits
    n_total = len(y)
    test_size_est = max(1, n_total // (n_splits + 1))
    chance_mean = 0.5
    chance_q95 = float(stats.binom.ppf(0.95, test_size_est, 0.5) / test_size_est)

    for wi, (start, stop) in enumerate(zip(windows.starts, windows.stops)):
        print(f"Training Time Window {wi+1}/{n_windows} (center {windows.centers[wi]:.3f}s) ...")
        X_win = X[:, :, start:stop]

        silent_config = TorchTrainConfig(
            epochs=config.epochs,
            batch_size=config.batch_size,
            lr=config.lr,
            weight_decay=config.weight_decay,
            patience=config.patience,
            n_splits=config.n_splits,
            val_size=config.val_size,
            random_seed=config.random_seed + wi,
            device=config.device,
            progress=False,
            l1_lambda=config.l1_lambda,
        )
        cv_metrics, _ = cross_validate_torch_model(X_win, y, make_model, silent_config)
        acc = cv_metrics["test_acc"].mean()
        real_map[wi] = acc
        surr_mean[wi] = chance_mean
        surr_q95[wi] = chance_q95
        print(f"Window {wi+1} Acc: {acc:.3f} | Chance Q95: {chance_q95:.3f}")

    return {
        "real_map": real_map,
        "surr_mean": surr_mean,
        "surr_q95": surr_q95,
        "centers": windows.centers,
    }
