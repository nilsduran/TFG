from __future__ import annotations

import copy
import random
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from ..dataset import EpochDataset
from ..explanations import ExplanationResult, explain_torch_gradient, stratified_background
from ..metrics import classification_metrics
from ..splits import SplitPlan, chronological_holdout_indices

TorchModelFactory = Callable[[tuple[int, ...], int], nn.Module]
InputTransform = Callable[[np.ndarray, float], np.ndarray]


@dataclass(frozen=True)
class TorchRunnerConfig:
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 15
    validation_fraction: float = 0.2
    normalization: str = "channel"
    selection_metric: str = "f1_macro"
    random_seed: int = 42
    device: str | None = None

    def __post_init__(self) -> None:
        if self.selection_metric not in {"balanced_accuracy", "f1_macro"}:
            raise ValueError(
                "selection_metric must be 'balanced_accuracy' or 'f1_macro'."
            )


@dataclass
class TorchRunResult:
    metrics: pd.DataFrame
    y_true: np.ndarray
    probabilities: np.ndarray
    subjects: np.ndarray
    trial_indices: np.ndarray
    folds: np.ndarray
    explanation: ExplanationResult | None = None
    skipped_subjects: tuple[str, ...] = ()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _identity_transform(X: np.ndarray, sfreq: float) -> np.ndarray:
    del sfreq
    return X


def _device(config: TorchRunnerConfig) -> torch.device:
    if config.device is not None:
        return torch.device(config.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _fit_normalization(
    X_train: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "none":
        shape = (1,) * X_train.ndim
        return np.zeros(shape, dtype=np.float32), np.ones(shape, dtype=np.float32)
    if mode != "channel":
        raise ValueError("normalization must be 'channel' or 'none'.")
    if X_train.ndim == 3:
        axes = (0, 2)
    elif X_train.ndim == 4:
        axes = (0, 1)
    else:
        raise ValueError("channel normalization supports 3D epochs or 4D graph features.")
    mean = X_train.mean(axis=axes, keepdims=True)
    std = X_train.std(axis=axes, keepdims=True)
    return mean.astype(np.float32), np.maximum(std, 1e-8).astype(np.float32)


def _loader(
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.as_tensor(X, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _predict_probabilities(
    model: nn.Module,
    X: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    probabilities: list[np.ndarray] = []
    loader = _loader(
        X,
        np.zeros(len(X), dtype=np.int64),
        batch_size=batch_size,
        shuffle=False,
    )
    with torch.no_grad():
        for inputs, _ in loader:
            logits = model(inputs.to(device))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probabilities)


def _train_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    make_model: TorchModelFactory,
    n_classes: int,
    config: TorchRunnerConfig,
    seed: int,
    progress_description: str,
) -> tuple[nn.Module, int]:
    _seed_everything(seed)
    device = _device(config)
    model = make_model(tuple(X_train.shape[1:]), n_classes).to(device)
    counts = np.bincount(y_train, minlength=n_classes).astype(float)
    weights = len(y_train) / (n_classes * np.maximum(counts, 1.0))
    loss_function = nn.CrossEntropyLoss(
        weight=torch.as_tensor(weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_loader = _loader(
        X_train,
        y_train,
        batch_size=config.batch_size,
        shuffle=True,
    )

    best_score = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 1
    epochs_without_improvement = 0
    epoch_progress = tqdm(
        range(1, config.epochs + 1),
        desc=progress_description,
        unit="epoch",
        leave=False,
        dynamic_ncols=True,
        disable=None,
    )
    for epoch in epoch_progress:
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(inputs.to(device)), targets.to(device))
            loss.backward()
            optimizer.step()

        validation_probabilities = _predict_probabilities(
            model,
            X_validation,
            batch_size=config.batch_size,
            device=device,
        )
        validation_metrics = classification_metrics(
            y_validation,
            validation_probabilities,
            n_classes=n_classes,
        )
        try:
            score = float(getattr(validation_metrics, config.selection_metric))
        except AttributeError as exc:
            raise ValueError(
                f"Unknown selection metric {config.selection_metric!r}."
            ) from exc
        epoch_progress.set_postfix(
            validation_metric=f"{score:.3f}",
            best=f"{max(best_score, score):.3f}",
            refresh=False,
        )
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break
    epoch_progress.close()

    model.load_state_dict(best_state)
    return model, best_epoch


def _train_fixed_epochs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    make_model: TorchModelFactory,
    n_classes: int,
    config: TorchRunnerConfig,
    seed: int,
    epochs: int,
    progress_description: str,
) -> nn.Module:
    _seed_everything(seed)
    device = _device(config)
    model = make_model(tuple(X_train.shape[1:]), n_classes).to(device)
    counts = np.bincount(y_train, minlength=n_classes).astype(float)
    weights = len(y_train) / (n_classes * np.maximum(counts, 1.0))
    loss_function = nn.CrossEntropyLoss(
        weight=torch.as_tensor(weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_loader = _loader(
        X_train,
        y_train,
        batch_size=config.batch_size,
        shuffle=True,
    )
    epoch_progress = tqdm(
        range(1, epochs + 1),
        desc=progress_description,
        unit="epoch",
        leave=False,
        dynamic_ncols=True,
        disable=None,
    )
    for _ in epoch_progress:
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(inputs.to(device)), targets.to(device))
            loss.backward()
            optimizer.step()
    epoch_progress.close()
    return model


def run_subjectwise_torch(
    dataset: EpochDataset,
    make_model: TorchModelFactory,
    *,
    input_transform: InputTransform = _identity_transform,
    split_plan: SplitPlan = SplitPlan(),
    config: TorchRunnerConfig = TorchRunnerConfig(),
    explain: bool = False,
    background_per_class: int = 10,
    explanation_samples_per_class: int = 10,
    progress_label: str | None = None,
) -> TorchRunResult:
    n_classes = len(dataset.class_ids)
    rows: list[dict[str, float | int | str]] = []
    y_blocks: list[np.ndarray] = []
    probability_blocks: list[np.ndarray] = []
    subject_blocks: list[np.ndarray] = []
    trial_blocks: list[np.ndarray] = []
    fold_blocks: list[np.ndarray] = []
    explanation_values: list[np.ndarray] = []
    explanation_samples: list[np.ndarray] = []

    subjects = dataset.subject_ids
    label = progress_label or dataset.task_id
    progress = tqdm(
        total=len(subjects),
        desc=label,
        unit="subject",
        dynamic_ncols=True,
        disable=None,
    )

    skipped_subjects: list[str] = []
    for subject_index, subject in enumerate(subjects):
        subject_mask = dataset.subjects == subject
        X_subject = input_transform(dataset.X[subject_mask], dataset.sfreq)
        y_subject = dataset.y[subject_mask]
        trial_subject = dataset.trial_indices[subject_mask]
        development, holdout = chronological_holdout_indices(
            len(y_subject),
            split_plan.holdout_fraction,
        )
        expected_classes = np.arange(n_classes)
        if (
            not len(holdout)
            or not np.array_equal(np.unique(y_subject[development]), expected_classes)
            or not np.array_equal(np.unique(y_subject[holdout]), expected_classes)
        ):
            skipped_subjects.append(
                f"{subject}: final chronological holdout does not contain every class"
            )
            progress.update()
            continue
        train_local, validation_local = chronological_holdout_indices(
            len(development),
            config.validation_fraction,
        )
        train = development[train_local]
        validation = development[validation_local]
        if (
            not np.array_equal(np.unique(y_subject[train]), expected_classes)
            or not np.array_equal(np.unique(y_subject[validation]), expected_classes)
        ):
            skipped_subjects.append(
                f"{subject}: development validation split does not contain every class"
            )
            progress.update()
            continue

        progress.set_postfix_str(f"subject={subject}, selecting epoch", refresh=True)
        selection_mean, selection_std = _fit_normalization(
            X_subject[train],
            config.normalization,
        )
        _, best_epoch = _train_fold(
            (X_subject[train] - selection_mean) / selection_std,
            y_subject[train],
            (X_subject[validation] - selection_mean) / selection_std,
            y_subject[validation],
            make_model=make_model,
            n_classes=n_classes,
            config=config,
            seed=config.random_seed + subject_index,
            progress_description=f"{subject} epoch selection",
        )
        final_mean, final_std = _fit_normalization(
            X_subject[development],
            config.normalization,
        )
        progress.set_postfix_str(
            f"subject={subject}, final fit ({best_epoch} epochs)",
            refresh=True,
        )
        model = _train_fixed_epochs(
            (X_subject[development] - final_mean) / final_std,
            y_subject[development],
            make_model=make_model,
            n_classes=n_classes,
            config=config,
            seed=config.random_seed + subject_index,
            epochs=best_epoch,
            progress_description=f"{subject} final fit",
        )
        X_holdout = (X_subject[holdout] - final_mean) / final_std
        probabilities = _predict_probabilities(
            model,
            X_holdout,
            batch_size=config.batch_size,
            device=_device(config),
        )
        if explain:
            background, _ = stratified_background(
                (X_subject[development] - final_mean) / final_std,
                y_subject[development],
                per_class=background_per_class,
                random_seed=config.random_seed + subject_index,
            )
            samples, _ = stratified_background(
                X_holdout,
                y_subject[holdout],
                per_class=explanation_samples_per_class,
                random_seed=config.random_seed + subject_index + 1,
            )
            holdout_explanation = explain_torch_gradient(
                model,
                background=background,
                samples=samples,
                class_ids=dataset.class_ids,
                batch_size=config.batch_size,
            )
            explanation_values.append(holdout_explanation.values)
            explanation_samples.append(holdout_explanation.samples)
        holdout_metrics = classification_metrics(
            y_subject[holdout],
            probabilities,
            n_classes=n_classes,
        )
        rows.append(
            {
                "subject": subject,
                "evaluation": "final_chronological_holdout",
                "selected_epochs": best_epoch,
                **holdout_metrics.to_dict(),
            }
        )
        y_blocks.append(y_subject[holdout])
        probability_blocks.append(probabilities)
        subject_blocks.append(np.full(len(holdout), subject, dtype=object))
        trial_blocks.append(trial_subject[holdout])
        fold_blocks.append(np.ones(len(holdout), dtype=np.int16))
        progress.set_postfix_str(
            f"subject={subject}, balanced_accuracy={holdout_metrics.balanced_accuracy:.3f}",
            refresh=False,
        )
        progress.update()
    progress.close()

    if not y_blocks:
        raise RuntimeError("No valid subject-level folds were produced.")
    explanation = None
    if explanation_values:
        explanation = ExplanationResult(
            values=np.concatenate(explanation_values),
            base_values=np.array([]),
            samples=np.concatenate(explanation_samples),
            class_ids=dataset.class_ids,
            method="shap_gradient",
        )
    return TorchRunResult(
        metrics=pd.DataFrame(rows),
        y_true=np.concatenate(y_blocks),
        probabilities=np.concatenate(probability_blocks),
        subjects=np.concatenate(subject_blocks),
        trial_indices=np.concatenate(trial_blocks),
        folds=np.concatenate(fold_blocks),
        explanation=explanation,
        skipped_subjects=tuple(skipped_subjects),
    )
