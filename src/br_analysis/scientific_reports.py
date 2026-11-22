from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from .dataset import EpochDataset
from .registry import ModelSpec
from .representations import band_summary_features
from .tasks import TaskSpec


@dataclass(frozen=True)
class SavedRun:
    directory: Path
    metadata: dict[str, Any]
    metrics: pd.DataFrame
    y_true: np.ndarray
    probabilities: np.ndarray
    trial_indices: np.ndarray
    folds: np.ndarray

    @property
    def predictions(self) -> np.ndarray:
        return self.probabilities.argmax(axis=1)


def load_saved_run(
    artifacts_root: Path,
    task_id: str,
    model_id: str,
) -> SavedRun | None:
    directory = artifacts_root / task_id / model_id
    metadata_path = directory / "metadata.json"
    metrics_path = directory / "metrics.csv"
    predictions_path = directory / "predictions.npz"
    if not all(path.is_file() for path in (metadata_path, metrics_path, predictions_path)):
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metrics = pd.read_csv(metrics_path)
    with np.load(predictions_path, allow_pickle=False) as arrays:
        return SavedRun(
            directory=directory,
            metadata=metadata,
            metrics=metrics,
            y_true=arrays["y_true"],
            probabilities=arrays["probabilities"],
            trial_indices=arrays["trial_indices"],
            folds=arrays["folds"],
        )


def configuration_table(task: TaskSpec, model: ModelSpec) -> pd.DataFrame:
    classes = " | ".join(
        f"{condition.id}: {list(condition.event_codes)}"
        for condition in task.classes
    )
    channels = "all EEG" if model.channels is None else ", ".join(model.channels)
    return pd.DataFrame(
        {
            "setting": (
                "task",
                "classes",
                "model",
                "family",
                "preprocessing",
                "representation",
                "channels",
                "analysis_window_s",
                "outer_cv",
                "chronological_holdout",
            ),
            "value": (
                task.id,
                classes,
                model.id,
                model.family,
                model.preprocessing,
                model.representation,
                channels,
                str(model.analysis_window),
                "StratifiedGroupKFold, up to 5 folds, 10 temporal blocks",
                "newest 20%",
            ),
        }
    )


def cohort_table(dataset: EpochDataset) -> pd.DataFrame:
    rows = []
    for class_index, (class_id, label) in enumerate(
        zip(dataset.class_ids, dataset.class_labels, strict=True)
    ):
        mask = dataset.y == class_index
        rows.append(
            {
                "class_id": class_id,
                "label": label,
                "trials": int(mask.sum()),
                "subjects": int(len(np.unique(dataset.subjects[mask]))),
                "fraction": float(mask.mean()),
            }
        )
    return pd.DataFrame(rows)


def subject_trial_table(dataset: EpochDataset) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "subject": dataset.subjects.astype(str),
            "class_id": np.asarray(dataset.class_ids)[dataset.y],
        }
    )
    return (
        frame.value_counts(["subject", "class_id"])
        .rename("trials")
        .reset_index()
        .pivot(index="subject", columns="class_id", values="trials")
        .fillna(0)
        .astype(int)
        .reset_index()
    )


def aggregate_metrics(run: SavedRun) -> pd.DataFrame:
    metric_columns = [
        column
        for column in ("accuracy", "balanced_accuracy", "f1_macro", "roc_auc")
        if column in run.metrics
    ]
    rows = []
    for metric in metric_columns:
        rows.append(
            {
                "metric": metric,
                "mean": float(run.metrics[metric].mean()),
                "std": float(run.metrics[metric].std()),
                "min": float(run.metrics[metric].min()),
                "max": float(run.metrics[metric].max()),
            }
        )
    return pd.DataFrame(rows)


def per_subject_metrics(run: SavedRun) -> pd.DataFrame:
    metric_columns = [
        column
        for column in ("accuracy", "balanced_accuracy", "f1_macro", "roc_auc")
        if column in run.metrics
    ]
    return run.metrics.groupby("subject", as_index=False)[metric_columns].mean()


def confusion_table(
    run: SavedRun,
    class_ids: tuple[str, ...],
    *,
    normalize: bool = True,
) -> pd.DataFrame:
    matrix = confusion_matrix(
        run.y_true,
        run.predictions,
        labels=np.arange(len(class_ids)),
        normalize="true" if normalize else None,
    )
    return pd.DataFrame(
        matrix,
        index=[f"true_{class_id}" for class_id in class_ids],
        columns=[f"pred_{class_id}" for class_id in class_ids],
    )


def prediction_diagnostics(run: SavedRun) -> pd.DataFrame:
    confidence = run.probabilities.max(axis=1)
    correct = run.predictions == run.y_true
    clipped = np.clip(run.probabilities, 1e-12, 1.0)
    entropy = -(clipped * np.log(clipped)).sum(axis=1)
    return pd.DataFrame(
        {
            "diagnostic": (
                "samples",
                "mean_confidence",
                "mean_confidence_correct",
                "mean_confidence_incorrect",
                "mean_entropy",
            ),
            "value": (
                len(run.y_true),
                float(confidence.mean()),
                float(confidence[correct].mean()) if correct.any() else np.nan,
                float(confidence[~correct].mean()) if (~correct).any() else np.nan,
                float(entropy.mean()),
            ),
        }
    )


def plot_subject_performance(
    run: SavedRun,
    *,
    metric: str = "balanced_accuracy",
    chance_level: float,
) -> plt.Figure:
    values = per_subject_metrics(run).sort_values(metric)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(values["subject"], values[metric], color="#2b7a78")
    ax.axhline(chance_level, color="#9c2f2f", linestyle="--", label="Chance")
    ax.set(ylabel=metric.replace("_", " ").title(), xlabel="Subject", ylim=(0, 1))
    ax.tick_params(axis="x", rotation=75)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_confusion(
    run: SavedRun,
    class_ids: tuple[str, ...],
) -> plt.Figure:
    frame = confusion_table(run, class_ids)
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(frame.to_numpy(), vmin=0, vmax=1, cmap="Blues")
    for row in range(len(frame)):
        for column in range(len(frame.columns)):
            ax.text(column, row, f"{frame.iloc[row, column]:.2f}", ha="center", va="center")
    ax.set(
        xticks=np.arange(len(class_ids)),
        yticks=np.arange(len(class_ids)),
        xticklabels=class_ids,
        yticklabels=class_ids,
        xlabel="Predicted class",
        ylabel="True class",
        title="Row-normalized confusion matrix",
    )
    fig.colorbar(image, ax=ax, label="Recall")
    fig.tight_layout()
    return fig


def plot_prediction_confidence(run: SavedRun) -> plt.Figure:
    confidence = run.probabilities.max(axis=1)
    correct = run.predictions == run.y_true
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(confidence[correct], bins=15, alpha=0.7, label="Correct")
    ax.hist(confidence[~correct], bins=15, alpha=0.7, label="Incorrect")
    ax.set(xlabel="Maximum predicted probability", ylabel="Samples", xlim=(0, 1))
    ax.legend()
    fig.tight_layout()
    return fig


def plot_roc_pr_comparison(
    runs: Mapping[str, SavedRun | None],
    *,
    title: str,
) -> plt.Figure:
    available = {
        label: run
        for label, run in runs.items()
        if run is not None
    }
    fig, (roc_axis, pr_axis) = plt.subplots(1, 2, figsize=(12, 5))
    for label, run in available.items():
        targets, scores, prevalence, averaging = _curve_inputs(run)
        false_positive_rate, true_positive_rate, _ = roc_curve(targets, scores)
        precision, recall, _ = precision_recall_curve(targets, scores)
        roc_score = auc(false_positive_rate, true_positive_rate)
        average_precision = average_precision_score(targets, scores)
        curve_label = f"{label} ({averaging}, AUC={roc_score:.3f})"
        (line,) = roc_axis.plot(
            false_positive_rate,
            true_positive_rate,
            label=curve_label,
        )
        pr_axis.plot(
            recall,
            precision,
            color=line.get_color(),
            label=f"{label} ({averaging}, AP={average_precision:.3f})",
        )
        pr_axis.axhline(
            prevalence,
            color=line.get_color(),
            linestyle=":",
            alpha=0.35,
        )

    roc_axis.plot([0, 1], [0, 1], color="#555555", linestyle="--", label="Chance")
    roc_axis.set(
        xlabel="False-positive rate",
        ylabel="True-positive rate",
        xlim=(0, 1),
        ylim=(0, 1),
        title="Pooled out-of-fold ROC",
    )
    pr_axis.set(
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1),
        ylim=(0, 1),
        title="Pooled out-of-fold precision-recall",
    )
    for axis in (roc_axis, pr_axis):
        if available:
            axis.legend(fontsize="small")
        else:
            axis.text(
                0.5,
                0.5,
                "No completed runs",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        axis.grid(alpha=0.2)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def _curve_inputs(
    run: SavedRun,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    n_classes = run.probabilities.shape[1]
    if n_classes == 2:
        targets = run.y_true
        scores = run.probabilities[:, 1]
        averaging = "positive class"
    else:
        targets = np.eye(n_classes, dtype=int)[run.y_true].ravel()
        scores = run.probabilities.ravel()
        averaging = "micro OVR"
    return targets, scores, float(np.mean(targets)), averaging


def plot_class_waveforms(dataset: EpochDataset) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    for class_index, label in enumerate(dataset.class_labels):
        waveform = dataset.X[dataset.y == class_index].mean(axis=(0, 1))
        ax.plot(dataset.times, waveform * 1e6, label=label)
    ax.set(xlabel="Time (s)", ylabel="Mean amplitude (uV)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_covariance_difference(dataset: EpochDataset) -> plt.Figure:
    covariances = []
    for class_index in range(len(dataset.class_ids)):
        class_covariances = [
            np.cov(epoch)
            for epoch in dataset.X[dataset.y == class_index]
        ]
        covariances.append(np.mean(class_covariances, axis=0))
    difference = covariances[0] - np.mean(covariances[1:], axis=0)
    fig, ax = plt.subplots(figsize=(6, 5))
    limit = float(np.max(np.abs(difference))) or 1.0
    image = ax.imshow(difference, cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.set(
        xticks=np.arange(len(dataset.channel_names)),
        yticks=np.arange(len(dataset.channel_names)),
        xticklabels=dataset.channel_names,
        yticklabels=dataset.channel_names,
        title="Class covariance contrast",
    )
    ax.tick_params(axis="x", rotation=90)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    return fig


def plot_band_feature_projection(dataset: EpochDataset, *, max_trials: int = 2_000) -> plt.Figure:
    indices = np.arange(min(len(dataset.y), max_trials))
    features, _ = band_summary_features(dataset.X[indices], dataset.sfreq)
    projection = PCA(n_components=2).fit_transform(features)
    fig, ax = plt.subplots(figsize=(6, 5))
    for class_index, label in enumerate(dataset.class_labels):
        mask = dataset.y[indices] == class_index
        ax.scatter(projection[mask, 0], projection[mask, 1], s=10, alpha=0.45, label=label)
    ax.set(xlabel="Feature PC1", ylabel="Feature PC2", title="Band-summary representation")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_channel_effects(dataset: EpochDataset, *, top_n: int = 15) -> plt.Figure:
    class_means = [
        dataset.X[dataset.y == class_index].mean(axis=0)
        for class_index in range(len(dataset.class_ids))
    ]
    reference = class_means[0] - np.mean(class_means[1:], axis=0)
    effects = np.mean(np.abs(reference), axis=1)
    indices = np.argsort(effects)[-min(top_n, len(effects)):]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.asarray(dataset.channel_names)[indices], effects[indices] * 1e6)
    ax.set(xlabel="Channel", ylabel="Mean absolute class effect (uV)")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    return fig
