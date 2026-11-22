from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from scipy import stats

from .registry import MODELS


def collect_metrics(
    artifacts_root: Path,
    *,
    task_ids: Sequence[str] | None = None,
    model_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    if not artifacts_root.exists():
        return pd.DataFrame()
    tasks = list(task_ids) if task_ids is not None else [
        path.name for path in artifacts_root.iterdir() if path.is_dir()
    ]
    for task_id in tasks:
        task_directory = artifacts_root / task_id
        if not task_directory.exists():
            continue
        models = list(model_ids) if model_ids is not None else [
            path.name for path in task_directory.iterdir() if path.is_dir()
        ]
        for model_id in models:
            directory = task_directory / model_id
            metrics_path = directory / "metrics.csv"
            metadata_path = directory / "metadata.json"
            if not metrics_path.exists() or not metadata_path.exists():
                continue
            frame = pd.read_csv(metrics_path)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            frame.insert(0, "model_id", metadata["model_id"])
            frame.insert(0, "task_id", metadata["task_id"])
            frame["n_classes"] = len(metadata["class_ids"])
            frame["chance_level"] = 1.0 / frame["n_classes"]
            frame["evaluation_protocol"] = metadata.get("extra", {}).get(
                "evaluation_protocol",
                "legacy development CV; final 20% not evaluated",
            )
            rows.append(frame)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def model_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics.copy()
    subjects = subject_metrics(metrics)
    summary = (
        subjects.groupby(["task_id", "model_id"], as_index=False)
        .agg(
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            f1_macro_mean=("f1_macro", "mean"),
            f1_macro_std=("f1_macro", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            n_subjects=("subject", "nunique"),
            n_classes=("n_classes", "first"),
            chance_level=("chance_level", "first"),
            evaluation_protocol=("evaluation_protocol", "first"),
        )
    )
    sem = (
        summary["balanced_accuracy_std"].fillna(0.0)
        / summary["n_subjects"].pow(0.5)
    )
    critical = summary["n_subjects"].map(
        lambda count: stats.t.ppf(0.975, max(int(count) - 1, 1))
    )
    summary["balanced_accuracy_ci_low"] = (
        summary["balanced_accuracy_mean"] - critical * sem
    )
    summary["balanced_accuracy_ci_high"] = (
        summary["balanced_accuracy_mean"] + critical * sem
    )
    summary["delta_chance"] = (
        summary["balanced_accuracy_mean"] - summary["chance_level"]
    )
    return summary.sort_values(
        ["task_id", "balanced_accuracy_mean"],
        ascending=[True, False],
    )


def subject_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics.copy()
    aggregations: dict[str, tuple[str, str]] = {
        column: (column, "mean")
        for column in (
            "accuracy",
            "balanced_accuracy",
            "f1_macro",
            "roc_auc",
        )
        if column in metrics
    }
    aggregations["n_classes"] = ("n_classes", "first")
    aggregations["chance_level"] = ("chance_level", "first")
    aggregations["evaluation_protocol"] = ("evaluation_protocol", "first")
    if "n_samples" in metrics:
        aggregations["n_samples"] = ("n_samples", "sum")
    return metrics.groupby(
        ["task_id", "model_id", "subject"],
        as_index=False,
    ).agg(**aggregations)


def classical_task_evidence(
    artifacts_root: Path,
    task_id: str,
    *,
    minimum_models: int = 2,
    minimum_effect: float = 0.02,
) -> bool | None:
    classical_models = [
        model_id
        for model_id, model in MODELS.items()
        if model.family == "sklearn"
    ]
    summary = model_summary(
        collect_metrics(
            artifacts_root,
            task_ids=[task_id],
            model_ids=classical_models,
        )
    )
    if len(summary) < minimum_models:
        return None
    return bool(
        (
            (summary["delta_chance"] >= minimum_effect)
            & (
                summary["balanced_accuracy_ci_low"]
                > summary["chance_level"]
            )
        ).any()
    )
