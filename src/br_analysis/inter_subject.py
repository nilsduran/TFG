from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .artifacts import implementation_signature, stable_signature
from .config import ProjectPaths
from .data import data_inventory_signature, load_epoch_dataset
from .metrics import classification_metrics
from .splits import chronological_holdout_indices
from .tasks import TASKS, get_task


@dataclass(frozen=True)
class InterSubjectConfig:
    holdout_fraction: float = 0.2
    permutations: int = 2000
    random_seed: int = 42


def run_inter_subject_analysis(
    *,
    paths: ProjectPaths | None = None,
    task_ids: Sequence[str] | None = None,
    config: InterSubjectConfig = InterSubjectConfig(),
    force: bool = False,
) -> Path:
    paths = paths or ProjectPaths.from_env("inter_subject")
    selected_tasks = list(task_ids) if task_ids is not None else list(TASKS)
    root = paths.artifacts_dir / "inter_subject"
    root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, task_id in enumerate(selected_tasks, start=1):
        print(f"[{index}/{len(selected_tasks)}] inter-subject / {task_id}")
        summaries.append(
            _run_task(paths, task_id, config=config, force=force)
        )
    pd.DataFrame(summaries).to_csv(root / "summary.csv", index=False)
    return root


def _run_task(
    paths: ProjectPaths,
    task_id: str,
    *,
    config: InterSubjectConfig,
    force: bool,
) -> dict[str, object]:
    directory = paths.artifacts_dir / "inter_subject" / task_id
    summary_path = directory / "summary.json"
    from pyriemann.estimation import Covariances
    from pyriemann.utils.base import invsqrtm, logm

    task = get_task(task_id)
    run_signature = stable_signature(
        {
            "task": asdict(task),
            "config": asdict(config),
            "data": data_inventory_signature(paths),
            "implementation": implementation_signature(
                (
                    "data.py",
                    "inter_subject.py",
                    "metrics.py",
                    "splits.py",
                    "tasks.py",
                )
            ),
        }
    )
    if summary_path.is_file() and not force:
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous.get("run_signature") == run_signature:
            print(f"Reusing inter-subject analysis: {directory}")
            return previous
    dataset = load_epoch_dataset(paths, task, "broadband")
    covariances = Covariances(estimator="oas").fit_transform(dataset.X)
    raw_logs = logm(covariances)
    subjects = dataset.subject_ids
    n_classes = task.n_classes
    chance = 1.0 / n_classes

    aligned_logs = np.empty_like(raw_logs)
    development_by_subject: dict[str, np.ndarray] = {}
    holdout_by_subject: dict[str, np.ndarray] = {}
    skipped = []
    for subject in subjects:
        indices = np.flatnonzero(dataset.subjects == subject)
        development_local, holdout_local = chronological_holdout_indices(
            len(indices),
            config.holdout_fraction,
        )
        development = indices[development_local]
        holdout = indices[holdout_local]
        expected = np.arange(n_classes)
        if (
            not len(holdout)
            or not np.array_equal(np.unique(dataset.y[development]), expected)
            or not np.array_equal(np.unique(dataset.y[holdout]), expected)
        ):
            skipped.append(subject)
            continue
        development_by_subject[subject] = development
        holdout_by_subject[subject] = holdout

        subject_reference = covariances[indices].mean(axis=0)
        whitening = invsqrtm(subject_reference)
        aligned = np.einsum(
            "ij,njk,lk->nil",
            whitening,
            covariances[indices],
            whitening,
            optimize=True,
        )
        aligned_logs[indices] = logm(aligned)

    valid_subjects = [
        subject for subject in subjects if subject not in skipped
    ]
    rows = []
    prediction_rows = []
    rng = np.random.default_rng(config.random_seed)
    for target_index, target in enumerate(valid_subjects, start=1):
        print(f"  [{target_index}/{len(valid_subjects)}] target={target}")
        source = np.flatnonzero(dataset.subjects != target)
        target_development = development_by_subject[target]
        target_holdout = holdout_by_subject[target]

        target_reference = covariances[target_development].mean(axis=0)
        whitening = invsqrtm(target_reference)
        target_aligned = np.einsum(
            "ij,njk,lk->nil",
            whitening,
            covariances[target_holdout],
            whitening,
            optimize=True,
        )
        target_aligned_logs = logm(target_aligned)

        variants = {
            "pooled_logeuclid_mdm": (
                raw_logs[source],
                raw_logs[target_holdout],
            ),
            "ea_logeuclid_mdm": (
                aligned_logs[source],
                target_aligned_logs,
            ),
        }
        for variant, (source_logs, target_logs) in variants.items():
            probabilities = _logeuclidean_probabilities(
                source_logs,
                dataset.y[source],
                target_logs,
                n_classes=n_classes,
            )
            metrics = classification_metrics(
                dataset.y[target_holdout],
                probabilities,
                n_classes=n_classes,
            )
            p_value = _prediction_permutation_pvalue(
                dataset.y[target_holdout],
                probabilities.argmax(axis=1),
                observed=metrics.balanced_accuracy,
                permutations=config.permutations,
                rng=rng,
            )
            rows.append(
                {
                    "task_id": task_id,
                    "variant": variant,
                    "subject": target,
                    "chance_level": chance,
                    "p_value": p_value,
                    **metrics.to_dict(),
                }
            )
            for trial, y_true, probability in zip(
                dataset.trial_indices[target_holdout],
                dataset.y[target_holdout],
                probabilities,
                strict=True,
            ):
                prediction_rows.append(
                    {
                        "task_id": task_id,
                        "variant": variant,
                        "subject": target,
                        "trial_index": int(trial),
                        "y_true": int(y_true),
                        **{
                            f"probability_{class_index}": float(value)
                            for class_index, value in enumerate(probability)
                        },
                    }
                )

    metrics = pd.DataFrame(rows)
    metrics["p_fdr"] = _fdr_by_variant(metrics)
    predictions = pd.DataFrame(prediction_rows)
    paired = metrics.pivot(
        index="subject",
        columns="variant",
        values="balanced_accuracy",
    ).dropna()
    difference = (
        paired["ea_logeuclid_mdm"] - paired["pooled_logeuclid_mdm"]
    )
    if np.allclose(difference, 0):
        paired_p = 1.0
    else:
        paired_p = float(
            stats.wilcoxon(difference, alternative="greater").pvalue
        )
    aligned = metrics[metrics["variant"] == "ea_logeuclid_mdm"]
    summary = {
        "task_id": task_id,
        "n_subjects": len(valid_subjects),
        "chance_level": chance,
        "pooled_balanced_accuracy": float(
            metrics.loc[
                metrics["variant"] == "pooled_logeuclid_mdm",
                "balanced_accuracy",
            ].mean()
        ),
        "aligned_balanced_accuracy": float(
            aligned["balanced_accuracy"].mean()
        ),
        "aligned_f1_macro": float(aligned["f1_macro"].mean()),
        "aligned_subjects_above_chance": int(
            (aligned["balanced_accuracy"] > chance).sum()
        ),
        "aligned_subjects_significant_fdr": int(
            (aligned["p_fdr"] <= 0.05).sum()
        ),
        "alignment_gain": float(difference.mean()),
        "alignment_paired_p": paired_p,
        "adaptation": (
            "Unsupervised target alignment uses only the first 80% of target "
            "trials; labels and all final holdout trials remain unused."
        ),
        "config": asdict(config),
        "run_signature": run_signature,
    }
    directory.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(directory / "metrics.csv", index=False)
    predictions.to_csv(directory / "predictions.csv", index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot_inter_subject(metrics, summary, directory / "comparison.png")
    return summary


def _logeuclidean_probabilities(
    source_logs: np.ndarray,
    source_y: np.ndarray,
    target_logs: np.ndarray,
    *,
    n_classes: int,
) -> np.ndarray:
    class_means = np.stack(
        [source_logs[source_y == class_id].mean(axis=0) for class_id in range(n_classes)]
    )
    distances = np.sqrt(
        np.square(
            target_logs[:, None, :, :] - class_means[None, :, :, :]
        ).sum(axis=(2, 3))
    )
    scale = np.maximum(np.median(distances), 1e-8)
    logits = -distances / scale
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def _prediction_permutation_pvalue(
    y_true: np.ndarray,
    predicted: np.ndarray,
    *,
    observed: float,
    permutations: int,
    rng: np.random.Generator,
) -> float:
    classes, counts = np.unique(y_true, return_counts=True)
    null = np.empty(permutations, dtype=float)
    for index in range(permutations):
        permuted = rng.permutation(y_true)
        null[index] = np.mean(
            [
                np.mean(predicted[permuted == class_id] == class_id)
                for class_id in classes
            ]
        )
    return float((1 + np.sum(null >= observed)) / (permutations + 1))


def _fdr_by_variant(metrics: pd.DataFrame) -> np.ndarray:
    adjusted = np.ones(len(metrics), dtype=float)
    for variant in metrics["variant"].unique():
        indices = np.flatnonzero(metrics["variant"].to_numpy() == variant)
        p_values = metrics.iloc[indices]["p_value"].to_numpy()
        order = np.argsort(p_values)
        ranked = p_values[order]
        corrected = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
        corrected = np.minimum.accumulate(corrected[::-1])[::-1]
        variant_adjusted = np.empty_like(corrected)
        variant_adjusted[order] = np.minimum(corrected, 1.0)
        adjusted[indices] = variant_adjusted
    return adjusted


def _plot_inter_subject(
    metrics: pd.DataFrame,
    summary: dict[str, object],
    destination: Path,
) -> None:
    paired = metrics.pivot(
        index="subject",
        columns="variant",
        values="balanced_accuracy",
    ).dropna()
    figure, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for _, row in paired.iterrows():
        ax.plot(
            [0, 1],
            [row["pooled_logeuclid_mdm"], row["ea_logeuclid_mdm"]],
            color="#a9b7bd",
            alpha=0.55,
        )
    ax.scatter(
        np.zeros(len(paired)),
        paired["pooled_logeuclid_mdm"],
        color="#6c757d",
        label="Pooled MDM",
    )
    ax.scatter(
        np.ones(len(paired)),
        paired["ea_logeuclid_mdm"],
        color="#176b87",
        label="EA + MDM",
    )
    ax.axhline(
        float(summary["chance_level"]),
        color="#9c2f2f",
        linestyle="--",
        label="Chance",
    )
    ax.set_xticks([0, 1], ["Pooled", "EA aligned"])
    ax.set_ylabel("Balanced accuracy")
    ax.set_title(f"{summary['task_id']}: inter-subject transfer")
    ax.legend()
    figure.savefig(destination, dpi=160)
    plt.close(figure)
