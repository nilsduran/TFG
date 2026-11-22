from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class MaxTResult:
    statistic: np.ndarray
    p_uncorrected: np.ndarray
    p_fwer: np.ndarray
    critical_value: float
    omnibus_p: float


def balanced_accuracy_curve(
    y_true: np.ndarray,
    predictions: np.ndarray,
    *,
    n_classes: int,
) -> np.ndarray:
    y_true = np.asarray(y_true)
    predictions = np.asarray(predictions)
    if predictions.ndim != 2 or predictions.shape[0] != len(y_true):
        raise ValueError("predictions must have shape (trials, times).")
    recalls = []
    for class_id in range(n_classes):
        mask = y_true == class_id
        if not mask.any():
            raise ValueError("Every class must occur in y_true.")
        recalls.append((predictions[mask] == class_id).mean(axis=0))
    return np.mean(recalls, axis=0)


def first_sustained(
    significant: np.ndarray,
    minimum_points: int,
) -> int | None:
    significant = np.asarray(significant, dtype=bool)
    if minimum_points < 1:
        raise ValueError("minimum_points must be at least one.")
    for index in range(len(significant) - minimum_points + 1):
        if significant[index : index + minimum_points].all():
            return index
    return None


def max_t_inference(
    deltas: np.ndarray,
    *,
    permutations: int,
    alpha: float,
    rng: np.random.Generator,
) -> MaxTResult:
    deltas = np.asarray(deltas, dtype=float)
    if deltas.ndim != 2 or deltas.shape[0] < 2:
        raise ValueError("deltas must have shape (subjects, times).")
    observed = _one_sample_t(deltas)
    null = np.empty((permutations, deltas.shape[1]), dtype=np.float32)
    for index in range(permutations):
        signs = rng.choice((-1.0, 1.0), size=(deltas.shape[0], 1))
        null[index] = _one_sample_t(deltas * signs)
    null_max = null.max(axis=1)
    p_uncorrected = (
        1 + (null >= observed[None, :]).sum(axis=0)
    ) / (permutations + 1)
    p_fwer = (
        1 + (null_max[:, None] >= observed[None, :]).sum(axis=0)
    ) / (permutations + 1)
    return MaxTResult(
        statistic=observed,
        p_uncorrected=p_uncorrected,
        p_fwer=p_fwer,
        critical_value=float(np.quantile(null_max, 1.0 - alpha)),
        omnibus_p=float(
            (1 + np.sum(null_max >= observed.max())) / (permutations + 1)
        ),
    )


def subject_max_t_pvalues(
    y_true: np.ndarray,
    predictions: np.ndarray,
    blocks: np.ndarray,
    *,
    n_classes: int,
    chance: float,
    permutations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    observed = balanced_accuracy_curve(
        y_true,
        predictions,
        n_classes=n_classes,
    )
    null_max = np.empty(permutations, dtype=np.float32)
    unique_blocks = np.unique(blocks)
    for permutation in range(permutations):
        permuted = np.asarray(y_true).copy()
        for block in unique_blocks:
            mask = blocks == block
            permuted[mask] = rng.permutation(permuted[mask])
        null = balanced_accuracy_curve(
            permuted,
            predictions,
            n_classes=n_classes,
        )
        null_max[permutation] = np.max(null - chance)
    return (
        1 + (
            null_max[:, None] >= (observed - chance)[None, :]
        ).sum(axis=0)
    ) / (permutations + 1)


def benjamini_yekutieli(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    if p_values.ndim != 1:
        raise ValueError("p_values must be one-dimensional.")
    count = len(p_values)
    if count == 0:
        return p_values.copy()
    order = np.argsort(p_values)
    ranked = p_values[order]
    harmonic = np.sum(1.0 / np.arange(1, count + 1))
    corrected = ranked * count * harmonic / np.arange(1, count + 1)
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    result = np.empty_like(corrected)
    result[order] = np.minimum(corrected, 1.0)
    return result


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    corrected = np.maximum.accumulate(
        ranked * np.arange(len(values), 0, -1)
    )
    result = np.empty_like(corrected)
    result[order] = np.minimum(corrected, 1.0)
    return result


def one_sided_sign_flip_pvalue(
    deltas: np.ndarray,
    *,
    permutations: int,
    rng: np.random.Generator,
) -> float:
    deltas = np.asarray(deltas, dtype=float)
    observed = float(deltas.mean())
    null = np.empty(permutations, dtype=float)
    for index in range(permutations):
        null[index] = np.mean(
            deltas * rng.choice((-1.0, 1.0), size=len(deltas))
        )
    return float((1 + np.sum(null >= observed)) / (permutations + 1))


def hierarchical_onset_bootstrap(
    subject_trials: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    n_classes: int,
    chance: float,
    critical_t: float,
    minimum_points: int,
    samples: int,
    trial_replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if not subject_trials:
        return np.full(samples, np.nan)
    trial_curves = []
    for y_true, predictions in subject_trials:
        curves = np.empty(
            (trial_replicates, predictions.shape[1]),
            dtype=np.float32,
        )
        class_indices = [
            np.flatnonzero(y_true == class_id)
            for class_id in range(n_classes)
        ]
        for replicate in range(trial_replicates):
            recalls = []
            for class_id, indices in enumerate(class_indices):
                selected = rng.choice(indices, size=len(indices), replace=True)
                recalls.append(
                    (predictions[selected] == class_id).mean(axis=0)
                )
            curves[replicate] = np.mean(recalls, axis=0)
        trial_curves.append(curves)

    onsets = np.full(samples, np.nan)
    n_subjects = len(trial_curves)
    for sample in range(samples):
        selected_subjects = rng.integers(0, n_subjects, size=n_subjects)
        curves = np.stack(
            [
                trial_curves[subject][
                    rng.integers(0, trial_replicates)
                ]
                for subject in selected_subjects
            ]
        )
        statistic = _one_sample_t(curves - chance)
        onset = first_sustained(
            statistic >= critical_t,
            minimum_points,
        )
        if onset is not None:
            onsets[sample] = onset
    return onsets


def one_sided_t_pvalues(deltas: np.ndarray) -> np.ndarray:
    statistic, p_two_sided = stats.ttest_1samp(
        deltas,
        popmean=0.0,
        axis=0,
        nan_policy="omit",
    )
    statistic = np.nan_to_num(statistic)
    p_two_sided = np.nan_to_num(p_two_sided, nan=1.0)
    return np.where(
        statistic >= 0,
        p_two_sided / 2.0,
        1.0 - p_two_sided / 2.0,
    )


def _one_sample_t(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=0)
    denominator = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
    statistic = np.divide(
        mean,
        denominator,
        out=np.zeros(values.shape[1], dtype=float),
        where=denominator > 0,
    )
    constant_effect = (denominator == 0) & (mean != 0)
    statistic[constant_effect] = np.sign(mean[constant_effect]) * np.inf
    return statistic
