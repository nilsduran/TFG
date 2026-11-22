"""Group-level statistics for subject-wise decoding accuracies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    recalls = [np.mean(y_pred[y_true == c] == c) for c in range(n_classes) if np.any(y_true == c)]
    return float(np.mean(recalls))


def permutation_null(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    n_permutations: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """Balanced accuracies of fixed predictions against permuted holdout labels.

    Index 0 holds the observed (unpermuted) accuracy, as required by
    permutation-based prevalence inference.
    """
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations + 1)
    null[0] = balanced_accuracy(y_true, y_pred, n_classes)
    for index in range(1, n_permutations + 1):
        null[index] = balanced_accuracy(rng.permutation(y_true), y_pred, n_classes)
    return null


@dataclass(frozen=True)
class PrevalenceResult:
    global_p: float
    gamma0: float
    alpha: float
    n_subjects: int

    def summary(self) -> str:
        if self.gamma0 <= 0:
            return f"global null not rejected at alpha={self.alpha} (p={self.global_p:.3g})"
        return (
            f"information present in at least {100 * self.gamma0:.0f}% of the population "
            f"(alpha={self.alpha}, global p={self.global_p:.3g}, N={self.n_subjects})"
        )


def prevalence_inference(
    first_level: np.ndarray,
    n_second_level: int = 100_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> PrevalenceResult:
    """Permutation-based information prevalence inference (Allefeld et al., 2016).

    ``first_level`` has shape ``(subjects, permutations + 1)`` with the observed
    accuracy in column 0. Returns the global-null p-value of the minimum
    statistic and the largest prevalence ``gamma0`` such that the null
    "prevalence <= gamma0" is rejected.
    """
    first_level = np.asarray(first_level, dtype=float)
    n_subjects, n_first = first_level.shape
    observed_minimum = first_level[:, 0].min()
    rng = np.random.default_rng(seed)
    exceed = 1  # the observed configuration counts as one permutation
    batch = 10_000
    remaining = n_second_level - 1
    while remaining > 0:
        size = min(batch, remaining)
        columns = rng.integers(0, n_first, size=(size, n_subjects))
        minima = first_level[np.arange(n_subjects), columns].min(axis=1)
        exceed += int(np.sum(minima >= observed_minimum))
        remaining -= size
    global_p = exceed / n_second_level
    root_p = global_p ** (1.0 / n_subjects)
    root_alpha = alpha ** (1.0 / n_subjects)
    gamma0 = 0.0 if global_p > alpha else (root_alpha - root_p) / (1.0 - root_p)
    return PrevalenceResult(float(global_p), float(max(0.0, gamma0)), alpha, n_subjects)


def group_summary(accuracies: np.ndarray, chance: float = 0.5) -> dict[str, float]:
    accuracies = np.asarray(accuracies, dtype=float)
    wilcoxon = stats.wilcoxon(accuracies - chance, alternative="greater")
    return {
        "mean": float(accuracies.mean()),
        "sd": float(accuracies.std(ddof=1)),
        "sem": float(accuracies.std(ddof=1) / np.sqrt(len(accuracies))),
        "median": float(np.median(accuracies)),
        "above_chance": int(np.sum(accuracies > chance)),
        "wilcoxon_p": float(wilcoxon.pvalue),
        "n": int(len(accuracies)),
    }


def paired_comparison(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    difference = np.asarray(candidate, dtype=float) - np.asarray(reference, dtype=float)
    test = stats.wilcoxon(difference)
    return {
        "mean_gain": float(difference.mean()),
        "wins": int(np.sum(difference > 0)),
        "losses": int(np.sum(difference < 0)),
        "wilcoxon_p": float(test.pvalue),
    }
