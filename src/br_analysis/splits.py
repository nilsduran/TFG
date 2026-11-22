from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

EVALUATION_PROTOCOL = "80% development selection; 20% final chronological holdout"


@dataclass(frozen=True)
class SplitPlan:
    n_splits: int = 5
    holdout_fraction: float = 0.2
    temporal_blocks: int = 10
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        if not 0.0 <= self.holdout_fraction < 1.0:
            raise ValueError("holdout_fraction must be in [0, 1).")
        if self.temporal_blocks < 2:
            raise ValueError("temporal_blocks must be at least 2.")


def chronological_holdout_indices(
    n_samples: int,
    fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    if n_samples < 2:
        raise ValueError("At least two samples are required.")
    if fraction <= 0:
        return np.arange(n_samples), np.array([], dtype=int)
    split_index = int(np.floor((1.0 - fraction) * n_samples))
    split_index = max(1, min(split_index, n_samples - 1))
    return np.arange(split_index), np.arange(split_index, n_samples)


def chronological_block_labels(n_samples: int, n_blocks: int = 10) -> np.ndarray:
    if n_samples < 1:
        return np.array([], dtype=int)
    effective_blocks = max(1, min(n_blocks, n_samples))
    labels = np.empty(n_samples, dtype=int)
    for block, indices in enumerate(np.array_split(np.arange(n_samples), effective_blocks)):
        labels[indices] = block
    return labels


def development_splits(
    y: np.ndarray,
    plan: SplitPlan = SplitPlan(),
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    y = np.asarray(y)
    development_indices, holdout_indices = chronological_holdout_indices(
        len(y),
        plan.holdout_fraction,
    )
    y_development = y[development_indices]
    groups = chronological_block_labels(len(y_development), plan.temporal_blocks)
    class_counts = np.bincount(y_development)
    nonzero_counts = class_counts[class_counts > 0]
    n_splits = min(plan.n_splits, len(np.unique(groups)), int(nonzero_counts.min()))
    if n_splits < 2 or len(nonzero_counts) < 2:
        return [], holdout_indices

    expected_classes = np.unique(y_development)
    for candidate_splits in range(n_splits, 1, -1):
        splitter = StratifiedGroupKFold(
            n_splits=candidate_splits,
            shuffle=True,
            random_state=plan.random_seed,
        )
        local_splits = list(
            splitter.split(
                np.zeros(len(y_development)),
                y_development,
                groups,
            )
        )
        if all(
            np.array_equal(np.unique(y_development[train]), expected_classes)
            and np.array_equal(np.unique(y_development[test]), expected_classes)
            for train, test in local_splits
        ):
            return [
                (development_indices[train], development_indices[test])
                for train, test in local_splits
            ], holdout_indices
    return [], holdout_indices
