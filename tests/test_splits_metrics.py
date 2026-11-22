from __future__ import annotations

import numpy as np

from br_analysis.metrics import classification_metrics
from br_analysis.splits import (
    SplitPlan,
    chronological_block_labels,
    development_splits,
)


def test_development_splits_keep_latest_trials_for_holdout() -> None:
    y = np.tile([0, 1], 30)
    splits, holdout = development_splits(
        y,
        SplitPlan(n_splits=3, holdout_fraction=0.2, temporal_blocks=6),
    )
    assert np.array_equal(holdout, np.arange(48, 60))
    assert splits
    assert all(np.intersect1d(test, holdout).size == 0 for _, test in splits)


def test_development_splits_keep_temporal_groups_disjoint() -> None:
    y = np.tile([0, 1], 50)
    plan = SplitPlan(n_splits=5, holdout_fraction=0.2, temporal_blocks=10)
    splits, _ = development_splits(y, plan)
    groups = chronological_block_labels(80, plan.temporal_blocks)

    assert len(splits) == 5
    for train, test in splits:
        assert set(groups[train]).isdisjoint(groups[test])
        assert set(y[test]) == {0, 1}


def test_development_splits_reduce_fold_count_for_clustered_rare_class() -> None:
    y = np.zeros(157, dtype=int)
    development_groups = np.array_split(np.arange(125), 10)
    y[development_groups[7][0]] = 1
    y[development_groups[9][:8]] = 1
    y[125:] = np.tile([0, 1], 16)
    plan = SplitPlan(n_splits=5, holdout_fraction=0.2, temporal_blocks=10)

    splits, _ = development_splits(y, plan)

    assert len(splits) == 2
    assert all(set(y[train]) == {0, 1} for train, _ in splits)
    assert all(set(y[test]) == {0, 1} for _, test in splits)


def test_multiclass_metrics() -> None:
    y_true = np.array([0, 1, 2, 0, 1, 2])
    probabilities = np.eye(3)[y_true] * 0.8 + 0.2 / 3
    metrics = classification_metrics(y_true, probabilities, n_classes=3)
    assert metrics.accuracy == 1.0
    assert metrics.balanced_accuracy == 1.0
    assert metrics.f1_macro == 1.0
    assert metrics.roc_auc == 1.0
