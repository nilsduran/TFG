from __future__ import annotations

import numpy as np

from br_analysis.explanations import (
    ExplanationResult,
    _normalize_shap_values,
    explanation_stability,
    stratified_background,
)


def test_stratified_background_uses_each_class() -> None:
    X = np.arange(60).reshape(20, 3)
    y = np.repeat([0, 1], 10)
    background, labels = stratified_background(X, y, per_class=3)
    assert background.shape == (6, 3)
    assert np.bincount(labels).tolist() == [3, 3]


def test_normalize_list_shap_output() -> None:
    values = [np.ones((4, 3)), np.full((4, 3), 2.0)]
    normalized = _normalize_shap_values(values, n_samples=4, n_classes=2)
    assert normalized.shape == (4, 2, 3)


def test_explanation_stability() -> None:
    values = np.arange(24, dtype=float).reshape(2, 2, 2, 3)
    result = ExplanationResult(
        values=values,
        base_values=np.array([]),
        samples=np.zeros((2, 2, 3)),
        class_ids=("a", "b"),
        method="test",
    )
    assert explanation_stability(result, result) == 1.0
