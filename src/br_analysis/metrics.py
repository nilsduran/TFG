from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    balanced_accuracy: float
    f1_macro: float
    roc_auc: float
    n_samples: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_classes: int,
) -> ClassificationMetrics:
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 2 or probabilities.shape != (len(y_true), n_classes):
        raise ValueError("probabilities must have shape (n_samples, n_classes).")
    predictions = probabilities.argmax(axis=1)
    try:
        if n_classes == 2:
            auc = float(roc_auc_score(y_true, probabilities[:, 1]))
        else:
            auc = float(
                roc_auc_score(
                    y_true,
                    probabilities,
                    multi_class="ovr",
                    average="macro",
                    labels=np.arange(n_classes),
                )
            )
    except ValueError:
        auc = float("nan")
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, predictions)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, predictions)),
        f1_macro=float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        roc_auc=auc,
        n_samples=len(y_true),
    )
