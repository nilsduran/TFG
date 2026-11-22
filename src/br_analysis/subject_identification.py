from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import EpochDataset
from .metrics import ClassificationMetrics, classification_metrics
from .splits import chronological_holdout_indices


@dataclass(frozen=True)
class SubjectIdentificationResult:
    metrics: ClassificationMetrics
    subject_ids: tuple[str, ...]
    y_true: np.ndarray
    probabilities: np.ndarray
    chance_level: float
    pipeline_steps: tuple[str, ...]


def run_subject_identification(
    dataset: EpochDataset,
    *,
    holdout_fraction: float = 0.2,
    random_seed: int = 42,
) -> SubjectIdentificationResult:
    try:
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
    except ImportError as exc:
        raise ImportError(
            "Install the core project dependencies to run subject identification."
        ) from exc

    subject_ids = tuple(dataset.subject_ids)
    subject_to_class = {subject: index for index, subject in enumerate(subject_ids)}
    train_indices: list[np.ndarray] = []
    test_indices: list[np.ndarray] = []

    for subject in subject_ids:
        subject_indices = np.flatnonzero(dataset.subjects == subject)
        train_local, test_local = chronological_holdout_indices(
            len(subject_indices),
            holdout_fraction,
        )
        train_indices.append(subject_indices[train_local])
        test_indices.append(subject_indices[test_local])

    train = np.concatenate(train_indices)
    test = np.concatenate(test_indices)
    y_identity = np.array([subject_to_class[str(subject)] for subject in dataset.subjects])

    model = make_pipeline(
        Covariances(estimator="oas"),
        TangentSpace(metric="riemann"),
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2_000,
            random_state=random_seed,
        ),
    )
    model.fit(dataset.X[train], y_identity[train])
    probabilities = model.predict_proba(dataset.X[test])
    metrics = classification_metrics(
        y_identity[test],
        probabilities,
        n_classes=len(subject_ids),
    )
    return SubjectIdentificationResult(
        metrics=metrics,
        subject_ids=subject_ids,
        y_true=y_identity[test],
        probabilities=probabilities,
        chance_level=1.0 / len(subject_ids),
        pipeline_steps=(
            "oas_covariance",
            "riemann_tangent_space",
            "standard_scaler",
            "logistic_regression",
        ),
    )
