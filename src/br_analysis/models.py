from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class SafeLogisticRegressionCV(LogisticRegressionCV):
    def fit(self, X, y, sample_weight=None, **params):
        _, counts = np.unique(y, return_counts=True)
        min_class_size = int(counts.min())
        if min_class_size < 2:
            classifier = LogisticRegression(
                C=1.0,
                solver=self.solver,
                max_iter=self.max_iter,
                random_state=self.random_state,
                class_weight=self.class_weight,
                **_logistic_regression_compatibility_options(),
            )
            classifier.fit(X, y, sample_weight=sample_weight, **params)
            self.classes_ = classifier.classes_
            self.coef_ = classifier.coef_
            self.intercept_ = classifier.intercept_
            self.n_iter_ = classifier.n_iter_
            self.C_ = np.array([classifier.C])
            return self

        original_cv = self.cv
        cv = 5 if original_cv is None else original_cv
        if isinstance(cv, int):
            self.cv = max(2, min(cv, min_class_size))
        try:
            super().fit(X, y, sample_weight=sample_weight, **params)
        finally:
            self.cv = original_cv
        return self


def _logistic_cv_compatibility_options() -> dict[str, object]:
    if "use_legacy_attributes" in inspect.signature(LogisticRegressionCV).parameters:
        return {
            "use_legacy_attributes": False,
            "l1_ratios": (0.0,),
        }
    return {"penalty": "l2"}


def _logistic_regression_compatibility_options() -> dict[str, object]:
    penalty_default = inspect.signature(LogisticRegression).parameters["penalty"].default
    if penalty_default == "deprecated":
        return {"l1_ratio": 0.0}
    return {"penalty": "l2"}


class ModelFactory(Protocol):
    name: str

    def make(self, selection_splits=None):
        ...


@dataclass(frozen=True)
class LogRegModelFactory:
    random_seed: int = 42
    name: str = "logreg_cv"

    def make(self, selection_splits=None):
        return make_pipeline(
            StandardScaler(),
            SafeLogisticRegressionCV(
                Cs=list(np.logspace(-4, 4, 9)),
                cv=selection_splits if selection_splits else 3,
                solver="lbfgs",
                scoring="f1_macro",
                max_iter=2000,
                random_state=self.random_seed,
                refit=True,
                class_weight="balanced",
                **_logistic_cv_compatibility_options(),
            ),
        )


@dataclass(frozen=True)
class SVCModelFactory:
    random_seed: int = 42
    name: str = "svc"
    C: float = 1.0

    def make(self, selection_splits=None):
        del selection_splits
        return make_pipeline(
            StandardScaler(),
            SVC(
                C=self.C,
                kernel="rbf",
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=self.random_seed,
            ),
        )


@dataclass(frozen=True)
class RiemannLdaModelFactory:
    n_classes: int = 2
    name: str = "riemann_lda"

    def make(self, selection_splits=None):
        del selection_splits
        try:
            from pyriemann.estimation import Covariances
            from pyriemann.tangentspace import TangentSpace
        except ImportError as exc:
            raise ImportError("Install the core project dependencies to use RiemannLdaModelFactory.") from exc

        priors = np.full(self.n_classes, 1.0 / self.n_classes)
        return make_pipeline(
            Covariances(estimator="oas"),
            TangentSpace(metric="riemann"),
            LinearDiscriminantAnalysis(
                solver="lsqr",
                shrinkage="auto",
                priors=priors,
            ),
        )
