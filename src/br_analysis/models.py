from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class SafeLogisticRegressionCV(LogisticRegressionCV):
    """LogisticRegressionCV subclass that dynamically adjusts cv splits

    or falls back to LogisticRegression if a class has fewer than 2 members,
    avoiding StratifiedKFold validation errors.
    """

    def fit(self, X, y, sample_weight=None, **params):
        min_class_size = int(np.bincount(y).min())
        if min_class_size < 2:
            from sklearn.linear_model import LogisticRegression

            clf = LogisticRegression(
                C=1.0,
                penalty=self.penalty,
                solver=self.solver,
                max_iter=self.max_iter,
                random_state=self.random_state,
                class_weight=self.class_weight,
            )
            clf.fit(X, y, sample_weight=sample_weight, **params)
            self.classes_ = clf.classes_
            self.coef_ = clf.coef_
            self.intercept_ = clf.intercept_
            self.n_iter_ = clf.n_iter_
            self.C_ = np.array([clf.C])
            return self

        original_cv = self.cv
        cv_val = 5 if original_cv is None else original_cv
        if isinstance(cv_val, int):
            actual_cv = max(2, min(cv_val, min_class_size))
            self.cv = actual_cv
        try:
            super().fit(X, y, sample_weight=sample_weight, **params)
        finally:
            self.cv = original_cv
        return self


class ModelFactory(Protocol):
    """Small interface future model runners can implement."""

    name: str

    def make(self):
        ...


@dataclass(frozen=True)
class LogRegModelFactory:
    random_seed: int = 42
    name: str = "logreg_cv"

    def make(self):
        return make_pipeline(
            StandardScaler(),
            SafeLogisticRegressionCV(
                Cs=list(np.logspace(-4, 4, 9)),
                cv=3,
                penalty="l2",
                solver="liblinear",
                scoring="accuracy",
                max_iter=2000,
                random_state=self.random_seed,
                refit=True,
                class_weight="balanced",
            ),
        )



@dataclass(frozen=True)
class RiemannLdaModelFactory:
    name: str = "riemann_lda"

    def make(self):
        try:
            from pyriemann.estimation import Covariances
            from pyriemann.tangentspace import TangentSpace
        except ImportError as exc:
            raise ImportError("RiemannLdaModelFactory requires pyriemann. Install it before running this experiment.") from exc

        return make_pipeline(
            Covariances(estimator="oas"),
            TangentSpace(metric="riemann"),
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=np.array([0.5, 0.5])),
        )
