from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


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
            LogisticRegressionCV(
                Cs=list(np.logspace(-4, 4, 9)),
                cv=3,
                penalty="l2",
                solver="liblinear",
                scoring="accuracy",
                max_iter=2000,
                random_state=self.random_seed,
                refit=True,
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
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        )
