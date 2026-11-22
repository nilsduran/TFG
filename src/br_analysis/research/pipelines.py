"""Candidate decoding pipelines for the post-thesis benchmark.

Every pipeline is an sklearn estimator that takes epochs shaped
``(trials, channels, times)`` restricted to the stimulus window and returns
class probabilities. Hyperparameters are fixed or selected internally from
training data only, so pipelines can be compared on the same outer splits.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from pyriemann.estimation import Covariances, XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from scipy.signal import butter, sosfiltfilt
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler

BANDS: dict[str, tuple[float, float]] = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}


# --------------------------------------------------------------------------- transformers


class BandpassFilter(BaseEstimator, TransformerMixin):
    """Zero-phase Butterworth band-pass applied along the time axis."""

    def __init__(self, band: tuple[float, float], sfreq: float, order: int = 4):
        self.band = band
        self.sfreq = sfreq
        self.order = order

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        sos = butter(self.order, self.band, btype="bandpass", fs=self.sfreq, output="sos")
        return sosfiltfilt(sos, X, axis=-1)


class TimeWindow(BaseEstimator, TransformerMixin):
    """Select samples between ``tmin`` and ``tmax`` seconds from the first sample."""

    def __init__(self, tmin: float, tmax: float, sfreq: float):
        self.tmin = tmin
        self.tmax = tmax
        self.sfreq = sfreq

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        start = int(round(self.tmin * self.sfreq))
        stop = int(round(self.tmax * self.sfreq))
        return X[..., start:stop]


class TimeDelayEmbedding(BaseEstimator, TransformerMixin):
    """Stack ``order`` delayed copies of every channel (augmented covariance input)."""

    def __init__(self, order: int = 3, lag: int = 2):
        self.order = order
        self.lag = lag

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        span = (self.order - 1) * self.lag
        n_times = X.shape[-1] - span
        copies = [X[..., k * self.lag : k * self.lag + n_times] for k in range(self.order)]
        return np.concatenate(copies, axis=1)


class BinnedAmplitudes(BaseEstimator, TransformerMixin):
    """Average voltage in consecutive bins and flatten to ``channels x bins`` features."""

    def __init__(self, bin_seconds: float, sfreq: float):
        self.bin_seconds = bin_seconds
        self.sfreq = sfreq

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        width = max(1, int(round(self.bin_seconds * self.sfreq)))
        n_bins = X.shape[-1] // width
        trimmed = X[..., : n_bins * width]
        binned = trimmed.reshape(*X.shape[:-1], n_bins, width).mean(axis=-1)
        return binned.reshape(len(X), -1)


class ProbabilityAverager(BaseEstimator):
    """Soft-voting ensemble of independently fitted pipelines."""

    def __init__(self, estimators: Sequence[BaseEstimator]):
        self.estimators = estimators

    def fit(self, X, y):
        self.fitted_ = [clone(estimator).fit(X, y) for estimator in self.estimators]
        self.classes_ = self.fitted_[0].classes_
        return self

    def predict_proba(self, X):
        return np.mean([estimator.predict_proba(X) for estimator in self.fitted_], axis=0)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# --------------------------------------------------------------------------- classifiers


def shrinkage_lda() -> LinearDiscriminantAnalysis:
    # Equal priors: the outer metric is balanced accuracy.
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5])


def balanced_logistic(C: float = 1.0) -> LogisticRegression:
    return LogisticRegression(C=C, class_weight="balanced", max_iter=2000)


def _covariance_branch(prefix: Sequence[BaseEstimator], metric: str = "riemann"):
    return make_pipeline(*prefix, Covariances(estimator="oas"), TangentSpace(metric=metric))


# --------------------------------------------------------------------------- candidate registry


@dataclass(frozen=True)
class Candidate:
    id: str
    preset: str
    description: str
    build: Callable[[float], BaseEstimator]


def _riemann_baseline(sfreq: float):
    return make_pipeline(Covariances(estimator="oas"), TangentSpace(metric="riemann"), shrinkage_lda())


def _filterbank_ts(sfreq: float):
    branches = [
        (name, _covariance_branch([BandpassFilter(band, sfreq)]))
        for name, band in BANDS.items()
    ]
    branches.append(("broad", _covariance_branch([])))
    return make_pipeline(FeatureUnion(branches), StandardScaler(), balanced_logistic(C=0.05))


def _augmented_cov(sfreq: float):
    return make_pipeline(
        TimeDelayEmbedding(order=3, lag=max(1, int(round(0.016 * sfreq)))),
        Covariances(estimator="oas"),
        TangentSpace(metric="riemann"),
        StandardScaler(),
        # ~16k tangent features: an L2 logistic model avoids LDA's dense feature covariance.
        balanced_logistic(C=0.01),
    )


def _subwindow_ts(sfreq: float):
    windows = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5)]
    branches = [
        (f"w{index}", _covariance_branch([TimeWindow(tmin, tmax, sfreq)]))
        for index, (tmin, tmax) in enumerate(windows)
    ]
    return make_pipeline(FeatureUnion(branches), StandardScaler(), balanced_logistic(C=0.05))


def _xdawn_ts(sfreq: float):
    return make_pipeline(
        XdawnCovariances(nfilter=4, estimator="oas", xdawn_estimator="oas"),
        TangentSpace(metric="riemann"),
        balanced_logistic(C=1.0),
    )


def _erp_lda(sfreq: float):
    return make_pipeline(BinnedAmplitudes(0.05, sfreq), StandardScaler(), shrinkage_lda())


def _xdawn_plus_cov(sfreq: float):
    return ProbabilityAverager([_xdawn_ts(sfreq), _erp_lda(sfreq)])


CANDIDATES: dict[str, Candidate] = {
    candidate.id: candidate
    for candidate in [
        Candidate("riemann", "broadband", "Thesis baseline: OAS covariance + tangent space + shrinkage LDA", _riemann_baseline),
        Candidate("fb_ts", "broadband", "Filter-bank tangent space (theta, alpha, beta, broad) + logistic regression", _filterbank_ts),
        Candidate("aug_cov", "broadband", "Augmented covariance (time-delay embedding) + tangent space + LDA", _augmented_cov),
        Candidate("subwin_ts", "broadband", "Tangent space of three 500 ms sub-windows + logistic regression", _subwindow_ts),
        Candidate("xdawn_ts", "erp", "xDAWN ERP-prototype covariances + tangent space + logistic regression", _xdawn_ts),
        Candidate("erp_lda", "erp", "Binned ERP amplitudes (50 ms) + shrinkage LDA", _erp_lda),
        Candidate("xdawn_erp_ens", "erp", "Soft-vote of xdawn_ts and erp_lda", _xdawn_plus_cov),
    ]
}
