from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import hilbert


@dataclass(frozen=True)
class HilbertMeanEnvelope:
    """Feature used by the current single-channel chance-level notebooks."""

    name: str = "hilbert_mean_envelope"

    def transform(self, X: np.ndarray) -> np.ndarray:
        envelope = np.abs(hilbert(X, axis=-1))
        return np.mean(envelope, axis=-1, keepdims=True)


@dataclass(frozen=True)
class MeanPower:
    name: str = "mean_power"

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.mean(X**2, axis=-1, keepdims=True)
