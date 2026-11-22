from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

DEFAULT_BANDS: Mapping[str, tuple[float, float]] = {
    "theta": (5.0, 7.0),
    "alpha": (8.0, 12.0),
}


def band_summary_features(
    X: np.ndarray,
    sfreq: float,
    *,
    bands: Mapping[str, tuple[float, float]] = DEFAULT_BANDS,
    channel_indices: Sequence[int] | None = None,
) -> tuple[np.ndarray, list[str]]:
    selected = X if channel_indices is None else X[:, channel_indices]
    blocks: list[np.ndarray] = []
    feature_names: list[str] = []
    n_channels = selected.shape[1]

    for band_name, (low, high) in bands.items():
        sos = butter(4, (low, high), btype="bandpass", fs=sfreq, output="sos")
        filtered = sosfiltfilt(sos, selected, axis=-1)
        envelope = np.abs(hilbert(filtered, axis=-1)).mean(axis=-1)
        log_power = np.log(np.maximum(np.square(filtered).mean(axis=-1), np.finfo(np.float32).eps))
        temporal_std = filtered.std(axis=-1)
        blocks.extend((envelope, log_power, temporal_std))
        for statistic in ("mean_envelope", "log_power", "temporal_std"):
            feature_names.extend(
                f"{band_name}_channel_{channel_index}_{statistic}"
                for channel_index in range(n_channels)
            )

    return np.concatenate(blocks, axis=1).astype(np.float32), feature_names


def mean_envelope_feature(
    X: np.ndarray,
    sfreq: float,
    *,
    band: tuple[float, float],
) -> np.ndarray:
    if X.shape[1] != 1:
        raise ValueError("mean_envelope_feature expects exactly one channel.")
    sos = butter(4, band, btype="bandpass", fs=sfreq, output="sos")
    filtered = sosfiltfilt(sos, X, axis=-1)
    return np.abs(hilbert(filtered, axis=-1)).mean(axis=-1).astype(np.float32)
