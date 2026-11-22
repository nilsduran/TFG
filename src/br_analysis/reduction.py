from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt
from sklearn.cluster import AgglomerativeClustering

DEFAULT_REDUCTION_BANDS = (
    ("theta", 5.0, 7.0),
    ("alpha", 8.0, 12.0),
)


def channel_adjacency(channel_names: Sequence[str]):
    import mne

    info = mne.create_info(list(channel_names), sfreq=1.0, ch_types="eeg")
    info.set_montage(
        mne.channels.make_standard_montage("standard_1005"),
        on_missing="ignore",
    )
    adjacency, names = mne.channels.find_ch_adjacency(info, ch_type="eeg")
    if list(names) != list(channel_names):
        indices = [list(names).index(channel) for channel in channel_names]
        adjacency = adjacency[indices][:, indices]
    return adjacency


def spatial_parcels(
    channel_names: Sequence[str],
    *,
    n_parcels: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    if not 1 <= n_parcels <= len(channel_names):
        raise ValueError("n_parcels must be between one and the number of channels.")
    coordinates = channel_coordinates(channel_names)
    labels = AgglomerativeClustering(
        n_clusters=n_parcels,
        connectivity=channel_adjacency(channel_names),
        linkage="ward",
    ).fit_predict(coordinates)
    matrix = np.zeros((n_parcels, len(channel_names)), dtype=np.float32)
    for parcel in range(n_parcels):
        members = labels == parcel
        matrix[parcel, members] = 1.0 / members.sum()
    return labels, matrix


def channel_coordinates(channel_names: Sequence[str]) -> np.ndarray:
    import mne

    positions = mne.channels.make_standard_montage(
        "standard_1005"
    ).get_positions()["ch_pos"]
    coordinates = np.asarray(
        [positions.get(channel, [np.nan, np.nan, np.nan]) for channel in channel_names],
        dtype=float,
    )
    if np.isnan(coordinates).any():
        raise ValueError("All channels require standard montage coordinates.")
    return coordinates


def apply_spatial_parcels(X: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.einsum("pc,nct->npt", matrix, X, optimize=True).astype(np.float32)


def parcel_filterbank(
    X: np.ndarray,
    sfreq: float,
    parcel_matrix: np.ndarray,
    *,
    bands: tuple[tuple[str, float, float], ...] = DEFAULT_REDUCTION_BANDS,
) -> np.ndarray:
    parcel_signals = apply_spatial_parcels(X, parcel_matrix)
    filtered = []
    for _, low, high in bands:
        sos = butter(4, (low, high), btype="bandpass", fs=sfreq, output="sos")
        filtered.append(sosfiltfilt(sos, parcel_signals, axis=-1))
    return np.concatenate(filtered, axis=1).astype(np.float32)


def parcel_band_features(
    X: np.ndarray,
    sfreq: float,
    parcel_matrix: np.ndarray,
    *,
    n_time_bins: int = 24,
    bands: tuple[tuple[str, float, float], ...] = DEFAULT_REDUCTION_BANDS,
) -> np.ndarray:
    filterbank = parcel_filterbank(
        X,
        sfreq,
        parcel_matrix,
        bands=bands,
    )
    envelope = np.abs(hilbert(filterbank, axis=-1))
    time_groups = np.array_split(np.arange(X.shape[-1]), n_time_bins)
    binned = np.stack(
        [envelope[:, :, indices].mean(axis=-1) for indices in time_groups],
        axis=-1,
    )
    return binned.reshape(len(X), -1).astype(np.float32)
