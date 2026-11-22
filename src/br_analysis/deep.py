from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import mne
import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch

from .config import BinaryTaskConfig, ProjectPaths
from .data import load_epoch_dataset
from .datasets import LoadedDataset
from .preprocessing import PreprocessingPreset


@dataclass(frozen=True)
class RawEpochSpec:
    picks: Sequence[str] | None = None
    bandpass: tuple[float, float] | None = None
    resample_sfreq: float | None = None
    analysis_window: tuple[float, float] | None = None
    peak_align_all_channels: bool = False


def load_raw_epoch_dataset(
    paths: ProjectPaths,
    task: BinaryTaskConfig,
    spec: RawEpochSpec,
    *,
    verbose: bool = True,
) -> LoadedDataset:
    """Load EEG epochs without subject selection or class undersampling."""
    analysis_window = spec.analysis_window if spec.analysis_window is not None else task.analysis_window
    selected_task = task.with_window(analysis_window)
    preset = PreprocessingPreset(
        id="raw_epochs",
        picks=tuple(spec.picks) if spec.picks is not None else None,
        bandpass=spec.bandpass,
        resample_sfreq=spec.resample_sfreq,
    )
    dataset = load_epoch_dataset(paths, selected_task, preset, verbose=verbose)
    return LoadedDataset(
        X=dataset.X,
        y=dataset.y,
        subjects=dataset.subjects,
        channel_names=dataset.channel_names,
        times=dataset.times,
        skipped=dataset.skipped,
        sfreq=dataset.sfreq,
        chs_info=[dict(channel) for channel in dataset.info["chs"]] if dataset.info is not None else None,
        info=dataset.info,
        trial_indices=dataset.trial_indices,
        class_ids=dataset.class_ids,
    )


def temporal_band_features(
    X: np.ndarray,
    sfreq: float,
    *,
    bands: Mapping[str, tuple[float, float]],
    window_sec: float,
    step_sec: float,
    mode: str = "de",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return graph node features with shape (trials, windows, channels, bands)."""
    n_trials, n_channels, n_times = X.shape
    win_samples = max(4, int(round(window_sec * sfreq)))
    step_samples = max(1, int(round(step_sec * sfreq)))
    starts = np.arange(0, n_times - win_samples + 1, step_samples)
    centers = (starts + win_samples / 2.0) / sfreq
    band_items = list(bands.items())
    out = np.empty((n_trials, len(starts), n_channels, len(band_items)), dtype=np.float32)

    for wi, start in enumerate(starts):
        segment = X[:, :, start : start + win_samples]
        freqs, psd = welch(segment, fs=sfreq, nperseg=min(win_samples, 256), axis=-1)
        for bi, (_, (lo, hi)) in enumerate(band_items):
            mask = (freqs >= lo) & (freqs <= hi)
            power = trapezoid(psd[..., mask], freqs[mask], axis=-1)
            power = np.maximum(power, np.finfo(np.float32).eps)
            out[:, wi, :, bi] = np.log(power) if mode.lower() == "de" else power
    return out, centers.astype(np.float32), [name for name, _ in band_items]


def electrode_distance_adjacency(
    ch_names: Sequence[str],
    *,
    montage_name: str = "standard_1005",
    sigma: float | None = None,
    threshold: float = 0.35,
) -> np.ndarray:
    montage = mne.channels.make_standard_montage(montage_name)
    pos = montage.get_positions()["ch_pos"]
    coords = np.array([pos.get(ch, [np.nan, np.nan, np.nan]) for ch in ch_names], dtype=float)
    valid = ~np.isnan(coords).any(axis=1)
    if valid.sum() < 2:
        return np.eye(len(ch_names), dtype=np.float32)
    coords[~valid] = np.nanmean(coords[valid], axis=0)
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    nonzero = dist[dist > 0]
    sigma = float(sigma or np.median(nonzero))
    adj = np.exp(-(dist**2) / (2 * sigma**2))
    adj[dist > threshold] = 0.0
    np.fill_diagonal(adj, 1.0)
    return adj.astype(np.float32)
