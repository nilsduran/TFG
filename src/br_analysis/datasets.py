from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import BandChannelSpec, BinaryTaskConfig, FusedBandSpec, ProjectPaths
from .data import load_epoch_dataset
from .dataset import EpochDataset
from .preprocessing import PreprocessingPreset


@dataclass
class LoadedDataset:
    X: np.ndarray | dict[str, np.ndarray]
    y: np.ndarray
    subjects: np.ndarray
    channel_names: list[str] | dict[str, list[str]]
    times: np.ndarray
    skipped: list[str]
    sfreq: float | None = None
    chs_info: list[dict] | None = None
    info: Any | None = None
    trial_indices: np.ndarray | None = None
    class_ids: tuple[str, ...] = ()

    @property
    def subject_ids(self) -> list[str]:
        return list(dict.fromkeys(self.subjects.tolist()))


def _as_loaded_dataset(dataset: EpochDataset, X: np.ndarray | dict[str, np.ndarray], channel_names) -> LoadedDataset:
    return LoadedDataset(
        X=X,
        y=dataset.y,
        subjects=dataset.subjects,
        channel_names=channel_names,
        times=dataset.times,
        skipped=dataset.skipped,
        sfreq=dataset.sfreq,
        chs_info=[dict(channel) for channel in dataset.info["chs"]] if dataset.info is not None else None,
        info=dataset.info,
        trial_indices=dataset.trial_indices,
        class_ids=dataset.class_ids,
    )


def load_band_channel_dataset(
    paths: ProjectPaths,
    task: BinaryTaskConfig,
    spec: BandChannelSpec,
    *,
    verbose: bool = True,
) -> LoadedDataset:
    task.require_binary()
    picks = [channel for channel, _ in spec.target_band_channels]
    base = load_epoch_dataset(
        paths,
        task,
        PreprocessingPreset("band_features", None, None, tuple(picks)),
        verbose=False,
    )
    blocks: list[np.ndarray] = []
    for channel, band in spec.target_band_channels:
        try:
            channel_index = base.channel_names.index(channel)
            low, high = spec.band_ranges[band]
        except (ValueError, KeyError) as exc:
            raise ValueError(f"Invalid band-channel specification: {channel}-{band}") from exc
        filtered = _filter_array(base.X[:, channel_index : channel_index + 1], base.sfreq, low, high)
        blocks.append(filtered[:, 0])
    loaded = _as_loaded_dataset(base, np.stack(blocks, axis=1), spec.channel_names)
    _print_summary(loaded, verbose)
    return loaded


def load_fused_band_dataset(
    paths: ProjectPaths,
    task: BinaryTaskConfig,
    spec: FusedBandSpec,
    *,
    verbose: bool = True,
) -> LoadedDataset:
    all_channels = list(
        dict.fromkeys(channel for channels in spec.target_bands_clusters.values() for channel in channels)
    )
    base = load_epoch_dataset(
        paths,
        task,
        PreprocessingPreset("fused_bands", None, None, tuple(all_channels)),
        verbose=False,
    )
    x_by_band: dict[str, np.ndarray] = {}
    channels_by_band: dict[str, list[str]] = {}
    for band, channels in spec.target_bands_clusters.items():
        available = [channel for channel in channels if channel in base.channel_names]
        if len(available) < len(channels) * spec.min_cluster_fraction:
            raise ValueError(f"Too few available channels for {band}.")
        indices = [base.channel_names.index(channel) for channel in available]
        low, high = spec.band_ranges[band]
        x_by_band[band] = _filter_array(base.X[:, indices], base.sfreq, low, high)
        channels_by_band[band] = available
    if "theta" in x_by_band and "alpha" in x_by_band:
        x_by_band["fused"] = np.concatenate([x_by_band["theta"], x_by_band["alpha"]], axis=1)
        channels_by_band["fused"] = channels_by_band["theta"] + channels_by_band["alpha"]
    loaded = _as_loaded_dataset(base, x_by_band, channels_by_band)
    _print_summary(loaded, verbose)
    return loaded


def _filter_array(X: np.ndarray, sfreq: float, low: float, high: float) -> np.ndarray:
    import mne

    return mne.filter.filter_data(
        np.ascontiguousarray(X),
        sfreq=sfreq,
        l_freq=low,
        h_freq=high,
        method="iir",
        verbose=False,
    ).astype(np.float32)


def _print_summary(dataset: LoadedDataset, verbose: bool) -> None:
    if verbose:
        counts = np.bincount(dataset.y, minlength=len(dataset.class_ids))
        print(
            f"Loaded {len(dataset.y)} trials from {len(dataset.subject_ids)} subjects; "
            f"class counts={counts.tolist()}"
        )


def chronological_holdout(
    X: np.ndarray,
    y: np.ndarray,
    holdout_frac: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_trials = X.shape[0]
    split_index = int(np.floor((1 - holdout_frac) * n_trials))
    split_index = max(1, min(split_index, n_trials - 1))
    return X[:split_index], X[split_index:], y[:split_index], y[split_index:]
