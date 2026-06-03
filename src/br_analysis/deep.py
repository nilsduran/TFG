from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import mne
import numpy as np
import pandas as pd

from .config import BinaryTaskConfig, ProjectPaths
from .datasets import LoadedDataset, _read_eeg_epochs, _select_binary_epochs, _subject_ids, _subject_seed, _undersample


@dataclass(frozen=True)
class RawEpochSpec:
    picks: Sequence[str] | None = None
    bandpass: tuple[float, float] | None = None
    resample_sfreq: float | None = None


def load_raw_epoch_dataset(
    paths: ProjectPaths,
    task: BinaryTaskConfig,
    spec: RawEpochSpec,
    *,
    verbose: bool = True,
) -> LoadedDataset:
    """Load binary EEG epochs for deep models with shape (trials, channels, times)."""
    import joblib

    # Construct a cache key/filename based on task and spec to allow fast loading
    # Normalize/clean model-specific suffix from task name to enable shared caching
    base_task_name = task.name
    for base_name in ["c_vs_ip", "im_vs_ip"]:
        if base_name in task.name:
            base_task_name = base_name
            break

    picks_hash = hash(tuple(spec.picks)) if spec.picks else "all"
    bandpass_str = f"{spec.bandpass[0]}_{spec.bandpass[1]}" if spec.bandpass else "none"
    sfreq_str = f"{spec.resample_sfreq}" if spec.resample_sfreq else "native"
    window_str = f"{task.analysis_window[0]}_{task.analysis_window[1]}" if task.analysis_window else "full"
    cache_name = f"dataset_{base_task_name}_subj{task.max_subjects}_bp{bandpass_str}_sf{sfreq_str}_win{window_str}_pk{picks_hash}.joblib"
    
    shared_cache_dir = paths.analysis_root / "cache"
    shared_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = shared_cache_dir / cache_name

    if cache_path.exists():
        if verbose:
            print(f"Loading preprocessed dataset from cache: {cache_path}")
        try:
            return joblib.load(cache_path)
        except Exception as e:
            if verbose:
                print(f"Cache read failed, reloading from source. Error: {e}")

    subject_ids = _subject_ids(task.selected_subjects_csv, task.max_subjects)
    fif_by_subject = {p.stem: p for p in paths.data_files}
    x_blocks: list[np.ndarray] = []
    y_blocks: list[np.ndarray] = []
    subj_blocks: list[str] = []
    skipped: list[str] = []
    common_ch_names: list[str] | None = None
    times: np.ndarray | None = None
    common_info = None
    common_chs_info: list[dict] | None = None
    sfreq: float | None = None

    if verbose:
        print(f"Selected subjects to load: {len(subject_ids)}")

    for subj_name in subject_ids:
        fif_path = fif_by_subject.get(subj_name)
        if fif_path is None:
            skipped.append(f"{subj_name}: .fif file not found")
            continue

        ep = _read_eeg_epochs(fif_path, preload=True)
        if spec.picks is not None:
            available = [ch for ch in spec.picks if ch in ep.ch_names]
            if len(available) < max(2, int(0.5 * len(spec.picks))):
                skipped.append(f"{fif_path.name}: too few required channels")
                continue
            ep = ep.copy().pick(available)

        ep, y_s, reason = _select_binary_epochs(ep, task)
        if ep is None or y_s is None:
            skipped.append(f"{fif_path.name}: {reason}")
            continue
        if task.undersample_classes:
            ep, y_s = _undersample(ep, y_s, seed=_subject_seed(task, subj_name))
        if spec.bandpass is not None:
            ep._data = np.ascontiguousarray(ep._data)
            ep.filter(l_freq=spec.bandpass[0], h_freq=spec.bandpass[1], method="fir", phase="zero-double", verbose=False)
        if spec.resample_sfreq is not None:
            ep.resample(spec.resample_sfreq, verbose=False)

        if common_ch_names is None:
            common_ch_names = list(ep.ch_names)
            common_info = ep.info.copy()
            common_chs_info = [dict(ch) for ch in ep.info["chs"]]
            sfreq = float(ep.info["sfreq"])
        else:
            keep = [ch for ch in common_ch_names if ch in ep.ch_names]
            if len(keep) != len(common_ch_names):
                common_ch_names = keep
                x_blocks = [x[:, : len(common_ch_names), :] for x in x_blocks]
            ep = ep.copy().pick(common_ch_names)

        if times is None:
            times = np.array(ep.times, dtype=np.float32)
        x_blocks.append(ep.get_data().astype(np.float32))
        y_blocks.append(y_s)
        subj_blocks.extend([subj_name] * len(y_s))

    if not x_blocks:
        raise RuntimeError("No valid subjects could be loaded.")

    X = np.concatenate(x_blocks, axis=0)
    dataset = LoadedDataset(
        X=X,
        y=np.concatenate(y_blocks, axis=0),
        subjects=np.array(subj_blocks, dtype=object),
        channel_names=common_ch_names or [],
        times=np.asarray(times, dtype=np.float32),
        skipped=skipped,
        sfreq=sfreq,
        chs_info=common_chs_info,
        info=common_info,
    )
    if verbose:
        print(f"Total samples: {len(dataset.y)} | {task.class_0}={(dataset.y == 0).sum()} | {task.class_1}={(dataset.y == 1).sum()}")
        print(f"Loaded subjects: {len(np.unique(dataset.subjects))} | Channels: {len(dataset.channel_names)}")
        if skipped:
            print("Skipped:")
            for message in skipped:
                print(f"  {message}")

    # Cache preprocessed dataset to disk
    try:
        paths.cache_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(dataset, cache_path)
        if verbose:
            print(f"Saved preprocessed dataset to cache: {cache_path}")
    except Exception as e:
        if verbose:
            print(f"Failed to cache dataset. Error: {e}")

    return dataset


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
    from scipy.signal import welch
    try:
        from scipy.integrate import trapezoid
    except ImportError:
        from scipy.integrate import trapz as trapezoid

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
