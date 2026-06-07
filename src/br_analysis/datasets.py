from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd

from .config import BandChannelSpec, BinaryTaskConfig, FusedBandSpec, ProjectPaths


@dataclass
class LoadedDataset:
    """Arrays and metadata for one binary decoding task."""

    X: np.ndarray | dict[str, np.ndarray]
    y: np.ndarray
    subjects: np.ndarray
    channel_names: list[str] | dict[str, list[str]]
    times: np.ndarray
    skipped: list[str]
    sfreq: float | None = None
    chs_info: list[dict] | None = None
    info: Any | None = None

    @property
    def subject_ids(self) -> list[str]:
        return pd.unique(self.subjects).tolist()


def _subject_ids(csv_path: Path, max_subjects: int | None) -> list[str]:
    df_sel = pd.read_csv(csv_path)
    col = "subject" if "subject" in df_sel.columns else "subjecte"
    ids = df_sel[col].dropna().astype(str).tolist()
    ids = list(dict.fromkeys(ids))
    return ids[:max_subjects] if max_subjects is not None else ids


def _read_eeg_epochs(fif_path: Path, *, preload: bool = True) -> mne.Epochs:
    import shutil
    import tempfile
    import os
    import weakref

    # Copiem a local per evitar OSError 22 de Google Drive Stream a Windows
    temp_dir = Path(tempfile.gettempdir())
    temp_fif_path = temp_dir / f"temp_eeg_{os.getpid()}_{fif_path.name}"
    shutil.copy(fif_path, temp_fif_path)

    # Always open with preload=False initially
    ep = mne.read_epochs(temp_fif_path, preload=False)
    if preload:
        ep.load_data()
        try:
            temp_fif_path.unlink()
        except Exception:
            pass
    else:
        # Clean up automatically when the Epochs object is destroyed
        def _cleanup(p: str):
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except Exception:
                pass

        weakref.finalize(ep, _cleanup, str(temp_fif_path))

    for ch_idx, _ in enumerate(ep.ch_names):
        if ep.info["chs"][ch_idx]["unit"] == 0:
            ep.info["chs"][ch_idx]["unit"] = mne.io.constants.FIFF.FIFF_UNIT_V
            ep.info["chs"][ch_idx]["unit_mul"] = mne.io.constants.FIFF.FIFF_UNITM_NONE
    ep.set_channel_types({ch: "eeg" for ch in ep.ch_names}, on_unit_change="ignore")
    return ep.copy().pick(mne.pick_types(ep.info, eeg=True))


def _select_binary_epochs(
    ep: mne.Epochs,
    task: BinaryTaskConfig,
    *,
    class_labels: tuple[str, str] | None = None,
) -> tuple[mne.Epochs | None, np.ndarray | None, str | None]:
    class_0, class_1 = class_labels or (task.class_0, task.class_1)
    codes_0 = task.event_codes[class_0]
    codes_1 = task.event_codes[class_1]
    events = ep.events[:, 2]
    mask_0 = np.isin(events, codes_0)
    mask_1 = np.isin(events, codes_1)
    if not mask_0.any() or not mask_1.any():
        return None, None, f"missing required codes ({class_0}={int(mask_0.sum())}, {class_1}={int(mask_1.sum())})"

    ep = ep[mask_0 | mask_1].copy().crop(tmin=task.analysis_window[0], tmax=task.analysis_window[1])
    events = ep.events[:, 2]
    y = np.where(np.isin(events, codes_0), 0, 1).astype(np.int8)
    n_0, n_1 = int((y == 0).sum()), int((y == 1).sum())
    if min(n_0, n_1) == 0:
        return None, None, f"empty class ({class_0}={n_0}, {class_1}={n_1})"
    return ep, y, None


def _undersample(ep: mne.Epochs, y: np.ndarray, *, seed: int) -> tuple[mne.Epochs, np.ndarray]:
    n_min = min(int((y == 0).sum()), int((y == 1).sum()))
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]
    rng = np.random.default_rng(seed)
    keep_idx = np.concatenate([rng.choice(idx_0, n_min, replace=False), rng.choice(idx_1, n_min, replace=False)])
    keep_idx.sort()
    return ep[keep_idx], y[keep_idx]


def _subject_seed(task: BinaryTaskConfig, subject: str) -> int:
    return task.random_seed + sum(bytearray(subject.encode("utf-8")))


def load_band_channel_dataset(
    paths: ProjectPaths,
    task: BinaryTaskConfig,
    spec: BandChannelSpec,
    *,
    verbose: bool = True,
) -> LoadedDataset:
    """Load filtered single-channel band features with shape (trials, specs, times)."""
    subject_ids = _subject_ids(task.selected_subjects_csv, task.max_subjects)
    if verbose:
        print(f"Selected subjects to load: {len(subject_ids)}")

    fif_by_subject = {p.stem: p for p in paths.data_files}
    x_blocks: list[np.ndarray] = []
    y_blocks: list[np.ndarray] = []
    subj_blocks: list[str] = []
    skipped: list[str] = []
    times: np.ndarray | None = None

    for subj_name in subject_ids:
        fif_path = fif_by_subject.get(subj_name)
        if fif_path is None:
            skipped.append(f"{subj_name}: .fif file not found")
            continue

        ep = _read_eeg_epochs(fif_path, preload=True)
        ep, y_s, reason = _select_binary_epochs(ep, task)
        if ep is None or y_s is None:
            skipped.append(f"{fif_path.name}: {reason}")
            continue
        if task.undersample_classes:
            ep, y_s = _undersample(ep, y_s, seed=_subject_seed(task, subj_name))

        band_arrays: list[np.ndarray] = []
        missing_specs: list[str] = []
        for ch_name, band in spec.target_band_channels:
            if ch_name not in ep.ch_names:
                missing_specs.append(f"{ch_name}-{band}")
                continue
            if band not in spec.band_ranges:
                raise ValueError(f"Band not supported: {band}. Define it in band_ranges.")

            l_freq, h_freq = spec.band_ranges[band]
            ep_band = ep.copy().pick([ch_name])
            ep_band._data = np.ascontiguousarray(ep_band._data)
            ep_band.filter(l_freq=l_freq, h_freq=h_freq, method="iir", verbose=False)
            band_arrays.append(ep_band.get_data()[:, 0, :].astype(np.float32))

        if missing_specs or len(band_arrays) != len(spec.target_band_channels):
            skipped.append(f"{fif_path.name}: missing required specs ({', '.join(missing_specs)})")
            continue

        if times is None:
            times = np.array(ep.times, dtype=np.float32)
        x_blocks.append(np.stack(band_arrays, axis=1))
        y_blocks.append(y_s)
        subj_blocks.extend([subj_name] * len(y_s))

    if not x_blocks:
        raise RuntimeError("No valid subjects could be loaded.")

    dataset = LoadedDataset(
        X=np.concatenate(x_blocks, axis=0),
        y=np.concatenate(y_blocks, axis=0),
        subjects=np.array(subj_blocks, dtype=object),
        channel_names=spec.channel_names,
        times=np.asarray(times, dtype=np.float32),
        skipped=skipped,
    )
    _print_dataset_summary(dataset, task, verbose=verbose)
    return dataset


def load_fused_band_dataset(
    paths: ProjectPaths,
    task: BinaryTaskConfig,
    spec: FusedBandSpec,
    *,
    verbose: bool = True,
) -> LoadedDataset:
    """Load band-specific channel clusters and a fused concatenation."""
    subject_ids = _subject_ids(task.selected_subjects_csv, task.max_subjects)
    if verbose:
        print(f"Selected subjects to load: {len(subject_ids)}")

    fif_by_subject = {p.stem: p for p in paths.data_files}
    x_blocks: dict[str, list[np.ndarray]] = {band: [] for band in spec.target_bands_clusters}
    y_blocks: list[np.ndarray] = []
    subj_blocks: list[str] = []
    avail_ch_names: dict[str, list[str]] = {band: [] for band in spec.target_bands_clusters}
    skipped: list[str] = []
    times: np.ndarray | None = None

    for subj_name in subject_ids:
        fif_path = fif_by_subject.get(subj_name)
        if fif_path is None:
            skipped.append(f"{subj_name}: .fif file not found")
            continue

        ep = _read_eeg_epochs(fif_path, preload=True)
        ep, y_s, reason = _select_binary_epochs(ep, task)
        if ep is None or y_s is None:
            skipped.append(f"{fif_path.name}: {reason}")
            continue
        if task.undersample_classes:
            ep, y_s = _undersample(ep, y_s, seed=_subject_seed(task, subj_name))
        if times is None:
            times = np.array(ep.times, dtype=np.float32)

        valid_for_subject = True
        subj_band_data: dict[str, np.ndarray] = {}
        for band, target_channels in spec.target_bands_clusters.items():
            avail_chs = [ch for ch in target_channels if ch in ep.ch_names]
            if len(avail_chs) < len(target_channels) * spec.min_cluster_fraction:
                skipped.append(f"{fif_path.name}: too few channels for cluster {band}")
                valid_for_subject = False
                break

            avail_ch_names[band] = avail_chs
            l_freq, h_freq = spec.band_ranges[band]
            ep_band = ep.copy().pick(avail_chs)
            ep_band._data = np.ascontiguousarray(ep_band._data)
            ep_band.filter(l_freq=l_freq, h_freq=h_freq, method="iir", verbose=False)
            subj_band_data[band] = ep_band.get_data().astype(np.float32)

        if not valid_for_subject:
            continue

        for band in spec.target_bands_clusters:
            x_blocks[band].append(subj_band_data[band])
        y_blocks.append(y_s)
        subj_blocks.extend([subj_name] * len(y_s))

    if not y_blocks:
        raise RuntimeError("No valid subjects could be loaded.")

    x_dict = {band: np.concatenate(blocks, axis=0) for band, blocks in x_blocks.items()}
    if "theta" in x_dict and "alpha" in x_dict:
        x_dict["fused"] = np.concatenate([x_dict["theta"], x_dict["alpha"]], axis=1)
        avail_ch_names["fused"] = avail_ch_names["theta"] + avail_ch_names["alpha"]
        if verbose:
            print(
                "Multi-network fusion: "
                f"Theta ({x_dict['theta'].shape[1]}) + Alpha ({x_dict['alpha'].shape[1]}) = {x_dict['fused'].shape[1]} channels"
            )

    dataset = LoadedDataset(
        X=x_dict,
        y=np.concatenate(y_blocks, axis=0),
        subjects=np.array(subj_blocks, dtype=object),
        channel_names=avail_ch_names,
        times=np.asarray(times, dtype=np.float32),
        skipped=skipped,
    )
    _print_dataset_summary(dataset, task, verbose=verbose)
    return dataset


def _print_dataset_summary(dataset: LoadedDataset, task: BinaryTaskConfig, *, verbose: bool) -> None:
    if not verbose:
        return
    print(
        f"Total samples: {len(dataset.y)} | "
        f"{task.class_0}={(dataset.y == 0).sum()} | {task.class_1}={(dataset.y == 1).sum()}"
    )
    print(f"Loaded subjects: {len(np.unique(dataset.subjects))}")
    print(f"Channels: {dataset.channel_names}")


def chronological_holdout(
    X: np.ndarray, y: np.ndarray, holdout_frac: float = 0.20
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split subject data chronologically.

    Parameters
    ----------
    X : np.ndarray
        Shape (n_trials, n_channels, n_times) or (n_trials, n_features).
    y : np.ndarray
        Shape (n_trials,).
    holdout_frac : float, optional
        Fraction of the *latest* trials to keep as a strict test set.
        Default is 0.20 (20 %).

    Returns
    -------
    X_dev, X_test, y_dev, y_test : tuple of np.ndarray
        Development (first 80 %) and test (last 20 %) splits.
    """
    n_trials = X.shape[0]
    split_idx = int(np.floor((1 - holdout_frac) * n_trials))
    split_idx = max(1, min(split_idx, n_trials - 1))
    X_dev, X_test = X[:split_idx], X[split_idx:]
    y_dev, y_test = y[:split_idx], y[split_idx:]
    return X_dev, X_test, y_dev, y_test
