from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import fftconvolve, windows
from tqdm.auto import tqdm

from .artifacts import file_inventory_signature, stable_signature
from .config import ProjectPaths
from .data import (
    _load_preprocessed_subject,
    _require_data_files,
)
from .dataset import SubjectExclusion
from .preprocessing import (
    POWER_ANALYSIS_WINDOW,
    POWER_BASELINE,
    get_preprocessing_preset,
)
from .tasks import TaskSpec

POWER_CACHE_VERSION = 1
DEFAULT_POWER_BANDS: Mapping[str, tuple[float, ...]] = {
    "theta": (5.0, 6.0, 7.0),
    "alpha": (8.0, 9.0, 10.0, 11.0, 12.0),
}


@dataclass(frozen=True)
class PowerConfig:
    epoch_window: tuple[float, float] = POWER_ANALYSIS_WINDOW
    baseline: tuple[float, float] = POWER_BASELINE
    output_window: tuple[float, float] = (0.0, 1.5)
    window_seconds: float = 0.5
    step_seconds: float = 0.02
    bands: Mapping[str, tuple[float, ...]] = field(
        default_factory=lambda: dict(DEFAULT_POWER_BANDS)
    )

    def __post_init__(self) -> None:
        if not (
            self.epoch_window[0] <= self.baseline[0] < self.baseline[1]
            <= self.epoch_window[1]
        ):
            raise ValueError("The power baseline must be contained in the epoch window.")
        if not (
            self.epoch_window[0] <= self.output_window[0] < self.output_window[1]
            <= self.epoch_window[1]
        ):
            raise ValueError("The power output window must be contained in the epoch window.")


@dataclass
class PowerDataset:
    X: np.ndarray
    y: np.ndarray
    subjects: np.ndarray
    trial_indices: np.ndarray
    band_names: tuple[str, ...]
    channel_names: list[str]
    times: np.ndarray
    task_id: str
    class_ids: tuple[str, ...]
    class_labels: tuple[str, ...]
    exclusions: list[SubjectExclusion]
    info: Any | None = None

    def __post_init__(self) -> None:
        if self.X.ndim != 4:
            raise ValueError("PowerDataset.X must have shape (trials, bands, channels, times).")
        if self.X.shape[0] != len(self.y):
            raise ValueError("Power data and labels must contain the same number of trials.")
        if self.X.shape[1] != len(self.band_names):
            raise ValueError("band_names must align with the band axis.")
        if self.X.shape[2] != len(self.channel_names):
            raise ValueError("channel_names must align with the channel axis.")
        if self.X.shape[3] != len(self.times):
            raise ValueError("times must align with the time axis.")

    @property
    def subject_ids(self) -> list[str]:
        return list(dict.fromkeys(self.subjects.tolist()))

    @property
    def class_counts(self) -> dict[str, int]:
        return {
            class_id: int(np.sum(self.y == class_index))
            for class_index, class_id in enumerate(self.class_ids)
        }

    def band(self, name: str) -> np.ndarray:
        try:
            index = self.band_names.index(name)
        except ValueError as exc:
            raise ValueError(f"Unknown power band {name!r}.") from exc
        return self.X[:, index]


@dataclass
class _SubjectPower:
    X: np.ndarray
    event_codes: np.ndarray
    trial_indices: np.ndarray
    band_names: tuple[str, ...]
    channel_names: list[str]
    times: np.ndarray


def compute_baseline_corrected_power(
    X: np.ndarray,
    times: np.ndarray,
    sfreq: float,
    config: PowerConfig = PowerConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Hanning-window power and return trial-wise dB change from baseline."""
    window_samples = max(3, int(round(config.window_seconds * sfreq)))
    if window_samples % 2 == 0:
        window_samples += 1
    taper = windows.hann(window_samples, sym=True)
    offsets = np.arange(window_samples) - window_samples // 2
    pad = window_samples // 2
    padded = np.pad(X, ((0, 0), (0, 0), (pad, pad)), mode="reflect")
    step = max(1, int(round(config.step_seconds * sfreq)))
    sampled_indices = np.arange(0, len(times), step)
    sampled_times = np.asarray(times[sampled_indices], dtype=np.float32)
    band_blocks: list[np.ndarray] = []

    for frequencies in config.bands.values():
        accumulated = np.zeros(
            (X.shape[0], X.shape[1], len(sampled_indices)),
            dtype=np.float64,
        )
        for frequency in frequencies:
            kernel = taper * np.exp(2j * np.pi * frequency * offsets / sfreq)
            transformed = fftconvolve(
                padded,
                kernel[None, None, ::-1],
                mode="valid",
                axes=-1,
            )
            accumulated += np.square(np.abs(transformed[..., sampled_indices]))
        accumulated /= len(frequencies)
        band_blocks.append(accumulated.astype(np.float32))

    power = np.stack(band_blocks, axis=1)
    baseline_mask = (
        (sampled_times >= config.baseline[0] - 1e-7)
        & (sampled_times <= config.baseline[1] + 1e-7)
    )
    if not baseline_mask.any():
        raise ValueError("No sampled power points fall inside the baseline window.")
    baseline = power[..., baseline_mask].mean(axis=-1, keepdims=True)
    epsilon = np.finfo(np.float32).tiny
    power_db = 10.0 * np.log10(
        np.maximum(power, epsilon) / np.maximum(baseline, epsilon)
    )
    output_mask = (
        (sampled_times >= config.output_window[0] - 1e-7)
        & (sampled_times <= config.output_window[1] + 1e-7)
    )
    return power_db[..., output_mask].astype(np.float32), sampled_times[output_mask]


def prepare_power_cache(
    paths: ProjectPaths,
    *,
    config: PowerConfig = PowerConfig(),
    force: bool = False,
    verbose: bool = True,
) -> None:
    preprocessing = get_preprocessing_preset("broadband")
    files = _require_data_files(paths)
    progress = tqdm(
        files,
        desc="Theta/alpha power",
        unit="subject",
        dynamic_ncols=True,
        disable=not verbose,
    )
    for fif_path in progress:
        data_path, metadata_path = _power_cache_paths(paths, fif_path, config)
        if not force and data_path.is_file() and metadata_path.is_file():
            progress.set_postfix_str(f"{fif_path.stem}: cached", refresh=False)
            continue
        progress.set_postfix_str(f"{fif_path.stem}: power", refresh=True)
        epochs, exclusion = _load_preprocessed_subject(
            paths,
            fif_path,
            preprocessing,
            config.epoch_window,
            use_cache=True,
            refresh_cache=False,
        )
        if exclusion is not None or epochs is None:
            reason = exclusion or SubjectExclusion(fif_path.stem, "unknown preprocessing error")
            tqdm.write(f"Skipped {reason}")
            continue
        power, power_times = compute_baseline_corrected_power(
            epochs.X,
            epochs.times,
            epochs.sfreq,
            config,
        )
        subject_power = _SubjectPower(
            X=power,
            event_codes=epochs.event_codes,
            trial_indices=epochs.trial_indices,
            band_names=tuple(config.bands),
            channel_names=epochs.channel_names,
            times=power_times,
        )
        _save_power_cache(subject_power, data_path, metadata_path, config)
    progress.close()


def load_power_dataset(
    paths: ProjectPaths,
    task: TaskSpec,
    *,
    config: PowerConfig = PowerConfig(),
    prepare_missing: bool = True,
    verbose: bool = True,
) -> PowerDataset:
    x_blocks: list[np.ndarray] = []
    y_blocks: list[np.ndarray] = []
    subject_blocks: list[np.ndarray] = []
    trial_blocks: list[np.ndarray] = []
    exclusions: list[SubjectExclusion] = []
    channel_names: list[str] | None = None
    band_names: tuple[str, ...] | None = None
    times: np.ndarray | None = None

    for fif_path in _require_data_files(paths):
        data_path, metadata_path = _power_cache_paths(paths, fif_path, config)
        if prepare_missing and not (data_path.is_file() and metadata_path.is_file()):
            _prepare_one_power_cache(paths, fif_path, config)
        if not (data_path.is_file() and metadata_path.is_file()):
            exclusions.append(SubjectExclusion(fif_path.stem, "power cache unavailable"))
            continue
        try:
            subject_power = _load_power_cache(data_path, metadata_path)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            exclusions.append(SubjectExclusion(fif_path.stem, "invalid power cache", str(exc)))
            continue
        if channel_names is None:
            channel_names = subject_power.channel_names
            band_names = subject_power.band_names
            times = subject_power.times
        if (
            subject_power.channel_names != channel_names
            or subject_power.band_names != band_names
            or times is None
            or not np.allclose(subject_power.times, times)
        ):
            exclusions.append(SubjectExclusion(fif_path.stem, "incompatible power grid"))
            continue
        task_mask = np.isin(subject_power.event_codes, list(task.event_to_class))
        y_subject = np.asarray(
            [task.event_to_class[int(code)] for code in subject_power.event_codes[task_mask]],
            dtype=np.int16,
        )
        counts = np.bincount(y_subject, minlength=task.n_classes)
        if np.any(counts < task.min_trials_per_class):
            exclusions.append(SubjectExclusion(fif_path.stem, "too few task trials"))
            continue
        x_blocks.append(subject_power.X[task_mask])
        y_blocks.append(y_subject)
        subject_blocks.append(np.full(len(y_subject), fif_path.stem, dtype=object))
        trial_blocks.append(subject_power.trial_indices[task_mask])

    if not x_blocks or channel_names is None or band_names is None or times is None:
        raise RuntimeError(f"No valid power data could be loaded for {task.id}.")
    dataset = PowerDataset(
        X=np.concatenate(x_blocks),
        y=np.concatenate(y_blocks),
        subjects=np.concatenate(subject_blocks),
        trial_indices=np.concatenate(trial_blocks),
        band_names=band_names,
        channel_names=channel_names,
        times=times,
        task_id=task.id,
        class_ids=task.class_ids,
        class_labels=task.class_labels,
        exclusions=exclusions,
    )
    if verbose:
        counts = ", ".join(f"{key}={value}" for key, value in dataset.class_counts.items())
        print(
            f"Loaded power for {len(dataset.y)} trials from "
            f"{len(dataset.subject_ids)} subjects: {counts}"
        )
    return dataset


def _prepare_one_power_cache(
    paths: ProjectPaths,
    fif_path: Path,
    config: PowerConfig,
) -> None:
    preprocessing = get_preprocessing_preset("broadband")
    epochs, exclusion = _load_preprocessed_subject(
        paths,
        fif_path,
        preprocessing,
        config.epoch_window,
        use_cache=True,
        refresh_cache=False,
    )
    if exclusion is not None or epochs is None:
        return
    power, power_times = compute_baseline_corrected_power(
        epochs.X,
        epochs.times,
        epochs.sfreq,
        config,
    )
    subject_power = _SubjectPower(
        X=power,
        event_codes=epochs.event_codes,
        trial_indices=epochs.trial_indices,
        band_names=tuple(config.bands),
        channel_names=epochs.channel_names,
        times=power_times,
    )
    _save_power_cache(
        subject_power,
        *_power_cache_paths(paths, fif_path, config),
        config,
    )


def _power_cache_paths(
    paths: ProjectPaths,
    fif_path: Path,
    config: PowerConfig,
) -> tuple[Path, Path]:
    fingerprint = stable_signature(
        {
            "version": POWER_CACHE_VERSION,
            "config": asdict(config),
            "source": file_inventory_signature([fif_path]),
        }
    )[:12]
    stem = f"{fif_path.stem}__theta-alpha-db__{fingerprint}"
    directory = paths.artifacts_dir / "_cache" / "power"
    return directory / f"{stem}.npz", directory / f"{stem}.json"


def _save_power_cache(
    power: _SubjectPower,
    data_path: Path,
    metadata_path: Path,
    config: PowerConfig,
) -> None:
    data_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = data_path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            X=power.X,
            event_codes=power.event_codes,
            trial_indices=power.trial_indices,
            times=power.times,
        )
    temporary.replace(data_path)
    metadata_path.write_text(
        json.dumps(
            {
                "cache_version": POWER_CACHE_VERSION,
                "band_names": power.band_names,
                "channel_names": power.channel_names,
                "config": asdict(config),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _load_power_cache(data_path: Path, metadata_path: Path) -> _SubjectPower:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(data_path, allow_pickle=False) as arrays:
        return _SubjectPower(
            X=arrays["X"],
            event_codes=arrays["event_codes"],
            trial_indices=arrays["trial_indices"],
            band_names=tuple(metadata["band_names"]),
            channel_names=list(metadata["channel_names"]),
            times=arrays["times"],
        )
