from __future__ import annotations

import json
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .artifacts import file_inventory_signature, stable_signature
from .config import ProjectPaths
from .dataset import EpochDataset, SubjectExclusion
from .preprocessing import (
    PreprocessingPreset,
    get_cache_analysis_window,
    get_preprocessing_preset,
)
from .tasks import TaskSpec

if TYPE_CHECKING:
    import mne


PREPROCESSING_CACHE_VERSION = 4
PORTABLE_PREPROCESSING_VERSION = 2


@dataclass
class _PreprocessedEpochs:
    X: np.ndarray
    event_codes: np.ndarray
    subjects: np.ndarray
    trial_indices: np.ndarray
    channel_names: list[str]
    times: np.ndarray
    sfreq: float
    exclusions: list[SubjectExclusion]
    info: Any | None = None
    preprocessing_metadata: dict[str, Any] = field(default_factory=dict)


def read_eeg_epochs(fif_path: Path, *, preload: bool = True) -> mne.Epochs:
    import mne

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"This filename .* does not conform to MNE naming conventions.*",
            category=RuntimeWarning,
        )
        epochs = mne.read_epochs(fif_path, preload=preload, verbose=False)
    if preload:
        epochs.load_data()

    for channel in epochs.info["chs"]:
        if channel["unit"] == 0:
            channel["unit"] = mne.io.constants.FIFF.FIFF_UNIT_V
            channel["unit_mul"] = mne.io.constants.FIFF.FIFF_UNITM_NONE

    epochs.set_channel_types(
        {channel_name: "eeg" for channel_name in epochs.ch_names},
        on_unit_change="ignore",
    )
    epochs = epochs.copy().pick(mne.pick_types(epochs.info, eeg=True))
    event_codes = np.unique(epochs.events[:, 2])
    epochs.event_id = {f"event_{int(code)}": int(code) for code in event_codes}
    return epochs


def select_task_epochs(
    epochs: mne.Epochs,
    task: TaskSpec,
) -> tuple[mne.Epochs | None, np.ndarray | None, np.ndarray | None, str | None]:
    events = epochs.events[:, 2]
    event_to_class = task.event_to_class
    selected_mask = np.isin(events, list(event_to_class))
    if not selected_mask.any():
        return None, None, None, "no task events"

    selected_indices = np.flatnonzero(selected_mask)
    selected_events = events[selected_mask]
    y = np.array([event_to_class[int(event)] for event in selected_events], dtype=np.int16)
    counts = np.bincount(y, minlength=task.n_classes)
    missing = [
        f"{task.class_ids[index]}={int(count)}"
        for index, count in enumerate(counts)
        if count < task.min_trials_per_class
    ]
    if missing:
        return None, None, None, "too few trials: " + ", ".join(missing)

    selected = epochs[selected_indices].copy()
    tmin, tmax = task.analysis_window
    if tmin < selected.tmin or tmax > selected.tmax:
        return (
            None,
            None,
            None,
            f"analysis window {task.analysis_window} outside available range ({selected.tmin}, {selected.tmax})",
        )
    selected.crop(tmin=tmin, tmax=tmax)
    return selected, y, selected_indices.astype(np.int32), None


def load_epoch_dataset(
    paths: ProjectPaths,
    task: TaskSpec,
    preset: str | PreprocessingPreset = "broadband",
    *,
    picks: Sequence[str] | None = None,
    verbose: bool = True,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> EpochDataset:
    preprocessing = (
        get_preprocessing_preset(preset)
        if isinstance(preset, str)
        else preset
    )
    requested_picks = tuple(picks) if picks is not None else preprocessing.picks
    preprocessing = preprocessing.with_picks(None)
    cache_window = get_cache_analysis_window(task.analysis_window)

    x_blocks: list[np.ndarray] = []
    y_blocks: list[np.ndarray] = []
    subject_blocks: list[np.ndarray] = []
    trial_blocks: list[np.ndarray] = []
    exclusions: list[SubjectExclusion] = []
    channel_names: list[str] | None = None
    times: np.ndarray | None = None
    sfreq: float | None = None
    common_info: Any | None = None
    preprocessing_metadata: dict[str, Any] = {}
    event_to_class = task.event_to_class

    for subject_data, exclusion in _iter_preprocessed_subjects(
        paths,
        preprocessing,
        cache_window,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
    ):
        if exclusion is not None:
            exclusions.append(exclusion)
            continue
        assert subject_data is not None
        subject = str(subject_data.subjects[0])

        if channel_names is None:
            channel_names = list(subject_data.channel_names)
            times = subject_data.times
            sfreq = subject_data.sfreq
            common_info = subject_data.info
            preprocessing_metadata = dict(subject_data.preprocessing_metadata)
        missing_channels = [
            channel for channel in channel_names if channel not in subject_data.channel_names
        ]
        if missing_channels:
            exclusions.append(
                SubjectExclusion(subject, "missing required channels", ",".join(missing_channels))
            )
            continue
        if (
            times is None
            or len(times) != len(subject_data.times)
            or not np.allclose(times, subject_data.times, atol=1e-6)
        ):
            exclusions.append(SubjectExclusion(subject, "incompatible time grid"))
            continue
        channel_indices = [
            subject_data.channel_names.index(channel) for channel in channel_names
        ]

        task_mask = np.isin(subject_data.event_codes, list(event_to_class))
        if not task_mask.any():
            exclusions.append(SubjectExclusion(subject, "no task events"))
            continue
        y_subject = np.array(
            [event_to_class[int(event)] for event in subject_data.event_codes[task_mask]],
            dtype=np.int16,
        )
        counts = np.bincount(y_subject, minlength=task.n_classes)
        missing_classes = [
            f"{task.class_ids[index]}={int(count)}"
            for index, count in enumerate(counts)
            if count < task.min_trials_per_class
        ]
        if missing_classes:
            exclusions.append(
                SubjectExclusion(subject, "too few trials", ", ".join(missing_classes))
            )
            continue

        x_blocks.append(subject_data.X[task_mask][:, channel_indices])
        y_blocks.append(y_subject)
        subject_blocks.append(subject_data.subjects[task_mask])
        trial_blocks.append(subject_data.trial_indices[task_mask])

    if not x_blocks or channel_names is None or times is None or sfreq is None:
        details = "; ".join(str(exclusion) for exclusion in exclusions[:5])
        raise RuntimeError(f"No valid subjects could be loaded for {task.id}. {details}")

    dataset = EpochDataset(
        X=np.concatenate(x_blocks),
        y=np.concatenate(y_blocks),
        subjects=np.concatenate(subject_blocks),
        trial_indices=np.concatenate(trial_blocks),
        channel_names=channel_names,
        times=times,
        sfreq=sfreq,
        task_id=task.id,
        class_ids=task.class_ids,
        class_labels=task.class_labels,
        exclusions=exclusions,
        info=common_info,
        preprocessing_metadata=preprocessing_metadata,
    )
    if task.analysis_window != cache_window:
        dataset = _crop_dataset(dataset, task.analysis_window)
    if requested_picks is not None:
        dataset = _pick_dataset_channels(dataset, requested_picks)
    if verbose:
        counts = ", ".join(f"{label}={count}" for label, count in dataset.class_counts.items())
        print(
            f"Selected {len(dataset.y)} trials from {len(dataset.subject_ids)} subjects "
            f"for {task.id}: {counts}"
        )
        if exclusions:
            print(f"Excluded {len(exclusions)} subjects; see dataset.exclusions for details.")
    return dataset


def prepare_epoch_cache(
    paths: ProjectPaths,
    preset: str | PreprocessingPreset,
    analysis_window: tuple[float, float],
    *,
    refresh_cache: bool = False,
    verbose: bool = True,
) -> None:
    preprocessing = (
        get_preprocessing_preset(preset)
        if isinstance(preset, str)
        else preset
    ).with_picks(None)
    files = _require_data_files(paths)
    exclusions: list[SubjectExclusion] = []
    for index, fif_path in enumerate(files, start=1):
        if verbose:
            print(f"  [{index}/{len(files)}] {fif_path.stem}", flush=True)
        _, exclusion = _load_preprocessed_subject(
            paths,
            fif_path,
            preprocessing,
            analysis_window,
            use_cache=True,
            refresh_cache=refresh_cache,
        )
        if exclusion is not None:
            exclusions.append(exclusion)
    if exclusions:
        print(f"Skipped {len(exclusions)} subjects during preprocessing.")


def portable_data_inventory(paths: ProjectPaths) -> list[Path]:
    manifest_path = paths.data_dir / "preprocessed-manifest.json"
    if not manifest_path.is_file():
        return []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format_version") == 1:
            entries = manifest["subjects"]
        else:
            entries = [
                entry
                for variant in manifest["variants"]
                for entry in variant["subjects"]
            ]
        return [
            manifest_path,
            *[
                paths.data_dir / entry[key]
                for entry in entries
                for key in ("data_file", "metadata_file")
            ],
        ]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return [manifest_path]


def data_inventory_signature(paths: ProjectPaths) -> str:
    files = paths.data_files or portable_data_inventory(paths)
    return file_inventory_signature(files)


def _require_data_files(paths: ProjectPaths) -> list[Path]:
    files = paths.data_files
    if not files:
        raise FileNotFoundError(f"No FIF files found in {paths.data_dir}.")
    return files


def _iter_preprocessed_subjects(
    paths: ProjectPaths,
    preprocessing: PreprocessingPreset,
    analysis_window: tuple[float, float],
    *,
    use_cache: bool,
    refresh_cache: bool,
) -> Iterator[tuple[_PreprocessedEpochs | None, SubjectExclusion | None]]:
    manifest_path = paths.data_dir / "preprocessed-manifest.json"
    if manifest_path.is_file():
        yield from _iter_portable_preprocessing(
            paths,
            preprocessing,
            analysis_window,
            manifest_path,
        )
        return
    for fif_path in _require_data_files(paths):
        yield _load_preprocessed_subject(
            paths,
            fif_path,
            preprocessing,
            analysis_window,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
        )


def _iter_portable_preprocessing(
    paths: ProjectPaths,
    preprocessing: PreprocessingPreset,
    analysis_window: tuple[float, float],
    manifest_path: Path,
) -> Iterator[tuple[_PreprocessedEpochs | None, SubjectExclusion | None]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    format_version = manifest.get("format_version")
    if format_version == 1:
        variants = [manifest]
    elif format_version == PORTABLE_PREPROCESSING_VERSION:
        variants = manifest["variants"]
    else:
        raise ValueError("Unsupported portable preprocessing manifest version.")
    matching = [
        variant
        for variant in variants
        if tuple(variant["preprocessing"]["bandpass"]) == preprocessing.bandpass
        and variant["preprocessing"]["resample_sfreq"]
        == preprocessing.resample_sfreq
    ]
    if not matching:
        raise ValueError(
            f"Portable data does not contain preprocessing preset {preprocessing.id!r}."
        )
    variant = matching[0]
    available_window = tuple(variant["analysis_window"])
    if (
        analysis_window[0] < available_window[0]
        or analysis_window[1] > available_window[1]
    ):
        raise ValueError(
            f"Requested window {analysis_window} is outside portable data "
            f"window {available_window}."
        )
    for entry in variant["subjects"]:
        subject = entry["subject"]
        try:
            yield (
                _load_cached_preprocessing(
                    paths.data_dir / entry["data_file"],
                    paths.data_dir / entry["metadata_file"],
                ),
                None,
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            yield None, SubjectExclusion(
                subject,
                "invalid portable preprocessing",
                str(exc),
            )


def _load_preprocessed_subject(
    paths: ProjectPaths,
    fif_path: Path,
    preprocessing: PreprocessingPreset,
    analysis_window: tuple[float, float],
    *,
    use_cache: bool,
    refresh_cache: bool,
) -> tuple[_PreprocessedEpochs | None, SubjectExclusion | None]:
    cache_data, cache_metadata = _preprocessing_cache_paths(
        paths,
        fif_path,
        preprocessing,
        analysis_window,
    )
    if use_cache and not refresh_cache and cache_data.is_file() and cache_metadata.is_file():
        try:
            return _load_cached_preprocessing(cache_data, cache_metadata), None
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass

    subject = fif_path.stem
    try:
        epochs = read_eeg_epochs(fif_path, preload=True)
    except (OSError, RuntimeError, ValueError) as exc:
        return None, SubjectExclusion(subject, "read error", str(exc))

    tmin, tmax = analysis_window
    if tmin < epochs.tmin or tmax > epochs.tmax:
        return None, SubjectExclusion(
            subject,
            "analysis window outside available range",
            f"{analysis_window} vs ({epochs.tmin}, {epochs.tmax})",
        )
    selected = epochs.copy().crop(tmin=tmin, tmax=tmax)
    source_filter = {
        "highpass_hz": float(epochs.info["highpass"]),
        "lowpass_hz": float(epochs.info["lowpass"]),
        "available_tmin_s": float(epochs.tmin),
        "available_tmax_s": float(epochs.tmax),
        "filtering_domain": "epoched_data",
    }
    if preprocessing.bandpass is not None:
        selected._data = np.ascontiguousarray(selected.get_data(copy=False))
        selected.filter(
            l_freq=preprocessing.bandpass[0],
            h_freq=preprocessing.bandpass[1],
            method="iir",
            verbose=False,
        )
    if preprocessing.resample_sfreq is not None:
        selected.resample(preprocessing.resample_sfreq, verbose=False)

    event_codes = selected.events[:, 2].astype(np.int16)
    preprocessed = _PreprocessedEpochs(
        X=selected.get_data(copy=True).astype(np.float32),
        event_codes=event_codes,
        subjects=np.full(len(event_codes), subject, dtype=object),
        trial_indices=np.arange(len(event_codes), dtype=np.int32),
        channel_names=list(selected.ch_names),
        times=np.asarray(selected.times, dtype=np.float32),
        sfreq=float(selected.info["sfreq"]),
        exclusions=[],
        info=selected.info.copy(),
        preprocessing_metadata={
            "source_filter": source_filter,
            "applied_bandpass_hz": preprocessing.bandpass,
            "resample_sfreq_hz": preprocessing.resample_sfreq,
        },
    )
    if use_cache:
        _save_cached_preprocessing(preprocessed, cache_data, cache_metadata)
    return preprocessed, None


def _preprocessing_cache_paths(
    paths: ProjectPaths,
    fif_path: Path,
    preprocessing: PreprocessingPreset,
    analysis_window: tuple[float, float],
) -> tuple[Path, Path]:
    processing = {
        "bandpass": preprocessing.bandpass,
        "resample_sfreq": preprocessing.resample_sfreq,
    }
    cache_fingerprint = stable_signature(
        {
            "version": PREPROCESSING_CACHE_VERSION,
            "analysis_window": analysis_window,
            "preprocessing": processing,
            "data": file_inventory_signature([fif_path]),
        }
    )[:12]
    variant_id = _cache_variant_id(preprocessing)
    window_id = "-".join(_cache_number(value) for value in analysis_window)
    cache_stem = f"{fif_path.stem}__{variant_id}__{window_id}__{cache_fingerprint}"
    cache_directory = paths.artifacts_dir / "_cache" / "datasets"
    cache_data = cache_directory / f"{cache_stem}.npz"
    cache_metadata = cache_directory / f"{cache_stem}.json"
    return cache_data, cache_metadata


def _save_cached_preprocessing(
    preprocessed: _PreprocessedEpochs,
    data_path: Path,
    metadata_path: Path,
) -> None:
    data_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_data = data_path.with_suffix(".tmp")
    with temporary_data.open("wb") as handle:
        np.savez_compressed(
            handle,
            X=preprocessed.X,
            event_codes=preprocessed.event_codes,
            subjects=preprocessed.subjects.astype(str),
            trial_indices=preprocessed.trial_indices,
            times=preprocessed.times,
        )
    temporary_data.replace(data_path)
    metadata_path.write_text(
        json.dumps(
            {
                "channel_names": preprocessed.channel_names,
                "sfreq": preprocessed.sfreq,
                "exclusions": [asdict(exclusion) for exclusion in preprocessed.exclusions],
                "cache_version": PREPROCESSING_CACHE_VERSION,
                "preprocessing_metadata": preprocessed.preprocessing_metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _load_cached_preprocessing(
    data_path: Path,
    metadata_path: Path,
) -> _PreprocessedEpochs:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(data_path, allow_pickle=False) as arrays:
        return _PreprocessedEpochs(
            X=arrays["X"],
            event_codes=arrays["event_codes"],
            subjects=arrays["subjects"],
            trial_indices=arrays["trial_indices"],
            channel_names=list(metadata["channel_names"]),
            times=arrays["times"],
            sfreq=float(metadata["sfreq"]),
            exclusions=[SubjectExclusion(**item) for item in metadata["exclusions"]],
            info=None,
            preprocessing_metadata=dict(
                metadata.get("preprocessing_metadata", {})
            ),
        )


def _pick_dataset_channels(
    dataset: EpochDataset,
    picks: Sequence[str],
) -> EpochDataset:
    missing = [channel for channel in picks if channel not in dataset.channel_names]
    if missing:
        raise ValueError(f"Missing requested channels: {', '.join(missing)}")
    indices = [dataset.channel_names.index(channel) for channel in picks]
    if dataset.info is not None:
        import mne

        info = mne.pick_info(dataset.info.copy(), indices)
    else:
        info = None
    return EpochDataset(
        X=dataset.X[:, indices],
        y=dataset.y,
        subjects=dataset.subjects,
        trial_indices=dataset.trial_indices,
        channel_names=list(picks),
        times=dataset.times,
        sfreq=dataset.sfreq,
        task_id=dataset.task_id,
        class_ids=dataset.class_ids,
        class_labels=dataset.class_labels,
        exclusions=dataset.exclusions,
        info=info,
        preprocessing_metadata=dataset.preprocessing_metadata,
    )


def _crop_dataset(
    dataset: EpochDataset,
    analysis_window: tuple[float, float],
) -> EpochDataset:
    tmin, tmax = analysis_window
    time_mask = (dataset.times >= tmin - 1e-7) & (dataset.times <= tmax + 1e-7)
    if not time_mask.any():
        raise ValueError(f"No samples found inside analysis window {analysis_window}.")
    return EpochDataset(
        X=dataset.X[:, :, time_mask],
        y=dataset.y,
        subjects=dataset.subjects,
        trial_indices=dataset.trial_indices,
        channel_names=dataset.channel_names,
        times=dataset.times[time_mask],
        sfreq=dataset.sfreq,
        task_id=dataset.task_id,
        class_ids=dataset.class_ids,
        class_labels=dataset.class_labels,
        exclusions=dataset.exclusions,
        info=dataset.info,
        preprocessing_metadata=dataset.preprocessing_metadata,
    )


def _cache_variant_id(preprocessing: PreprocessingPreset) -> str:
    if preprocessing.bandpass == (1.0, 40.0) and preprocessing.resample_sfreq == 128.0:
        return "broadband"
    if preprocessing.bandpass == (4.0, 13.0) and preprocessing.resample_sfreq == 128.0:
        return "band_limited"
    return preprocessing.id.replace("-", "_")


def _cache_number(value: float) -> str:
    text = f"{value:.6f}".rstrip("0")
    if text.endswith("."):
        text += "0"
    return text.replace("-", "m").replace(".", "p")
