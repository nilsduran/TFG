"""Reusable EEG analysis tools for binocular-rivalry experiments."""

from __future__ import annotations

import importlib

from .config import (
    COMMON_BAND_RANGES,
    TARGET_BANDS_CLUSTERS,
    BandChannelSpec,
    BinaryTaskConfig,
    FusedBandSpec,
    ProjectPaths,
    detect_project_paths,
    get_task_config,
)
from .dataset import EpochDataset, SubjectExclusion
from .preprocessing import (
    PREPROCESSING_PRESETS,
    PreprocessingPreset,
    get_preprocessing_preset,
)
from .registry import MODELS, ModelSpec, get_model_spec
from .tasks import CONDITIONS, TASKS, ConditionSpec, TaskSpec, get_task

_LAZY_EXPORTS = {
    "LoadedDataset": (".datasets", "LoadedDataset"),
    "load_band_channel_dataset": (".datasets", "load_band_channel_dataset"),
    "load_epoch_dataset": (".data", "load_epoch_dataset"),
    "load_fused_band_dataset": (".datasets", "load_fused_band_dataset"),
    "RawEpochSpec": (".deep", "RawEpochSpec"),
    "load_raw_epoch_dataset": (".deep", "load_raw_epoch_dataset"),
    "temporal_band_features": (".deep", "temporal_band_features"),
    "electrode_distance_adjacency": (".deep", "electrode_distance_adjacency"),
    "LogRegModelFactory": (".models", "LogRegModelFactory"),
    "RiemannLdaModelFactory": (".models", "RiemannLdaModelFactory"),
    "SVCModelFactory": (".models", "SVCModelFactory"),
    "EEGNet": (".torch_models", "EEGNet"),
    "STGNNChebConv": (".torch_models", "STGNNChebConv"),
}


def __getattr__(name: str):
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


__all__ = [
    "BandChannelSpec",
    "BinaryTaskConfig",
    "COMMON_BAND_RANGES",
    "CONDITIONS",
    "ConditionSpec",
    "EpochDataset",
    "FusedBandSpec",
    "MODELS",
    "ModelSpec",
    "PREPROCESSING_PRESETS",
    "PreprocessingPreset",
    "ProjectPaths",
    "SubjectExclusion",
    "TARGET_BANDS_CLUSTERS",
    "TASKS",
    "TaskSpec",
    "detect_project_paths",
    "get_model_spec",
    "get_preprocessing_preset",
    "get_task",
    "get_task_config",
    *_LAZY_EXPORTS,
]
