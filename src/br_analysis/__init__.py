"""Reusable analysis helpers for the binocular rivalry EEG notebooks."""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


from .config import (
    BandChannelSpec,
    BinaryTaskConfig,
    FusedBandSpec,
    ProjectPaths,
    detect_project_paths,
    get_task_config,
    get_epochs_grid,
    COMMON_BAND_RANGES,
    TARGET_BANDS_CLUSTERS,
)
from .datasets import LoadedDataset, load_band_channel_dataset, load_fused_band_dataset
from .experiments import ChanceLevelExperiment, ExperimentSettings, TimeWindowGrid, temporal_stratified_splits
from .models import LogRegModelFactory, RiemannLdaModelFactory
from .deep import RawEpochSpec, electrode_distance_adjacency, load_raw_epoch_dataset, temporal_band_features
from .torch_models import inject_lora_into_linears
from .pretrained import BackboneClassifier, LabramConfig, STEEGFormerConfig, load_backbone_weights, prepare_lora_backbone

__all__ = [
    "BandChannelSpec",
    "BinaryTaskConfig",
    "FusedBandSpec",
    "ProjectPaths",
    "detect_project_paths",
    "get_task_config",
    "get_epochs_grid",
    "COMMON_BAND_RANGES",
    "TARGET_BANDS_CLUSTERS",
    "LoadedDataset",
    "load_band_channel_dataset",
    "load_fused_band_dataset",
    "ChanceLevelExperiment",
    "ExperimentSettings",
    "TimeWindowGrid",
    "temporal_stratified_splits",
    "LogRegModelFactory",
    "RiemannLdaModelFactory",
    "RawEpochSpec",
    "load_raw_epoch_dataset",
    "temporal_band_features",
    "electrode_distance_adjacency",
    "inject_lora_into_linears",
    "BackboneClassifier",
    "LabramConfig",
    "STEEGFormerConfig",
    "load_backbone_weights",
    "prepare_lora_backbone",
]
