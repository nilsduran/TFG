"""Reusable analysis helpers for the binocular rivalry EEG notebooks."""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")


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
from .explainability import (
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_ORDER,
    OOF_CACHE_VERSION,
    build_context,
    build_evidence_table,
    convergence_table,
    evidence_pyramid,
    load_oof_predictions,
    plot_convergence,
    plot_evidence_overview,
    save_tables,
    synthesis_text,
)

_LAZY_EXPORTS = {
    "LoadedDataset": (".datasets", "LoadedDataset"),
    "load_band_channel_dataset": (".datasets", "load_band_channel_dataset"),
    "load_fused_band_dataset": (".datasets", "load_fused_band_dataset"),
    "ChanceLevelExperiment": (".experiments", "ChanceLevelExperiment"),
    "ExperimentSettings": (".experiments", "ExperimentSettings"),
    "TimeWindowGrid": (".experiments", "TimeWindowGrid"),
    "chronological_group_labels": (".experiments", "chronological_group_labels"),
    "stratified_group_splits": (".experiments", "stratified_group_splits"),
    "LogRegModelFactory": (".models", "LogRegModelFactory"),
    "RiemannLdaModelFactory": (".models", "RiemannLdaModelFactory"),
    "RawEpochSpec": (".deep", "RawEpochSpec"),
    "load_raw_epoch_dataset": (".deep", "load_raw_epoch_dataset"),
    "temporal_band_features": (".deep", "temporal_band_features"),
    "electrode_distance_adjacency": (".deep", "electrode_distance_adjacency"),
    "inject_lora_into_linears": (".torch_models", "inject_lora_into_linears"),
    "BackboneClassifier": (".pretrained", "BackboneClassifier"),
    "LabramConfig": (".pretrained", "LabramConfig"),
    "STEEGFormerConfig": (".pretrained", "STEEGFormerConfig"),
    "load_backbone_weights": (".pretrained", "load_backbone_weights"),
    "prepare_lora_backbone": (".pretrained", "prepare_lora_backbone"),
    "mae_vit_small_patch16": (".models_mae_eeg", "mae_vit_small_patch16"),
}


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module_name, attr_name = _LAZY_EXPORTS[name]
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


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
    "chronological_group_labels",
    "stratified_group_splits",
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
    "mae_vit_small_patch16",
    "MODEL_COLORS",
    "MODEL_LABELS",
    "MODEL_ORDER",
    "OOF_CACHE_VERSION",
    "build_context",
    "build_evidence_table",
    "convergence_table",
    "evidence_pyramid",
    "load_oof_predictions",
    "plot_convergence",
    "plot_evidence_overview",
    "save_tables",
    "synthesis_text",
]
