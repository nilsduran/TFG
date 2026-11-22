from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .artifacts import (
    ARTIFACT_VERSION,
    RunMetadata,
    artifact_directory,
    implementation_signature,
    reusable_run,
    save_run_artifacts,
    stable_signature,
)
from .config import ProjectPaths
from .data import data_inventory_signature, load_epoch_dataset
from .deep import electrode_distance_adjacency, temporal_band_features
from .explanations import save_explanation
from .pretrained import (
    LABRAM_REPO,
    LABRAM_REVISION,
    ST_EEGFORMER_REPO,
    ST_EEGFORMER_REVISION,
    load_labram,
    load_st_eegformer,
    standard_channel_info,
    trim_to_patch_size,
)
from .reduction import parcel_filterbank, spatial_parcels
from .registry import get_model_spec
from .runners.torch import TorchRunnerConfig, run_subjectwise_torch
from .splits import EVALUATION_PROTOCOL, SplitPlan
from .tasks import get_task
from .torch_models import EEGNet, STGNNChebConv


def _identity(X: np.ndarray, sfreq: float) -> np.ndarray:
    del sfreq
    return X


def run_torch_benchmark(
    *,
    task_id: str,
    model_id: str,
    paths: ProjectPaths | None = None,
    checkpoint: Path | None = None,
    split_plan: SplitPlan = SplitPlan(),
    runner_config: TorchRunnerConfig = TorchRunnerConfig(),
    explain: bool = False,
    force: bool = False,
) -> Path:
    model_spec = get_model_spec(model_id)
    if model_spec.family != "torch":
        raise ValueError(f"Model {model_id!r} requires the classical runner.")

    paths = paths or ProjectPaths.from_env("benchmark")
    task = get_task(task_id, analysis_window=model_spec.analysis_window)
    picks = model_spec.channels
    checkpoint_inventory = None
    if checkpoint is not None:
        checkpoint = checkpoint.expanduser().resolve()
        config_path = checkpoint / "config.json"
        if not checkpoint.is_dir() or not config_path.is_file():
            raise ValueError(
                "--checkpoint must point to a complete Hugging Face model "
                "directory containing config.json."
            )
        checkpoint_inventory = {
            "path": str(checkpoint),
            "config_size": config_path.stat().st_size,
            "config_modified_ns": config_path.stat().st_mtime_ns,
        }
    elif model_id == "st_eegformer":
        checkpoint_inventory = {
            "repository": ST_EEGFORMER_REPO,
            "revision": ST_EEGFORMER_REVISION,
        }
    elif model_id == "labram":
        checkpoint_inventory = {
            "repository": LABRAM_REPO,
            "revision": LABRAM_REVISION,
        }
    directory = artifact_directory(paths.artifacts_dir, task.id, model_id)
    run_signature = stable_signature(
        {
            "artifact_version": ARTIFACT_VERSION,
            "task": asdict(task),
            "model": asdict(model_spec),
            "channels": picks,
            "checkpoint": checkpoint_inventory,
            "split_plan": asdict(split_plan),
            "runner_config": asdict(runner_config),
            "data": data_inventory_signature(paths),
            "implementation": implementation_signature(
                (
                    "data.py",
                    "dataset.py",
                    "deep.py",
                    "deep_benchmark.py",
                    "explanations.py",
                    "metrics.py",
                    "preprocessing.py",
                    "pretrained.py",
                    "reduction.py",
                    "registry.py",
                    "runners/torch.py",
                    "splits.py",
                    "tasks.py",
                    "torch_models.py",
                )
            ),
        }
    )
    if not force and reusable_run(
        directory,
        run_signature=run_signature,
        require_explanation=explain,
    ):
        print(f"Reusing completed run: {directory}")
        return directory

    dataset = load_epoch_dataset(
        paths,
        task,
        model_spec.preprocessing,
        picks=picks,
    )

    reduction_metadata = None
    if model_id.startswith("eegnet_"):
        if model_id == "eegnet_clustering":
            parcel_labels, parcel_matrix = spatial_parcels(
                dataset.channel_names,
                n_parcels=12,
            )

            def input_transform(X: np.ndarray, sfreq: float) -> np.ndarray:
                return parcel_filterbank(X, sfreq, parcel_matrix)

            reduction_metadata = {
                "n_spatial_parcels": 12,
                "bands": ["theta", "alpha"],
                "output_channels": 24,
                "channel_parcels": {
                    channel: int(parcel)
                    for channel, parcel in zip(
                        dataset.channel_names,
                        parcel_labels,
                        strict=True,
                    )
                },
            }
        else:
            input_transform = _identity

        def make_model(shape: tuple[int, ...], classes: int):
            return EEGNet(shape[0], shape[1], n_classes=classes)

    elif model_id == "st_gnn":
        adjacency = electrode_distance_adjacency(dataset.channel_names)

        def input_transform(X: np.ndarray, sfreq: float) -> np.ndarray:
            features, _, _ = temporal_band_features(
                X,
                sfreq,
                bands={"theta": (5.0, 7.0), "alpha": (8.0, 12.0)},
                window_sec=0.25,
                step_sec=0.05,
            )
            return features

        def make_model(shape: tuple[int, ...], classes: int):
            _, n_nodes, n_features = shape
            return STGNNChebConv(
                n_nodes=n_nodes,
                in_features=n_features,
                adjacency=adjacency,
                n_classes=classes,
            )

    elif model_id == "st_eegformer":
        chs_info = standard_channel_info(
            dataset.channel_names,
            sfreq=dataset.sfreq,
        )
        template = load_st_eegformer(
            n_chans=len(dataset.channel_names),
            n_times=dataset.X.shape[-1],
            n_outputs=len(dataset.class_ids),
            chs_info=chs_info,
            local_directory=checkpoint,
        )

        def make_model(shape: tuple[int, ...], classes: int):
            if shape != tuple(dataset.X.shape[1:]) or classes != len(dataset.class_ids):
                raise ValueError("STEEGFormer input shape changed after model loading.")
            return copy.deepcopy(template)

        input_transform = _identity
    elif model_id == "labram":
        chs_info = standard_channel_info(
            dataset.channel_names,
            sfreq=dataset.sfreq,
        )
        n_times = trim_to_patch_size(dataset.X[:1], 200).shape[-1]
        template = load_labram(
            n_times=n_times,
            n_outputs=len(dataset.class_ids),
            chs_info=chs_info,
            local_directory=checkpoint,
        )

        def make_model(shape: tuple[int, ...], classes: int):
            if shape[-1] != n_times or classes != len(dataset.class_ids):
                raise ValueError("LaBraM input shape changed after model loading.")
            return copy.deepcopy(template)

        def input_transform(X: np.ndarray, sfreq: float) -> np.ndarray:
            del sfreq
            return trim_to_patch_size(X, 200)
    else:
        raise ValueError(f"Unsupported torch model: {model_id}")

    result = run_subjectwise_torch(
        dataset,
        make_model,
        input_transform=input_transform,
        split_plan=split_plan,
        config=runner_config,
        explain=explain,
        progress_label=f"{task.id}/{model_id}",
    )
    metadata = RunMetadata.create(
        task_id=task.id,
        model_id=model_id,
        preprocessing_id=model_spec.preprocessing,
        class_ids=task.class_ids,
        random_seed=split_plan.random_seed,
    )
    save_run_artifacts(
        directory,
        metadata=metadata,
        predictions={
            "y_true": result.y_true,
            "probabilities": result.probabilities,
            "subjects": result.subjects,
            "trial_indices": result.trial_indices,
            "folds": result.folds,
        },
        metrics=result.metrics,
        extra_metadata={
            "channels": dataset.channel_names,
            "representation": model_spec.representation,
            "checkpoint": checkpoint_inventory,
            "reduction": reduction_metadata,
            "excluded_subjects": [
                *dataset.skipped,
                *result.skipped_subjects,
            ],
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "selection_metric": runner_config.selection_metric,
            "run_signature": run_signature,
        },
    )
    if result.explanation is not None:
        save_explanation(result.explanation, directory / "shap_values.npz")
    return directory
