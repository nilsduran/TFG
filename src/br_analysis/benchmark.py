from __future__ import annotations

from dataclasses import asdict
from functools import partial
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
from .explanations import save_explanation
from .models import LogRegModelFactory, RiemannLdaModelFactory, SVCModelFactory
from .reduction import parcel_band_features, spatial_parcels
from .registry import get_model_spec
from .representations import band_summary_features, mean_envelope_feature
from .runners.classical import run_subjectwise_classical
from .splits import EVALUATION_PROTOCOL, SplitPlan
from .tasks import get_task


def _band_summary(X: np.ndarray, sfreq: float) -> np.ndarray:
    features, _ = band_summary_features(X, sfreq)
    return features


def _identity(X: np.ndarray, sfreq: float) -> np.ndarray:
    del sfreq
    return X


def _classical_factory(
    model_id: str,
    n_classes: int,
    random_seed: int,
):
    if model_id.startswith("lr_"):
        return LogRegModelFactory(random_seed=random_seed)
    if model_id == "riemann":
        return RiemannLdaModelFactory(n_classes=n_classes)
    if model_id.startswith("svc_"):
        return SVCModelFactory(random_seed=random_seed)
    raise ValueError(f"{model_id!r} is not a classical model.")


def run_classical_benchmark(
    *,
    task_id: str,
    model_id: str,
    paths: ProjectPaths | None = None,
    split_plan: SplitPlan = SplitPlan(),
    explain: bool = False,
    force: bool = False,
    n_jobs: int = 1,
) -> Path:
    model_spec = get_model_spec(model_id)
    if model_spec.family != "sklearn":
        raise ValueError(f"Model {model_id!r} requires the torch runner.")
    paths = paths or ProjectPaths.from_env("benchmark")
    task = get_task(task_id, analysis_window=model_spec.analysis_window)
    picks = model_spec.channels
    directory = artifact_directory(paths.artifacts_dir, task.id, model_id)
    run_signature = stable_signature(
        {
            "artifact_version": ARTIFACT_VERSION,
            "task": asdict(task),
            "model": asdict(model_spec),
            "channels": picks,
            "split_plan": asdict(split_plan),
            "data": data_inventory_signature(paths),
            "implementation": implementation_signature(
                (
                    "benchmark.py",
                    "data.py",
                    "dataset.py",
                    "explanations.py",
                    "metrics.py",
                    "models.py",
                    "preprocessing.py",
                    "reduction.py",
                    "registry.py",
                    "representations.py",
                    "runners/classical.py",
                    "splits.py",
                    "tasks.py",
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

    if model_spec.representation == "theta_envelope":
        transform = partial(mean_envelope_feature, band=(5.0, 7.0))
    elif model_spec.representation == "alpha_envelope":
        transform = partial(mean_envelope_feature, band=(8.0, 12.0))
    elif model_spec.representation == "band_summary":
        transform = _band_summary
    elif model_spec.representation == "parcel_band_features":
        parcel_labels, parcel_matrix = spatial_parcels(
            dataset.channel_names,
            n_parcels=12,
        )
        transform = partial(
            parcel_band_features,
            parcel_matrix=parcel_matrix,
            n_time_bins=24,
        )
    elif model_spec.representation == "covariance":
        transform = _identity
    else:
        raise ValueError(f"Unsupported classical representation: {model_spec.representation}")

    factory = _classical_factory(model_id, task.n_classes, split_plan.random_seed)
    result = run_subjectwise_classical(
        dataset,
        factory,
        feature_transform=transform,
        split_plan=split_plan,
        explain=explain,
        progress_label=f"{task.id}/{model_id}",
        n_jobs=n_jobs,
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
            "reduction": (
                {
                    "n_spatial_parcels": 12,
                    "n_time_bins": 24,
                    "bands": ["theta", "alpha"],
                    "channel_parcels": {
                        channel: int(parcel)
                        for channel, parcel in zip(
                            dataset.channel_names,
                            parcel_labels,
                            strict=True,
                        )
                    },
                }
                if model_id == "svc_clustering"
                else None
            ),
            "excluded_subjects": [
                *dataset.skipped,
                *result.skipped_subjects,
            ],
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "selection_metric": "f1_macro",
            "run_signature": run_signature,
        },
    )
    if result.explanation is not None:
        save_explanation(result.explanation, directory / "shap_values.npz")
    return directory
