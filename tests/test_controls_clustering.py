from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import br_analysis.reduction as reduction_module
from br_analysis.dataset import EpochDataset
from br_analysis.reduction import (
    apply_spatial_parcels,
    parcel_band_features,
    parcel_filterbank,
    spatial_parcels,
)
from br_analysis.subject_identification import run_subject_identification


def _dataset() -> EpochDataset:
    rng = np.random.default_rng(7)
    n_subjects = 3
    n_trials = 30
    n_channels = 3
    n_times = 32
    covariance_matrices = [
        np.diag([2.0 if channel == subject else 0.2 for channel in range(n_channels)])
        for subject in range(n_subjects)
    ]
    X = np.stack(
        [
            rng.multivariate_normal(
                np.zeros(n_channels),
                covariance_matrices[subject],
                size=n_times,
            ).T
            for subject in range(n_subjects)
            for _ in range(n_trials)
        ]
    )
    return EpochDataset(
        X=X.astype(np.float32),
        y=np.tile([0, 1], n_subjects * n_trials // 2),
        subjects=np.repeat([f"P{index:03d}" for index in range(n_subjects)], n_trials),
        trial_indices=np.tile(np.arange(n_trials), n_subjects),
        channel_names=["Fz", "Cz", "Oz"],
        times=np.arange(n_times, dtype=np.float32),
        sfreq=float(n_times),
        task_id="test",
        class_ids=("a", "b"),
        class_labels=("A", "B"),
        exclusions=[],
    )


def test_subject_identification_uses_chronological_holdout() -> None:
    result = run_subject_identification(_dataset())
    assert result.chance_level == 1 / 3
    assert result.probabilities.shape == (18, 3)
    assert result.metrics.balanced_accuracy > result.chance_level
    assert result.pipeline_steps[:2] == ("oas_covariance", "riemann_tangent_space")


def test_spatial_parcels_are_complete_connected_and_deterministic(monkeypatch) -> None:
    channel_names = tuple(f"C{index}" for index in range(6))
    adjacency = np.eye(6, dtype=bool)
    for index in range(5):
        adjacency[index, index + 1] = True
        adjacency[index + 1, index] = True
    sparse_adjacency = csr_matrix(adjacency)
    coordinates = np.column_stack(
        [np.arange(6, dtype=float), np.zeros(6), np.zeros(6)]
    )
    monkeypatch.setattr(
        reduction_module,
        "channel_adjacency",
        lambda names: sparse_adjacency,
    )
    monkeypatch.setattr(
        reduction_module,
        "channel_coordinates",
        lambda names: coordinates,
    )

    labels, matrix = spatial_parcels(channel_names, n_parcels=3)
    repeated_labels, repeated_matrix = spatial_parcels(channel_names, n_parcels=3)

    np.testing.assert_array_equal(labels, repeated_labels)
    np.testing.assert_allclose(matrix, repeated_matrix)
    np.testing.assert_allclose(matrix.sum(axis=0), 1.0 / np.bincount(labels)[labels])
    assert set(labels) == {0, 1, 2}
    for parcel in range(3):
        members = np.flatnonzero(labels == parcel)
        n_components, _ = connected_components(
            sparse_adjacency[members][:, members],
            directed=False,
        )
        assert n_components == 1


def test_shared_parcel_filterbank_has_model_specific_output_shapes() -> None:
    rng = np.random.default_rng(4)
    X = rng.normal(size=(5, 6, 64)).astype(np.float32)
    matrix = np.zeros((3, 6), dtype=np.float32)
    matrix[0, :2] = 0.5
    matrix[1, 2:4] = 0.5
    matrix[2, 4:] = 0.5

    parcels = apply_spatial_parcels(X, matrix)
    eegnet_tensor = parcel_filterbank(X, 64.0, matrix)
    svc_features = parcel_band_features(
        X,
        64.0,
        matrix,
        n_time_bins=8,
    )

    assert parcels.shape == (5, 3, 64)
    assert eegnet_tensor.shape == (5, 6, 64)
    assert svc_features.shape == (5, 48)
