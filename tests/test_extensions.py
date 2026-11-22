from pathlib import Path

import numpy as np

import br_analysis.cli as cli_module
import br_analysis.inter_subject as inter_subject_module
import br_analysis.temporal_onset as temporal_onset_module
from br_analysis.config import ProjectPaths
from br_analysis.dataset import EpochDataset
from br_analysis.inter_subject import (
    InterSubjectConfig,
    _fdr_by_variant,
    _logeuclidean_probabilities,
)
from br_analysis.splits import SplitPlan
from br_analysis.temporal_onset import (
    TemporalOnsetConfig,
    _channel_groups,
    _filter_sensitivity,
    _oas_lda_predict,
    _right_censored_median,
)
from br_analysis.temporal_onset import (
    _run_task as run_temporal_task,
)
from br_analysis.temporal_statistics import (
    balanced_accuracy_curve,
    benjamini_yekutieli,
    first_sustained,
    holm_adjust,
)


def test_temporal_statistics_and_channel_controls() -> None:
    assert first_sustained(
        np.array([False, True, True, False]),
        2,
    ) == 1
    assert first_sustained(np.array([False, True, False]), 2) is None
    curve = balanced_accuracy_curve(
        np.array([0, 0, 1, 1]),
        np.array([[0, 0], [0, 1], [1, 1], [1, 0]]),
        n_classes=2,
    )
    np.testing.assert_allclose(curve, [1.0, 0.5])
    np.testing.assert_allclose(
        holm_adjust([0.01, 0.04, 0.03]),
        [0.03, 0.06, 0.06],
    )
    assert np.all(
        benjamini_yekutieli(np.array([0.01, 0.2]))
        >= np.array([0.01, 0.2])
    )
    primary, eog, frontal = _channel_groups(
        ["Fp1", "Fz", "Cz", "Oz", "VEOG"]
    )
    assert primary == [0, 1, 2, 3]
    assert eog == [4]
    assert frontal == [0, 1]
    assert _right_censored_median(
        np.array([0.1, 0.2, 0.3, np.nan, np.nan]),
        np.array([True, True, True, False, False]),
    ) == 0.3
    assert _right_censored_median(
        np.array([0.1, 0.2, np.nan, np.nan, np.nan]),
        np.array([True, True, False, False, False]),
    ) is None
    sensitivity = _filter_sensitivity(
        np.arange(-0.74, 1.501, 0.004),
        250.0,
        (0.1, 30.0),
    )
    assert sensitivity["pre_response_1pct_ms"] >= 0


def test_batched_oas_lda_matches_sklearn_pointwise_pipeline() -> None:
    from sklearn.covariance import OAS
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(31)
    for n_trials, n_channels in ((36, 5), (12, 20)):
        X_train = rng.normal(size=(n_trials, n_channels, 4))
        y_train = np.tile([0, 1], n_trials // 2)
        X_train[y_train == 1, 0] += 0.8
        X_test = rng.normal(size=(10, n_channels, 4))
        actual = _oas_lda_predict(
            X_train,
            y_train,
            X_test,
            n_classes=2,
        )
        expected = np.empty_like(actual)
        for time_index in range(X_train.shape[-1]):
            model = make_pipeline(
                StandardScaler(),
                LinearDiscriminantAnalysis(
                    solver="lsqr",
                    covariance_estimator=OAS(),
                    priors=np.array([0.5, 0.5]),
                ),
            )
            model.fit(X_train[:, :, time_index], y_train)
            expected[:, time_index] = model.predict(X_test[:, :, time_index])
        np.testing.assert_array_equal(actual, expected)


def test_logeuclidean_mdm_probabilities_are_class_aligned() -> None:
    source_logs = np.array(
        [
            np.eye(2) * -1.0,
            np.eye(2) * -0.8,
            np.eye(2) * 0.8,
            np.eye(2) * 1.0,
        ]
    )
    source_y = np.array([0, 0, 1, 1])
    target_logs = np.array([np.eye(2) * -0.9, np.eye(2) * 0.9])

    probabilities = _logeuclidean_probabilities(
        source_logs,
        source_y,
        target_logs,
        n_classes=2,
    )

    assert probabilities.shape == (2, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.array_equal(probabilities.argmax(axis=1), np.array([0, 1]))


def test_fdr_is_computed_separately_for_each_variant() -> None:
    import pandas as pd

    metrics = pd.DataFrame(
        {
            "variant": ["a", "a", "b", "b"],
            "p_value": [0.01, 0.2, 0.04, 0.8],
        }
    )

    adjusted = _fdr_by_variant(metrics)

    assert np.allclose(adjusted, [0.02, 0.2, 0.08, 0.8])


def test_im_vs_ip_bypasses_classical_evidence_gate(
    tmp_path,
    monkeypatch,
) -> None:
    calls = []

    def fake_run(**kwargs):
        calls.append((kwargs["task_id"], kwargs["model_id"]))
        return Path("artifacts") / kwargs["task_id"] / kwargs["model_id"]

    monkeypatch.setattr(cli_module, "_run_combination", fake_run)
    monkeypatch.setenv("BR_PROJECT_ROOT", str(tmp_path))

    assert cli_module.main(
        ["run", "--task", "im_vs_ip", "--model", "eegnet_agnostic"]
    ) == 0
    assert calls == [("im_vs_ip", "eegnet_agnostic")]


def test_inter_subject_analysis_writes_both_transfer_variants(
    tmp_path,
    monkeypatch,
) -> None:
    rng = np.random.default_rng(7)
    subjects = np.repeat(["s01", "s02", "s03", "s04"], 20)
    y = np.tile(np.arange(2).repeat(2), 20)
    X = rng.normal(size=(80, 3, 16))
    X[y == 1, 0] *= 1.8
    dataset = EpochDataset(
        X=X,
        y=y,
        subjects=subjects,
        trial_indices=np.tile(np.arange(20), 4),
        channel_names=["C1", "C2", "C3"],
        times=np.arange(16) / 100,
        sfreq=100.0,
        task_id="c_vs_ip",
        class_ids=("c", "ip"),
        class_labels=("C", "IP"),
        exclusions=[],
    )
    monkeypatch.setattr(
        inter_subject_module,
        "load_epoch_dataset",
        lambda *_args, **_kwargs: dataset,
    )
    monkeypatch.setattr(
        inter_subject_module,
        "data_inventory_signature",
        lambda _paths: "synthetic-data",
    )
    paths = ProjectPaths.from_env(
        "inter_subject",
        project_dir=tmp_path,
    )

    summary = inter_subject_module._run_task(
        paths,
        "c_vs_ip",
        config=InterSubjectConfig(permutations=20),
        force=True,
    )

    metrics_path = (
        tmp_path / "artifacts" / "inter_subject" / "c_vs_ip" / "metrics.csv"
    )
    metrics = __import__("pandas").read_csv(metrics_path)
    assert summary["n_subjects"] == 4
    assert set(metrics["variant"]) == {
        "pooled_logeuclid_mdm",
        "ea_logeuclid_mdm",
    }
    assert len(metrics) == 8


def test_temporal_onset_recovers_injected_effect_without_opening_holdout_early(
    tmp_path,
    monkeypatch,
) -> None:
    rng = np.random.default_rng(19)
    times = np.arange(-0.4, 0.22, 0.02)
    subject_ids = ["s01", "s02", "s03", "s04", "s05", "s06"]
    subjects = np.repeat(subject_ids, 40)
    y = np.tile(np.arange(2).repeat(2), 60)
    X = rng.normal(scale=0.4, size=(len(y), 4, len(times)))
    effect = times >= 0.04
    X[np.ix_(np.flatnonzero(y == 1), [0], np.flatnonzero(effect))] += 3.0
    dataset = EpochDataset(
        X=X,
        y=y,
        subjects=subjects,
        trial_indices=np.tile(np.arange(40), len(subject_ids)),
        channel_names=["Fz", "Cz", "Pz", "Oz"],
        times=times,
        sfreq=50.0,
        task_id="c_vs_ip",
        class_ids=("c", "ip"),
        class_labels=("C", "IP"),
        exclusions=[],
        preprocessing_metadata={"source_filter": {"highpass_hz": 0.1}},
    )
    monkeypatch.setattr(
        temporal_onset_module,
        "load_epoch_dataset",
        lambda *_args, **_kwargs: dataset,
    )
    monkeypatch.setattr(
        temporal_onset_module,
        "data_inventory_signature",
        lambda _paths: "synthetic-data",
    )
    paths = ProjectPaths.from_env(
        "temporal_onset",
        project_dir=tmp_path,
    )

    summary = run_temporal_task(
        paths,
        "c_vs_ip",
        config=TemporalOnsetConfig(
            permutations=99,
            bootstrap_samples=40,
            bootstrap_trial_replicates=20,
            pseudo_iterations=1,
            pseudo_folds=3,
            minimum_duration_seconds=0.04,
            baseline=(-0.4, -0.2),
            search_window=(-0.1, 0.2),
        ),
        split_plan=SplitPlan(n_splits=2, temporal_blocks=8),
        n_jobs=1,
        family_id="synthetic-family",
        force=True,
    )

    assert summary["discovery_onset_seconds"] is not None
    assert summary["discovery_onset_seconds"] >= 0.0
    assert summary["confirmation_balanced_accuracy"] > 0.5
    assert (
        tmp_path
        / "artifacts"
        / "temporal_onset"
        / "c_vs_ip"
        / "development_predictions.npz"
    ).is_file()


def test_temporal_onset_does_not_open_holdout_without_discovery(
    tmp_path,
    monkeypatch,
) -> None:
    times = np.arange(-0.4, 0.22, 0.02)
    subject_ids = ["s01", "s02", "s03", "s04"]
    subjects = np.repeat(subject_ids, 40)
    y = np.tile(np.arange(2).repeat(2), 40)
    X = np.zeros((len(y), 3, len(times)))
    for subject_index in range(len(subject_ids)):
        start = subject_index * 40 + 32
        stop = (subject_index + 1) * 40
        holdout_class_one = np.arange(start, stop)[y[start:stop] == 1]
        X[np.ix_(holdout_class_one, [0], np.flatnonzero(times >= 0.04))] = 5
    dataset = EpochDataset(
        X=X,
        y=y,
        subjects=subjects,
        trial_indices=np.tile(np.arange(40), len(subject_ids)),
        channel_names=["Fz", "Cz", "Oz"],
        times=times,
        sfreq=50.0,
        task_id="c_vs_ip",
        class_ids=("c", "ip"),
        class_labels=("C", "IP"),
        exclusions=[],
    )
    monkeypatch.setattr(
        temporal_onset_module,
        "load_epoch_dataset",
        lambda *_args, **_kwargs: dataset,
    )
    monkeypatch.setattr(
        temporal_onset_module,
        "data_inventory_signature",
        lambda _paths: "holdout-only-signal",
    )

    def fail_if_holdout_is_opened(*_args, **_kwargs):
        raise AssertionError("The final holdout was opened without discovery.")

    monkeypatch.setattr(
        temporal_onset_module,
        "_interval_scores",
        fail_if_holdout_is_opened,
    )
    paths = ProjectPaths.from_env(
        "temporal_onset",
        project_dir=tmp_path,
    )
    summary = run_temporal_task(
        paths,
        "c_vs_ip",
        config=TemporalOnsetConfig(
            permutations=19,
            bootstrap_samples=10,
            bootstrap_trial_replicates=5,
            pseudo_iterations=0,
            minimum_duration_seconds=0.04,
            baseline=(-0.4, -0.2),
            search_window=(-0.1, 0.2),
        ),
        split_plan=SplitPlan(n_splits=2, temporal_blocks=8),
        n_jobs=1,
        family_id="null-family",
        force=True,
    )

    assert summary["discovery_onset_seconds"] is None
    assert summary["confirmation_p"] is None
