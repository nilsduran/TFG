from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

import br_analysis.runners.classical as classical_module
from br_analysis.dataset import EpochDataset
from br_analysis.runners import run_subjectwise_classical
from br_analysis.splits import SplitPlan


@dataclass(frozen=True)
class DummyFactory:
    name: str = "test"

    def make(self):
        return LogisticRegression(class_weight="balanced")


def test_subjectwise_runner_aligns_predictions(monkeypatch) -> None:
    progress_bars = []

    class ProgressRecorder:
        def __init__(self, *, total, **kwargs):
            del kwargs
            self.total = total
            self.completed = 0
            progress_bars.append(self)

        def set_postfix_str(self, *args, **kwargs):
            pass

        def update(self, amount=1):
            self.completed += amount

        def close(self):
            pass

    monkeypatch.setattr(classical_module, "tqdm", ProgressRecorder)
    rng = np.random.default_rng(42)
    n_per_subject = 40
    y_subject = np.tile([0, 1], n_per_subject // 2)
    y = np.tile(y_subject, 2)
    signal = y[:, None, None] + rng.normal(0, 0.1, size=(len(y), 2, 4))
    dataset = EpochDataset(
        X=signal.astype(np.float32),
        y=y,
        subjects=np.repeat(["P001", "P002"], n_per_subject),
        trial_indices=np.tile(np.arange(n_per_subject), 2),
        channel_names=["Cz", "Oz"],
        times=np.arange(4, dtype=np.float32),
        sfreq=4.0,
        task_id="test",
        class_ids=("a", "b"),
        class_labels=("A", "B"),
        exclusions=[],
    )
    result = run_subjectwise_classical(
        dataset,
        DummyFactory(),
        feature_transform=lambda X, sfreq: X.reshape(len(X), -1),
        split_plan=SplitPlan(n_splits=2, holdout_fraction=0.2, temporal_blocks=4),
        n_jobs=2,
    )
    assert progress_bars[0].total == 2
    assert progress_bars[0].completed == 2
    assert result.probabilities.shape == (len(result.y_true), 2)
    assert len(result.y_true) == 16
    assert set(result.subjects) == {"P001", "P002"}
    assert set(result.metrics["subject"]) == {"P001", "P002"}
    assert set(result.metrics["evaluation"]) == {"final_chronological_holdout"}
    assert list(dict.fromkeys(result.subjects)) == ["P001", "P002"]
    for subject in ("P001", "P002"):
        assert np.array_equal(
            result.trial_indices[result.subjects == subject],
            np.arange(32, 40),
        )
