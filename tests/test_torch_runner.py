from __future__ import annotations

import numpy as np
from torch import nn

import br_analysis.runners.torch as torch_runner_module
from br_analysis.dataset import EpochDataset
from br_analysis.runners.torch import TorchRunnerConfig, run_subjectwise_torch
from br_analysis.splits import SplitPlan


class TinyModel(nn.Module):
    def __init__(self, shape: tuple[int, ...], n_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(int(np.prod(shape)), n_classes)

    def forward(self, inputs):
        return self.classifier(inputs.flatten(start_dim=1))


def test_torch_runner_returns_multiclass_probabilities(monkeypatch) -> None:
    progress_bars = []

    class ProgressRecorder:
        def __init__(self, iterable=None, *, total=None, **kwargs):
            del kwargs
            self.iterable = iterable
            self.total = total if total is not None else len(iterable)
            self.completed = 0
            progress_bars.append(self)

        def __iter__(self):
            for value in self.iterable:
                self.completed += 1
                yield value

        def set_postfix(self, *args, **kwargs):
            pass

        def set_postfix_str(self, *args, **kwargs):
            pass

        def update(self, amount=1):
            self.completed += amount

        def close(self):
            pass

    monkeypatch.setattr(torch_runner_module, "tqdm", ProgressRecorder)
    rng = np.random.default_rng(9)
    y = np.tile([0, 1, 2], 20)
    X = y[:, None, None] + rng.normal(0, 0.2, size=(len(y), 2, 4))
    dataset = EpochDataset(
        X=X.astype(np.float32),
        y=y,
        subjects=np.repeat(["P001"], len(y)),
        trial_indices=np.arange(len(y)),
        channel_names=["Cz", "Oz"],
        times=np.arange(4, dtype=np.float32),
        sfreq=4.0,
        task_id="multiclass",
        class_ids=("a", "b", "c"),
        class_labels=("A", "B", "C"),
        exclusions=[],
    )
    result = run_subjectwise_torch(
        dataset,
        lambda shape, classes: TinyModel(shape, classes),
        split_plan=SplitPlan(n_splits=2, holdout_fraction=0.2, temporal_blocks=4),
        config=TorchRunnerConfig(
            epochs=2,
            patience=1,
            batch_size=16,
            normalization="channel",
            device="cpu",
        ),
    )
    assert result.probabilities.shape[1] == 3
    np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0, atol=1e-6)
    fold_progress = progress_bars[0]
    assert fold_progress.total == 1
    assert fold_progress.completed == 1
    assert progress_bars[1].total == 2
    assert 1 <= progress_bars[2].total <= 2
    assert set(result.metrics["evaluation"]) == {"final_chronological_holdout"}
    assert np.array_equal(result.trial_indices, np.arange(48, 60))
