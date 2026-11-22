import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from br_analysis.dataset import EpochDataset
from br_analysis.registry import get_model_spec
from br_analysis.scientific_reports import (
    aggregate_metrics,
    cohort_table,
    configuration_table,
    confusion_table,
    load_saved_run,
    plot_confusion,
    plot_roc_pr_comparison,
    prediction_diagnostics,
)
from br_analysis.tasks import get_task


def _dataset() -> EpochDataset:
    return EpochDataset(
        X=np.arange(8 * 2 * 4, dtype=np.float32).reshape(8, 2, 4),
        y=np.tile([0, 1], 4),
        subjects=np.repeat(["P01", "P02"], 4),
        trial_indices=np.tile(np.arange(4), 2),
        channel_names=["Cz", "Oz"],
        times=np.arange(4, dtype=np.float32),
        sfreq=4.0,
        task_id="c_vs_ip",
        class_ids=("c", "ip"),
        class_labels=("Congruent", "Incongruent Pure"),
        exclusions=[],
    )


def test_configuration_and_cohort_tables() -> None:
    config = configuration_table(get_task("c_vs_ip"), get_model_spec("riemann"))
    cohort = cohort_table(_dataset())

    assert "StratifiedGroupKFold" in config.loc[config["setting"] == "outer_cv", "value"].item()
    assert cohort["trials"].tolist() == [4, 4]


def test_saved_run_tables_and_plots(tmp_path) -> None:
    directory = tmp_path / "c_vs_ip" / "riemann"
    directory.mkdir(parents=True)
    (directory / "metadata.json").write_text(
        json.dumps({"task_id": "c_vs_ip", "model_id": "riemann"}),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "subject": ["P01", "P02"],
            "fold": [1, 1],
            "accuracy": [1.0, 0.5],
            "balanced_accuracy": [1.0, 0.5],
            "f1_macro": [1.0, 0.5],
            "roc_auc": [1.0, 0.5],
        }
    ).to_csv(directory / "metrics.csv", index=False)
    np.savez(
        directory / "predictions.npz",
        y_true=np.array([0, 1, 0, 1]),
        probabilities=np.array(
            [[0.8, 0.2], [0.1, 0.9], [0.6, 0.4], [0.7, 0.3]]
        ),
        trial_indices=np.arange(4),
        folds=np.ones(4),
    )

    run = load_saved_run(tmp_path, "c_vs_ip", "riemann")

    assert run is not None
    assert len(aggregate_metrics(run)) == 4
    assert confusion_table(run, ("c", "ip")).shape == (2, 2)
    assert prediction_diagnostics(run).iloc[0]["value"] == 4
    figure = plot_confusion(run, ("c", "ip"))
    assert figure.axes
    plt.close(figure)
    figure = plot_roc_pr_comparison(
        {"riemann": run},
        title="Binary comparison",
    )
    assert len(figure.axes) == 2
    plt.close(figure)


def test_multiclass_roc_pr_comparison_uses_micro_ovr(tmp_path) -> None:
    directory = tmp_path / "three_class" / "riemann"
    directory.mkdir(parents=True)
    (directory / "metadata.json").write_text(
        json.dumps({"task_id": "three_class", "model_id": "riemann"}),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "subject": ["P01"],
            "balanced_accuracy": [1.0],
        }
    ).to_csv(directory / "metrics.csv", index=False)
    np.savez(
        directory / "predictions.npz",
        y_true=np.array([0, 1, 2, 0, 1, 2]),
        probabilities=np.array(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
                [0.7, 0.2, 0.1],
                [0.2, 0.7, 0.1],
                [0.2, 0.1, 0.7],
            ]
        ),
        trial_indices=np.arange(6),
        folds=np.ones(6),
    )
    run = load_saved_run(tmp_path, "three_class", "riemann")

    figure = plot_roc_pr_comparison(
        {"three_class": run},
        title="Multiclass comparison",
    )

    assert "micro OVR" in figure.axes[0].get_legend().get_texts()[0].get_text()
    plt.close(figure)
