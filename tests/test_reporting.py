import json

import numpy as np
import pandas as pd
import pytest

from br_analysis.artifacts import artifact_directory, reusable_run, stable_signature
from br_analysis.report import generate_html_report
from br_analysis.reporting import collect_metrics, model_summary


@pytest.mark.parametrize(
    ("task_id", "class_ids", "chance_level"),
    [
        ("c_vs_ip", [0, 1], 0.5),
        ("c_vs_ip_vs_im", [0, 1, 2], 1 / 3),
    ],
)
def test_reporting_uses_task_specific_chance_level(
    tmp_path,
    task_id,
    class_ids,
    chance_level,
):
    run_directory = tmp_path / task_id / "riemann"
    run_directory.mkdir(parents=True)
    pd.DataFrame(
        {
            "subject": ["S01", "S02"],
            "balanced_accuracy": [0.5, 0.6],
            "f1_macro": [0.4, 0.5],
            "roc_auc": [0.5, 0.7],
        }
    ).to_csv(run_directory / "metrics.csv", index=False)
    (run_directory / "metadata.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "model_id": "riemann",
                "class_ids": class_ids,
            }
        ),
        encoding="utf-8",
    )

    summary = model_summary(collect_metrics(tmp_path))

    assert summary.loc[0, "n_classes"] == len(class_ids)
    assert summary.loc[0, "chance_level"] == pytest.approx(chance_level)


def test_model_summary_weights_subjects_equally(tmp_path) -> None:
    run_directory = tmp_path / "c_vs_ip" / "riemann"
    run_directory.mkdir(parents=True)
    pd.DataFrame(
        {
            "subject": ["S01", "S01", "S01", "S02"],
            "accuracy": [1.0, 1.0, 1.0, 0.0],
            "balanced_accuracy": [1.0, 1.0, 1.0, 0.0],
            "f1_macro": [1.0, 1.0, 1.0, 0.0],
            "roc_auc": [1.0, 1.0, 1.0, 0.0],
        }
    ).to_csv(run_directory / "metrics.csv", index=False)
    (run_directory / "metadata.json").write_text(
        json.dumps(
            {
                "task_id": "c_vs_ip",
                "model_id": "riemann",
                "class_ids": ["c", "ip"],
            }
        ),
        encoding="utf-8",
    )

    summary = model_summary(collect_metrics(tmp_path))

    assert summary.loc[0, "balanced_accuracy_mean"] == pytest.approx(0.5)
    assert summary.loc[0, "n_subjects"] == 2


def test_html_report_groups_saved_results_by_task(tmp_path):
    run_directory = tmp_path / "c_vs_ip" / "riemann"
    run_directory.mkdir(parents=True)
    pd.DataFrame(
        {
            "subject": ["S01"],
            "accuracy": [0.7],
            "balanced_accuracy": [0.7],
            "f1_macro": [0.68],
            "roc_auc": [0.75],
            "n_samples": [20],
        }
    ).to_csv(run_directory / "metrics.csv", index=False)
    (run_directory / "metadata.json").write_text(
        json.dumps(
            {
                "task_id": "c_vs_ip",
                "model_id": "riemann",
                "class_ids": ["c", "ip"],
            }
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        run_directory / "predictions.npz",
        y_true=np.array([0, 1]),
        probabilities=np.array([[0.8, 0.2], [0.3, 0.7]]),
    )

    report_path = generate_html_report(tmp_path)
    document = report_path.read_text(encoding="utf-8")

    assert report_path == tmp_path / "report" / "index.html"
    assert "Run coverage" in document
    assert "Metrics by subject" in document
    assert "True c" in document
    assert 'id="task-c_vs_ip"' in document
    assert "<th>model_id</th>" in document
    assert "<th>c_vs_ip</th>" in document
    assert "<td>riemann</td>" in document


def test_completed_run_is_reused_only_for_matching_signature(tmp_path):
    signature = stable_signature({"task": "c_vs_ip", "model": "riemann"})
    (tmp_path / "metadata.json").write_text(
        json.dumps({"extra": {"run_signature": signature}}),
        encoding="utf-8",
    )
    pd.DataFrame({"score": [0.5]}).to_csv(tmp_path / "metrics.csv", index=False)
    np.savez_compressed(tmp_path / "predictions.npz", y_true=np.array([0]))

    assert reusable_run(
        tmp_path,
        run_signature=signature,
        require_explanation=False,
    )
    assert not reusable_run(
        tmp_path,
        run_signature="different",
        require_explanation=False,
    )
    assert not reusable_run(
        tmp_path,
        run_signature=signature,
        require_explanation=True,
    )


def test_artifact_directory_has_one_run_per_task_and_model(tmp_path):
    directory = artifact_directory(tmp_path, "c_vs_ip", "riemann")

    assert directory == tmp_path / "c_vs_ip" / "riemann"
    assert directory.is_dir()
