import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import br_analysis.cli as cli_module
from br_analysis.cli import (
    _build_parser,
    _checkpoint_paths,
    _selection,
    main,
)
from br_analysis.registry import MODELS
from br_analysis.splits import EVALUATION_PROTOCOL
from br_analysis.tasks import TASKS


def test_help_command_matches_help_flags(capsys) -> None:
    expected = _build_parser().format_help()

    assert main(["help"]) == 0
    assert capsys.readouterr().out == expected

    for flag in ("--help", "-h"):
        with pytest.raises(SystemExit) as error:
            main([flag])
        assert error.value.code == 0
        assert capsys.readouterr().out == expected


def test_list_models_prints_every_configured_model(capsys) -> None:
    assert main(["list-models"]) == 0

    output = capsys.readouterr().out
    for model in MODELS.values():
        assert f"{model.id}:" in output
        assert f"family={model.family}" in output
        assert f"preprocessing={model.preprocessing}" in output
        assert f"representation={model.representation}" in output


def test_help_lists_only_completed_runs(tmp_path, monkeypatch) -> None:
    artifacts = tmp_path / "artifacts"
    complete = artifacts / "c_vs_ip" / "riemann"
    incomplete = artifacts / "r_vs_g" / "svc_agnostic"
    complete.mkdir(parents=True)
    incomplete.mkdir(parents=True)
    metadata = {
        "task_id": "c_vs_ip",
        "model_id": "riemann",
        "extra": {"evaluation_protocol": EVALUATION_PROTOCOL},
    }
    (complete / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (incomplete / "metadata.json").write_text(
        json.dumps({"task_id": "r_vs_g", "model_id": "svc_agnostic"}),
        encoding="utf-8",
    )
    pd.DataFrame({"score": [0.5]}).to_csv(complete / "metrics.csv", index=False)
    np.savez(complete / "predictions.npz", y_true=np.array([0]))
    monkeypatch.setenv("BR_PROJECT_ROOT", str(tmp_path))

    help_text = _build_parser().format_help()

    assert "br-analysis run --task c_vs_ip --model riemann" in help_text
    riemann_row = next(line for line in help_text.splitlines() if line.strip().startswith("riemann"))
    svc_row = next(line for line in help_text.splitlines() if line.strip().startswith("svc_agnostic"))
    assert "X" in riemann_row
    assert "X" not in svc_row


def test_all_selection_expands_in_registry_order() -> None:
    assert _selection("all", ("a", "b"), "item") == ["a", "b"]
    assert _selection("b", ("a", "b"), "item") == ["b"]


def test_batch_checkpoint_mapping_supports_named_paths() -> None:
    paths = _checkpoint_paths(
        ["st_eegformer=former.pt", "labram=labram.pt"],
        ["st_eegformer", "labram"],
    )
    assert paths == {
        "st_eegformer": Path("former.pt"),
        "labram": Path("labram.pt"),
    }


def test_all_tasks_runs_each_task_for_selected_model(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    calls = []

    def fake_run(**kwargs):
        calls.append((kwargs["task_id"], kwargs["model_id"], kwargs["n_jobs"]))
        return Path("artifacts") / kwargs["task_id"] / kwargs["model_id"]

    monkeypatch.setattr(cli_module, "_run_combination", fake_run)
    monkeypatch.setenv("BR_PROJECT_ROOT", str(tmp_path))

    assert main(
        ["run", "--task", "all", "--model", "riemann", "--jobs", "4"]
    ) == 0
    assert calls == [(task_id, "riemann", 4) for task_id in TASKS]
    assert "6 completed" in capsys.readouterr().out


def test_all_models_runs_each_model_for_selected_task(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    calls = []

    def fake_run(**kwargs):
        calls.append((kwargs["task_id"], kwargs["model_id"]))
        return Path("artifacts") / kwargs["task_id"] / kwargs["model_id"]

    monkeypatch.setattr(cli_module, "_run_combination", fake_run)
    monkeypatch.setenv("BR_PROJECT_ROOT", str(tmp_path))

    arguments = [
        "run",
        "--task",
        "c_vs_ip",
        "--model",
        "all",
    ]
    assert main(arguments) == 0
    assert calls == [("c_vs_ip", model_id) for model_id in MODELS]
    assert f"{len(MODELS)} completed" in capsys.readouterr().out


def test_batch_skips_completed_artifacts_without_force(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    completed = tmp_path / "artifacts" / "c_vs_ip" / "lr_theta"
    completed.mkdir(parents=True)
    (completed / "metadata.json").write_text(
        json.dumps(
            {
                "task_id": "c_vs_ip",
                "model_id": "lr_theta",
                "extra": {"evaluation_protocol": EVALUATION_PROTOCOL},
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame({"score": [0.5]}).to_csv(
        completed / "metrics.csv",
        index=False,
    )
    np.savez(completed / "predictions.npz", y_true=np.array([0]))
    calls = []

    def fake_run(**kwargs):
        calls.append((kwargs["task_id"], kwargs["model_id"]))
        return tmp_path / "artifacts" / kwargs["task_id"] / kwargs["model_id"]

    monkeypatch.setattr(cli_module, "_run_combination", fake_run)
    monkeypatch.setenv("BR_PROJECT_ROOT", str(tmp_path))

    assert main(["run", "--task", "all", "--model", "lr_theta"]) == 0
    assert ("c_vs_ip", "lr_theta") not in calls
    assert len(calls) == len(TASKS) - 1
    output = capsys.readouterr().out
    assert "SKIP: completed artifacts already exist" in output
    assert "1 skipped" in output
