from __future__ import annotations

import json

import pytest

import br_analysis.kaggle as kaggle_module
from br_analysis.config import ProjectPaths
from br_analysis.data import _preprocessing_cache_paths
from br_analysis.preprocessing import get_preprocessing_preset


def test_submit_builds_private_gpu_kernel(tmp_path, monkeypatch) -> None:
    paths = ProjectPaths.from_env(
        "test",
        project_dir=tmp_path,
        create_dirs=False,
    )
    calls = []
    monkeypatch.setattr(
        kaggle_module,
        "_run_kaggle",
        lambda arguments: calls.append(tuple(arguments)),
    )

    staging = kaggle_module.submit_kernel(
        paths,
        kernel_reference="user/br-former",
        source_dataset="user/br-source",
        data_dataset="user/br-eeg",
        task_id="c_vs_ip",
        model_id="st_eegformer",
    )

    metadata = json.loads(
        (staging / "kernel-metadata.json").read_text(encoding="utf-8")
    )
    script = (staging / "run.py").read_text(encoding="utf-8")
    compile(script, "run.py", "exec")
    assert metadata["enable_gpu"] is True
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["is_private"] is True
    assert metadata["dataset_sources"] == [
        "user/br-source",
        "user/br-eeg",
    ]
    assert 'Path("/kaggle/input") / slug' in script
    assert "'br-eeg'" in script
    assert "kagglehub.dataset_download(reference)" in script
    assert "--checkpoint" not in script
    assert "shutil.copytree(SOURCE_DIR, INSTALL_DIR" in script
    assert "shutil.unpack_archive(source_archive, INSTALL_DIR / \"src\")" in script
    assert "br-analysis[deep] @ {INSTALL_DIR.as_uri()}" in script
    assert '"--task",' in script
    assert "'c_vs_ip'" in script
    assert calls == [
        ("datasets", "status", "user/br-source"),
        ("datasets", "status", "user/br-eeg"),
        (
            "kernels",
            "push",
            "-p",
            str(staging),
            "--accelerator",
            "NvidiaTeslaT4",
        ),
    ]


def test_pretrained_submission_uses_public_checkpoint(tmp_path, monkeypatch) -> None:
    paths = ProjectPaths.from_env(
        "test",
        project_dir=tmp_path,
        create_dirs=False,
    )
    monkeypatch.setattr(kaggle_module, "_run_kaggle", lambda arguments: None)

    staging = kaggle_module.submit_kernel(
        paths,
        kernel_reference="user/br-former",
        source_dataset="user/br-source",
        data_dataset="user/br-eeg",
        task_id="c_vs_ip",
        model_id="st_eegformer",
    )

    script = (staging / "run.py").read_text(encoding="utf-8")
    assert "--checkpoint" not in script
    assert '"--model",' in script
    assert "'st_eegformer'" in script


def test_publish_data_uses_preprocessed_cache_not_fif(
    tmp_path,
    monkeypatch,
) -> None:
    paths = ProjectPaths.from_env(
        "test",
        project_dir=tmp_path,
        create_dirs=False,
    )
    paths.data_dir.mkdir()
    fif_path = paths.data_dir / "P01.fif"
    fif_path.write_bytes(b"large-original-placeholder")
    cache_files = []
    for preset_id in ("pretrained", "labram_pretrained"):
        analysis_window = (0.0, 1.5)
        cache_data, cache_metadata = _preprocessing_cache_paths(
            paths,
            fif_path,
            get_preprocessing_preset(preset_id),
            analysis_window,
        )
        cache_data.parent.mkdir(parents=True, exist_ok=True)
        cache_data.write_bytes(f"portable-{preset_id}".encode())
        cache_metadata.write_text("{}", encoding="utf-8")
        cache_files.append(cache_data)
    monkeypatch.setattr(kaggle_module, "_run_kaggle", lambda arguments: None)

    staging = kaggle_module.publish_data_dataset(
        paths,
        "user/br-eeg",
    )

    manifest = json.loads(
        (staging / "preprocessed-manifest.json").read_text(encoding="utf-8")
    )
    assert not (staging / "P01.fif").exists()
    assert all((staging / path.name).is_file() for path in cache_files)
    assert all(
        variant["subjects"][0]["subject"] == "P01"
        for variant in manifest["variants"]
    )


def test_download_merge_refuses_implicit_replacement(tmp_path) -> None:
    source = tmp_path / "remote"
    destination = tmp_path / "local"
    (source / "task" / "model").mkdir(parents=True)
    (destination / "task" / "model").mkdir(parents=True)
    (source / "task" / "model" / "metrics.csv").write_text(
        "remote",
        encoding="utf-8",
    )
    target = destination / "task" / "model" / "metrics.csv"
    target.write_text("local", encoding="utf-8")

    with pytest.raises(FileExistsError):
        kaggle_module._merge_artifacts(
            source,
            destination,
            replace=False,
        )

    kaggle_module._merge_artifacts(source, destination, replace=True)
    assert target.read_text(encoding="utf-8") == "remote"


def test_dataset_reference_validation() -> None:
    assert kaggle_module._validate_reference("user/dataset-name") == (
        "user/dataset-name"
    )
    with pytest.raises(ValueError, match="owner/slug"):
        kaggle_module._validate_reference("dataset-name")
