from __future__ import annotations

from pathlib import Path

from br_analysis.config import ProjectPaths


def test_local_paths_default_to_project_layout(tmp_path: Path) -> None:
    paths = ProjectPaths.from_env("unit-test", project_dir=tmp_path)
    assert paths.project_dir == tmp_path.resolve()
    assert paths.data_dir == tmp_path.resolve() / "data"
    assert paths.analysis_dir == tmp_path.resolve() / "artifacts" / "unit-test"
    assert paths.artifacts_dir.is_dir()
    assert not paths.cache_dir.exists()
    assert not paths.tables_dir.exists()
    assert not paths.figures_dir.exists()


def test_environment_can_override_data_and_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "input"
    artifacts_dir = tmp_path / "output"
    monkeypatch.setenv("BR_DATA_DIR", str(data_dir))
    monkeypatch.setenv("BR_ARTIFACTS_DIR", str(artifacts_dir))
    paths = ProjectPaths.from_env("run", project_dir=tmp_path)
    assert paths.data_dir == data_dir.resolve()
    assert paths.artifacts_dir == artifacts_dir.resolve()


def test_paths_can_be_discovered_without_creating_directories(tmp_path: Path) -> None:
    paths = ProjectPaths.from_env("report", project_dir=tmp_path, create_dirs=False)

    assert paths.artifacts_dir == tmp_path / "artifacts"
    assert not paths.artifacts_dir.exists()
