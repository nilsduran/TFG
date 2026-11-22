from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .tasks import TaskSpec, get_task


@dataclass(frozen=True)
class ProjectPaths:
    """Local filesystem layout for data and generated artifacts."""

    project_dir: Path
    data_dir: Path
    artifacts_dir: Path
    analysis_dir: Path
    cache_dir: Path
    tables_dir: Path
    figures_dir: Path

    @classmethod
    def from_env(
        cls,
        analysis_name: str = "default",
        *,
        project_dir: str | Path | None = None,
        data_dir: str | Path | None = None,
        artifacts_dir: str | Path | None = None,
        create_dirs: bool = True,
    ) -> ProjectPaths:
        package_root = Path(__file__).resolve().parents[2]
        root = Path(project_dir or os.environ.get("BR_PROJECT_ROOT", package_root)).expanduser().resolve()
        data = Path(data_dir or os.environ.get("BR_DATA_DIR", root / "data")).expanduser().resolve()
        artifacts = Path(
            artifacts_dir or os.environ.get("BR_ARTIFACTS_DIR", root / "artifacts")
        ).expanduser().resolve()
        analysis = artifacts / analysis_name
        paths = cls(
            project_dir=root,
            data_dir=data,
            artifacts_dir=artifacts,
            analysis_dir=analysis,
            cache_dir=analysis / "cache",
            tables_dir=analysis / "tables",
            figures_dir=analysis / "figures",
        )
        if create_dirs:
            paths.ensure()
        return paths

    def ensure(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    @property
    def analysis_root(self) -> Path:
        return self.artifacts_dir

    @property
    def data_files(self) -> list[Path]:
        return sorted(self.data_dir.glob("*.fif"))


@dataclass(frozen=True)
class BandChannelSpec:
    target_band_channels: Sequence[tuple[str, str]]
    band_ranges: Mapping[str, tuple[float, float]]

    @property
    def channel_names(self) -> list[str]:
        return [f"{channel}-{band}" for channel, band in self.target_band_channels]


@dataclass(frozen=True)
class FusedBandSpec:
    target_bands_clusters: Mapping[str, Sequence[str]]
    band_ranges: Mapping[str, tuple[float, float]]
    channel_mode: Sequence[str] = ("fused",)
    min_cluster_fraction: float = 0.5


COMMON_BAND_RANGES = {
    "theta": (5.0, 7.0),
    "alpha": (8.0, 12.0),
}

TARGET_BANDS_CLUSTERS = {
    "theta": ["Fp1", "Fp2", "Fz", "F1", "F2", "FCz", "FC1", "FC2", "Cz"],
    "alpha": ["Pz", "P1", "P2", "P3", "P4", "P7", "P8", "Oz", "O1", "O2", "PO3", "PO4", "POz"],
}

BinaryTaskConfig = TaskSpec


def detect_project_paths(
    analysis_name: str,
    **legacy_options: object,
) -> ProjectPaths:
    """Compatibility wrapper around the local path configuration."""
    supported = {"project_dir", "data_dir", "artifacts_dir"}
    options = {key: value for key, value in legacy_options.items() if key in supported}
    return ProjectPaths.from_env(analysis_name, **options)


def get_task_config(
    task_name: str,
    analysis_root: Path | None = None,
    *,
    name: str | None = None,
    analysis_window: tuple[float, float] | None = None,
) -> TaskSpec:
    del analysis_root, name
    return get_task(task_name, analysis_window=analysis_window)
