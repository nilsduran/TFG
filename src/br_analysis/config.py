from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem layout used by the notebooks."""

    project_dir: Path
    data_dir: Path
    analysis_root: Path
    analysis_dir: Path
    cache_dir: Path
    tables_dir: Path
    figures_dir: Path

    def ensure(self) -> None:
        for folder in (
            self.data_dir,
            self.analysis_root,
            self.analysis_dir,
            self.cache_dir,
            self.tables_dir,
            self.figures_dir,
        ):
            folder.mkdir(parents=True, exist_ok=True)

    @property
    def data_files(self) -> list[Path]:
        return sorted(self.data_dir.glob("*.fif"))


@dataclass(frozen=True)
class BinaryTaskConfig:
    """Definition of a two-class event decoding task."""

    name: str
    class_0: str
    class_1: str
    event_codes: Mapping[str, Sequence[int]]
    selected_subjects_csv: Path
    max_subjects: int | None = None
    undersample_classes: bool = True
    analysis_window: tuple[float, float] = (0.0, 1.5)
    random_seed: int = 42

    @property
    def cond_events(self) -> dict[str, list[str]]:
        return {name: [f"event_{code}" for code in codes] for name, codes in self.event_codes.items()}


@dataclass(frozen=True)
class BandChannelSpec:
    """Single-channel band-pass features, e.g. FCz-theta and Oz-alpha."""

    target_band_channels: Sequence[tuple[str, str]]
    band_ranges: Mapping[str, tuple[float, float]]

    @property
    def channel_names(self) -> list[str]:
        return [f"{ch_name}-{band}" for ch_name, band in self.target_band_channels]


@dataclass(frozen=True)
class FusedBandSpec:
    """Multi-band channel clusters concatenated along the sensor axis."""

    target_bands_clusters: Mapping[str, Sequence[str]]
    band_ranges: Mapping[str, tuple[float, float]]
    channel_mode: Sequence[str] = ("fused",)
    min_cluster_fraction: float = 0.5


def detect_project_paths(
    analysis_name: str,
    *,
    in_colab: bool = False,
    local_drive_root: str | Path = r"H:\.shortcut-targets-by-id\19KyEKxqClpX2n1-oU8TXRrmUspaPJKNU",
    colab_drive_root: str | Path = "/content/drive/MyDrive",
    project_folder: str = "TFG_Nils_2026",
) -> ProjectPaths:
    drive_root = Path(colab_drive_root if in_colab else local_drive_root)
    project_dir = drive_root / project_folder
    analysis_root = project_dir / "04_Anàlisi"
    analysis_dir = analysis_root / analysis_name
    paths = ProjectPaths(
        project_dir=project_dir,
        data_dir=project_dir / "02_Data" / "01_Preprocessed_epochs",
        analysis_root=analysis_root,
        analysis_dir=analysis_dir,
        cache_dir=analysis_dir / "cache",
        tables_dir=analysis_dir / "tables",
        figures_dir=analysis_dir / "figures",
    )
    paths.ensure()
    return paths
