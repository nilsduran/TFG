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
            self.analysis_dir,
            self.cache_dir,
            self.tables_dir,
            self.figures_dir,
        ):
            try:
                folder.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                # Ignore read-only filesystem errors if the folder already exists
                if folder.exists():
                    continue
                raise e

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
    import os
    import sys
    in_kaggle = "KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.exists("/kaggle/working")
    is_colab = in_colab or 'google.colab' in sys.modules or 'COLAB_GPU' in os.environ or os.path.exists('/content/drive')
    
    if in_kaggle:
        # On Kaggle, code and data may be mounted from separate datasets. Prefer
        # an explicitly uploaded/extracted project folder, then fall back to the
        # package parent for bundles that include TFG_Nils_2026 next to br_analysis.
        search_roots = [Path("/kaggle/input"), Path("/kaggle/working"), Path(__file__).resolve().parent.parent]
        input_project_dir = None
        for root in search_roots:
            if not root.exists():
                continue
            candidates = [root / project_folder]
            candidates.extend(path for path in root.glob("**/" + project_folder) if path.is_dir())
            candidates.extend(path for path in root.glob("**") if path.is_dir() and (path / "02_Data").exists())
            for candidate in candidates:
                if candidate.exists() and ((candidate / "02_Data").exists() or (candidate / "04_Anàlisi").exists()):
                    input_project_dir = candidate
                    break
            if input_project_dir is not None:
                break
        if input_project_dir is None:
            input_project_dir = Path(__file__).resolve().parent.parent / project_folder
        output_project_dir = Path("/kaggle/working/TFG_Nils_2026")
        
        paths = ProjectPaths(
            project_dir=output_project_dir,
            data_dir=input_project_dir / "02_Data" / "01_Preprocessed_epochs",
            analysis_root=input_project_dir / "04_Anàlisi",
            analysis_dir=output_project_dir / "04_Anàlisi" / analysis_name,
            cache_dir=output_project_dir / "04_Anàlisi" / analysis_name / "cache",
            tables_dir=output_project_dir / "04_Anàlisi" / analysis_name / "tables",
            figures_dir=output_project_dir / "04_Anàlisi" / analysis_name / "figures",
        )
    else:
        drive_root = Path(colab_drive_root if is_colab else local_drive_root)
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


COMMON_BAND_RANGES = {
    "theta": (5.0, 7.0),
    "alpha": (8.0, 12.0),
}

TARGET_BANDS_CLUSTERS = {
    "theta": ["Fp1", "Fp2", "Fz", "F1", "F2", "FCz", "FC1", "FC2", "Cz"],
    "alpha": ["Pz", "P1", "P2", "P3", "P4", "P7", "P8", "Oz", "O1", "O2", "PO3", "PO4", "POz"],
}


def get_task_config(
    task_name: str,
    analysis_root: Path,
    *,
    name: str | None = None,
    analysis_window: tuple[float, float] | None = None,
) -> BinaryTaskConfig:
    """Get standardized BinaryTaskConfig for the given task name."""
    if task_name == "c_vs_ip":
        return BinaryTaskConfig(
            name=name or "c_vs_i",
            class_0="Congruent",
            class_1="Incongruent_Pure",
            event_codes={"Congruent": [1, 2], "Incongruent_Pure": [3, 4, 5, 6]},
            selected_subjects_csv=analysis_root / "exploration" / "tables" / "subjectes_seleccionats.csv",
            max_subjects=24,
            undersample_classes=True,
            analysis_window=analysis_window or (0.0, 1.5),
        )
    elif task_name == "im_vs_ip":
        return BinaryTaskConfig(
            name=name or "im_vs_ip",
            class_0="Incongruent_Mixed",
            class_1="Incongruent_Pure",
            event_codes={"Incongruent_Mixed": [7, 8], "Incongruent_Pure": [3, 4, 5, 6]},
            selected_subjects_csv=analysis_root / "exploration" / "tables" / "subjectes_seleccionats_im_vs_ip.csv",
            max_subjects=29,
            undersample_classes=True,
            analysis_window=analysis_window or (0.0, 1.5),
        )
    else:
        raise ValueError(f"Unknown task name: {task_name}")


def get_epochs_grid(task_name: str) -> list[int]:
    """Get the customized number of epochs for the given task name."""
    if task_name == "c_vs_ip":
        return [20, 40, 70, 100, 150, 200, 300, 380]
    elif task_name == "im_vs_ip":
        return [20, 40, 60, 100, 150, 200, 240, 270]
    else:
        raise ValueError(f"Unknown task name: {task_name}")
