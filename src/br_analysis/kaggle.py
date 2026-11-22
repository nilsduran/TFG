from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from .config import ProjectPaths
from .data import (
    PORTABLE_PREPROCESSING_VERSION,
    _preprocessing_cache_paths,
)
from .preparation import preparation_specs
from .preprocessing import get_preprocessing_preset
from .registry import MODELS, get_model_spec
from .tasks import get_task

KAGGLE_REFERENCE = re.compile(r"^[A-Za-z0-9_-]+/[A-Za-z0-9_-]+$")
DEFAULT_REMOTE_MODELS = ("st_eegformer", "labram")
KAGGLE_GPU = "NvidiaTeslaT4"


def publish_source_dataset(
    paths: ProjectPaths,
    reference: str,
    *,
    update: bool = False,
) -> Path:
    reference = _validate_reference(reference)
    staging = _reset_staging(paths, "source_dataset")
    for filename in ("pyproject.toml", "README.md", "requirements.txt"):
        source = paths.project_dir / filename
        if source.is_file():
            shutil.copy2(source, staging / filename)
    shutil.copytree(
        paths.project_dir / "src" / "br_analysis",
        staging / "src" / "br_analysis",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    _write_dataset_metadata(staging, reference, "BR analysis source")
    _publish_dataset(
        staging,
        update=update,
        message="Update analysis source",
        directory_mode="zip",
    )
    return staging


def publish_data_dataset(
    paths: ProjectPaths,
    reference: str,
    *,
    model_ids: Sequence[str] | None = None,
    update: bool = False,
) -> Path:
    reference = _validate_reference(reference)
    files = paths.data_files
    if not files:
        raise FileNotFoundError(f"No FIF files found in {paths.data_dir}.")
    staging = _reset_staging(paths, "eeg_dataset")
    selected_models = tuple(model_ids or DEFAULT_REMOTE_MODELS)
    invalid_models = [
        model_id
        for model_id in selected_models
        if model_id not in MODELS or MODELS[model_id].family != "torch"
    ]
    if invalid_models:
        raise ValueError(
            "Kaggle data can only be prepared for torch models: "
            + ", ".join(invalid_models)
        )
    variants = []
    missing = []
    for spec in preparation_specs(selected_models):
        preprocessing = get_preprocessing_preset(spec.preset_id)
        subjects = []
        for source in files:
            data_path, metadata_path = _preprocessing_cache_paths(
                paths,
                source,
                preprocessing,
                spec.analysis_window,
            )
            if not data_path.is_file() or not metadata_path.is_file():
                missing.append(f"{source.stem}/{spec.preset_id}")
                continue
            _hardlink(data_path, staging / data_path.name)
            _hardlink(metadata_path, staging / metadata_path.name)
            subjects.append(
                {
                    "subject": source.stem,
                    "data_file": data_path.name,
                    "metadata_file": metadata_path.name,
                }
            )
        variants.append(
            {
                "id": spec.preset_id,
                "analysis_window": spec.analysis_window,
                "preprocessing": {
                    "bandpass": preprocessing.bandpass,
                    "resample_sfreq": preprocessing.resample_sfreq,
                },
                "subjects": subjects,
            }
        )
    if missing:
        raise FileNotFoundError(
            "Missing current preprocessing caches for "
            f"{', '.join(missing[:8])}{'...' if len(missing) > 8 else ''}. "
            "Run: br-analysis prepare-data --skip-power"
        )
    (staging / "preprocessed-manifest.json").write_text(
        json.dumps(
            {
                "format_version": PORTABLE_PREPROCESSING_VERSION,
                "variants": variants,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_dataset_metadata(
        staging,
        reference,
        "Binocular rivalry preprocessed EEG",
    )
    _publish_dataset(
        staging,
        update=update,
        message="Update preprocessed EEG",
    )
    return staging


def submit_kernel(
    paths: ProjectPaths,
    *,
    kernel_reference: str,
    source_dataset: str,
    data_dataset: str,
    task_id: str,
    model_id: str,
    allow_unpromising: bool = False,
) -> Path:
    kernel_reference = _validate_reference(kernel_reference)
    source_dataset = _validate_reference(source_dataset)
    data_dataset = _validate_reference(data_dataset)
    get_task(task_id)
    model = get_model_spec(model_id)
    if model.family != "torch":
        raise ValueError("Kaggle GPU submission is intended for torch models.")
    staging = _reset_staging(paths, "kernel")
    dataset_sources = [source_dataset, data_dataset]
    for dataset_reference in dataset_sources:
        try:
            _run_kaggle(("datasets", "status", dataset_reference))
        except RuntimeError as exc:
            raise ValueError(
                f"Kaggle dataset is unavailable: {dataset_reference}"
            ) from exc
    metadata = {
        "id": kernel_reference,
        "title": f"BR {model_id} {task_id}",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "machine_shape": KAGGLE_GPU,
        "enable_internet": True,
        "dataset_sources": dataset_sources,
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (staging / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    (staging / "run.py").write_text(
        _kernel_script(
            source_dataset=source_dataset,
            data_dataset=data_dataset,
            task_id=task_id,
            model_id=model_id,
            allow_unpromising=allow_unpromising,
        ),
        encoding="utf-8",
    )
    _run_kaggle(
        (
            "kernels",
            "push",
            "-p",
            str(staging),
            "--accelerator",
            KAGGLE_GPU,
        )
    )
    return staging


def kernel_status(reference: str) -> None:
    _run_kaggle(("kernels", "status", _validate_reference(reference)))


def download_kernel_outputs(
    paths: ProjectPaths,
    reference: str,
    *,
    replace: bool = False,
) -> Path:
    reference = _validate_reference(reference)
    staging = _reset_staging(
        paths,
        Path("downloads") / reference.split("/", 1)[1],
    )
    _run_kaggle(("kernels", "output", reference, "-p", str(staging)))
    remote_artifacts = staging / "artifacts"
    if not remote_artifacts.is_dir():
        raise FileNotFoundError(
            f"Kaggle output did not contain an artifacts directory: {staging}"
        )
    _merge_artifacts(
        remote_artifacts,
        paths.artifacts_dir,
        replace=replace,
    )
    return staging


def _kernel_script(
    *,
    source_dataset: str,
    data_dataset: str,
    task_id: str,
    model_id: str,
    allow_unpromising: bool,
) -> str:
    source_slug = source_dataset.split("/", 1)[1]
    data_slug = data_dataset.split("/", 1)[1]
    return f"""from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

def dataset_path(reference: str, slug: str) -> Path:
    legacy_path = Path("/kaggle/input") / slug
    if legacy_path.is_dir():
        return legacy_path
    import kagglehub

    return Path(kagglehub.dataset_download(reference))


SOURCE_DIR = dataset_path({source_dataset!r}, {source_slug!r})
DATA_DIR = dataset_path({data_dataset!r}, {data_slug!r})
INSTALL_DIR = Path("/kaggle/working/br-analysis-source")
ARTIFACTS_DIR = Path("/kaggle/working/artifacts")

shutil.copytree(SOURCE_DIR, INSTALL_DIR, dirs_exist_ok=True)
source_archive = INSTALL_DIR / "src.zip"
if source_archive.is_file():
    shutil.unpack_archive(source_archive, INSTALL_DIR / "src")
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        f"br-analysis[deep] @ {{INSTALL_DIR.as_uri()}}",
    ],
    check=True,
)
os.environ["BR_PROJECT_ROOT"] = str(INSTALL_DIR)
os.environ["BR_DATA_DIR"] = str(DATA_DIR)
os.environ["BR_ARTIFACTS_DIR"] = str(ARTIFACTS_DIR)

command = [
    sys.executable,
    "-m",
    "br_analysis.cli",
    "run",
    "--task",
    {task_id!r},
    "--model",
    {model_id!r},
    "--force",
]
if {allow_unpromising!r}:
    command.append("--allow-unpromising")
subprocess.run(command, check=True)
"""


def _publish_dataset(
    staging: Path,
    *,
    update: bool,
    message: str,
    directory_mode: str = "skip",
) -> None:
    if update:
        _run_kaggle(
            (
                "datasets",
                "version",
                "-p",
                str(staging),
                "-m",
                message,
                "--dir-mode",
                directory_mode,
            )
        )
    else:
        _run_kaggle(
            (
                "datasets",
                "create",
                "-p",
                str(staging),
                "--dir-mode",
                directory_mode,
            )
        )


def _run_kaggle(arguments: Sequence[str]) -> None:
    executable = shutil.which("kaggle")
    if executable is None:
        raise RuntimeError(
            'Kaggle CLI is not installed. Run: python -m pip install -e ".[cloud]"'
        )
    completed = subprocess.run(
        [executable, *arguments],
        check=False,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"Kaggle CLI failed with exit code {completed.returncode}."
        )


def _reset_staging(paths: ProjectPaths, relative: str | Path) -> Path:
    root = (paths.artifacts_dir / "_cloud" / "kaggle").resolve()
    staging = (root / relative).resolve()
    if staging != root and root not in staging.parents:
        raise ValueError("Kaggle staging path escaped its root.")
    if staging.is_dir():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    return staging


def _write_dataset_metadata(
    staging: Path,
    reference: str,
    title: str,
) -> None:
    (staging / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "id": reference,
                "title": title,
                "licenses": [{"name": "other"}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _hardlink(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError as exc:
        raise OSError(
            f"Could not create a space-efficient hardlink for {source}. "
            "Keep BR_ARTIFACTS_DIR on the same drive as the source data."
        ) from exc


def _merge_artifacts(
    source: Path,
    destination: Path,
    *,
    replace: bool,
) -> None:
    files = [path for path in source.rglob("*") if path.is_file()]
    conflicts = [
        destination / path.relative_to(source)
        for path in files
        if (destination / path.relative_to(source)).exists()
    ]
    if conflicts and not replace:
        raise FileExistsError(
            f"{conflicts[0]} already exists. Use --replace to import this run."
        )
    for file_path in files:
        relative = file_path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, target)


def _validate_reference(reference: str) -> str:
    if not KAGGLE_REFERENCE.fullmatch(reference):
        raise ValueError(
            f"Invalid Kaggle reference {reference!r}; expected owner/slug."
        )
    return reference
