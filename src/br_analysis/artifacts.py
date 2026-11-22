from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ARTIFACT_VERSION = 3


@dataclass(frozen=True)
class RunMetadata:
    task_id: str
    model_id: str
    preprocessing_id: str
    class_ids: tuple[str, ...]
    random_seed: int
    created_at: str
    artifact_version: int = ARTIFACT_VERSION

    @classmethod
    def create(
        cls,
        *,
        task_id: str,
        model_id: str,
        preprocessing_id: str,
        class_ids: tuple[str, ...],
        random_seed: int,
    ) -> RunMetadata:
        return cls(
            task_id=task_id,
            model_id=model_id,
            preprocessing_id=preprocessing_id,
            class_ids=class_ids,
            random_seed=random_seed,
            created_at=datetime.now(UTC).isoformat(),
        )


def artifact_directory(
    artifacts_root: Path,
    task_id: str,
    model_id: str,
) -> Path:
    directory = artifacts_root / task_id / model_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_run_artifacts(
    directory: Path,
    *,
    metadata: RunMetadata,
    predictions: Mapping[str, np.ndarray],
    metrics: pd.DataFrame,
    extra_metadata: Mapping[str, Any] | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    metadata_payload = asdict(metadata)
    if extra_metadata:
        metadata_payload["extra"] = dict(extra_metadata)
    (directory / "metadata.json").write_text(
        json.dumps(metadata_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    np.savez_compressed(directory / "predictions.npz", **predictions)
    metrics.to_csv(directory / "metrics.csv", index=False)


def load_run_metadata(directory: Path) -> dict[str, Any]:
    return json.loads((directory / "metadata.json").read_text(encoding="utf-8"))


def stable_signature(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(serialized.encode("utf-8")).hexdigest()


def file_inventory_signature(paths: list[Path]) -> str:
    inventory = [
        {
            "name": path.name,
            "size": path.stat().st_size,
            "modified_ns": path.stat().st_mtime_ns,
        }
        for path in sorted(paths)
    ]
    return stable_signature({"files": inventory})


def implementation_signature(relative_paths: Sequence[str] | None = None) -> str:
    package_directory = Path(__file__).resolve().parent
    digest = sha256()
    paths = (
        [package_directory / relative_path for relative_path in relative_paths]
        if relative_paths is not None
        else sorted(package_directory.rglob("*.py"))
    )
    for path in paths:
        digest.update(path.relative_to(package_directory).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def reusable_run(
    directory: Path,
    *,
    run_signature: str,
    require_explanation: bool,
) -> bool:
    required = (directory / "metadata.json", directory / "metrics.csv", directory / "predictions.npz")
    if not all(path.is_file() for path in required):
        return False
    if require_explanation and not (directory / "shap_values.npz").is_file():
        return False
    try:
        metadata = load_run_metadata(directory)
    except (OSError, json.JSONDecodeError):
        return False
    return metadata.get("extra", {}).get("run_signature") == run_signature
