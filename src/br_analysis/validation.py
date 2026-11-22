from __future__ import annotations

import mne
import numpy as np

from .config import ProjectPaths


def validate_data(max_files: int | None = None) -> int:
    paths = ProjectPaths.from_env("validation")
    files = paths.data_files[:max_files] if max_files is not None else paths.data_files
    if not files:
        raise FileNotFoundError(f"No FIF files found in {paths.data_dir}.")

    observed_codes: set[int] = set()
    failures: list[str] = []
    for fif_path in files:
        try:
            epochs = mne.read_epochs(fif_path, preload=False, verbose=False)
        except (OSError, RuntimeError, ValueError) as exc:
            failures.append(f"{fif_path.name}: {exc}")
            continue
        observed_codes.update(np.unique(epochs.events[:, 2]).astype(int).tolist())

    print(f"Data directory: {paths.data_dir}")
    print(f"Valid FIF files: {len(files) - len(failures)}/{len(files)}")
    print(f"Observed event codes: {sorted(observed_codes)}")
    for failure in failures:
        print(f"ERROR: {failure}")
    return 1 if failures else 0
