"""Subject-wise benchmark harness with selection on development data only.

For each subject the first 80% of trials (chronological) is the development
period and the last 20% is the untouched holdout, exactly as in the main
pipeline. Each candidate gets:

* ``dev_bacc``: balanced accuracy of out-of-fold predictions from blocked,
  grouped cross-validation inside the development period. This is the only
  number used to *choose* between candidates.
* ``holdout_bacc``: one fit on the full development period, scored once on
  the holdout. This is the number that is *reported*.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from ..config import ProjectPaths
from ..data import load_epoch_dataset
from ..dataset import EpochDataset
from ..splits import SplitPlan, development_splits
from ..tasks import CONDITIONS, TaskSpec, get_task
from .pipelines import CANDIDATES, Candidate
from .stats import balanced_accuracy, permutation_null

STIMULUS_WINDOW = (0.0, 1.5)
PRESET_CACHE_WINDOW = {"broadband": (0.0, 1.5), "erp": (-0.74, 1.5), "band_limited": (0.0, 1.5)}

# Control tasks that hold the reported colour fixed, isolating interocular conflict
# from the physical colour composition of the stimulus.
CONTROL_TASKS: dict[str, TaskSpec] = {
    "cr_vs_ipr": TaskSpec("cr_vs_ipr", (CONDITIONS["cr"], CONDITIONS["ipr"])),
    "cg_vs_ipg": TaskSpec("cg_vs_ipg", (CONDITIONS["cg"], CONDITIONS["ipg"])),
}


def resolve_task(task_id: str) -> TaskSpec:
    return CONTROL_TASKS[task_id] if task_id in CONTROL_TASKS else get_task(task_id)


@cache
def load_stimulus_dataset(task_id: str, preset: str) -> EpochDataset:
    """Load a task restricted to the stimulus window (0-1.5 s) from the shared caches."""
    paths = ProjectPaths.from_env("research", create_dirs=False)
    task = replace(resolve_task(task_id), analysis_window=PRESET_CACHE_WINDOW[preset])
    dataset = load_epoch_dataset(paths, task, preset, verbose=False)
    mask = (dataset.times >= STIMULUS_WINDOW[0] - 1e-9) & (dataset.times < STIMULUS_WINDOW[1] - 1e-9)
    if mask.all():
        return dataset
    return replace(dataset, X=np.ascontiguousarray(dataset.X[..., mask]), times=dataset.times[mask])


@dataclass
class SubjectOutcome:
    subject: str
    dev_bacc: float
    holdout_bacc: float
    holdout_auc: float
    n_dev: int
    n_holdout: int
    holdout_y: np.ndarray
    holdout_proba: np.ndarray
    dev_y: np.ndarray
    dev_proba: np.ndarray


def _fit_predict(estimator, X_train, y_train, X_test):
    model = clone(estimator).fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    aligned = np.zeros((len(X_test), 2))
    aligned[:, np.asarray(model.classes_, dtype=int)] = proba
    return aligned


def evaluate_subject(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    subject: str,
    plan: SplitPlan = SplitPlan(),
) -> SubjectOutcome | None:
    splits, holdout = development_splits(y, plan)
    if not splits or len(np.unique(y[holdout])) < 2:
        return None
    n_dev = len(y) - len(holdout)
    dev_proba = np.full((n_dev, 2), np.nan)
    with threadpool_limits(1):
        for train, test in splits:
            dev_proba[test] = _fit_predict(estimator, X[train], y[train], X[test])
        holdout_proba = _fit_predict(estimator, X[:n_dev], y[:n_dev], X[holdout])
    covered = ~np.isnan(dev_proba[:, 0])
    dev_y = y[:n_dev]
    return SubjectOutcome(
        subject=subject,
        dev_bacc=balanced_accuracy(dev_y[covered], dev_proba[covered].argmax(1), 2),
        holdout_bacc=balanced_accuracy(y[holdout], holdout_proba.argmax(1), 2),
        holdout_auc=float(roc_auc_score(y[holdout], holdout_proba[:, 1])),
        n_dev=n_dev,
        n_holdout=len(holdout),
        holdout_y=y[holdout],
        holdout_proba=holdout_proba,
        dev_y=dev_y,
        dev_proba=dev_proba,
    )


def results_dir(task_id: str) -> Path:
    paths = ProjectPaths.from_env("research", create_dirs=False)
    directory = paths.artifacts_dir / "research" / task_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def run_candidate(
    task_id: str,
    candidate: Candidate | str,
    *,
    n_jobs: int = 8,
    force: bool = False,
    n_permutations: int = 1000,
) -> pd.DataFrame:
    """Evaluate one candidate on every subject; results are cached on disk."""
    candidate = CANDIDATES[candidate] if isinstance(candidate, str) else candidate
    out_dir = results_dir(task_id)
    table_path = out_dir / f"{candidate.id}.csv"
    if table_path.exists() and not force:
        return pd.read_csv(table_path)

    dataset = load_stimulus_dataset(task_id, candidate.preset)
    estimator = candidate.build(dataset.sfreq)
    subjects = dataset.subject_ids
    jobs = (
        delayed(evaluate_subject)(
            estimator,
            dataset.X[dataset.subjects == subject],
            dataset.y[dataset.subjects == subject].astype(int),
            subject,
        )
        for subject in subjects
    )
    outcomes = [o for o in Parallel(n_jobs=n_jobs)(jobs) if o is not None]
    rows = []
    arrays: dict[str, np.ndarray] = {}
    for index, outcome in enumerate(outcomes):
        null = permutation_null(
            outcome.holdout_y, outcome.holdout_proba.argmax(1), 2, n_permutations, seed=index
        )
        rows.append(
            {
                "subject": outcome.subject,
                "dev_bacc": outcome.dev_bacc,
                "holdout_bacc": outcome.holdout_bacc,
                "holdout_auc": outcome.holdout_auc,
                "holdout_perm_p": float(np.mean(null >= null[0])),
                "n_dev": outcome.n_dev,
                "n_holdout": outcome.n_holdout,
            }
        )
        arrays[f"{outcome.subject}__null"] = null
        arrays[f"{outcome.subject}__holdout_y"] = outcome.holdout_y
        arrays[f"{outcome.subject}__holdout_proba"] = outcome.holdout_proba
        arrays[f"{outcome.subject}__dev_y"] = outcome.dev_y
        arrays[f"{outcome.subject}__dev_proba"] = outcome.dev_proba
    table = pd.DataFrame(rows)
    np.savez_compressed(out_dir / f"{candidate.id}.npz", **arrays)
    table.to_csv(table_path, index=False)
    (out_dir / f"{candidate.id}.json").write_text(
        json.dumps(
            {
                "candidate": candidate.id,
                "preset": candidate.preset,
                "description": candidate.description,
                "stimulus_window": STIMULUS_WINDOW,
                "n_subjects": len(table),
                "sfreq": dataset.sfreq,
            },
            indent=2,
        )
    )
    return table


def load_predictions(task_id: str, candidate_id: str) -> dict[str, dict[str, np.ndarray]]:
    arrays = np.load(results_dir(task_id) / f"{candidate_id}.npz")
    out: dict[str, dict[str, np.ndarray]] = {}
    for key in arrays.files:
        subject, field = key.split("__", 1)
        out.setdefault(subject, {})[field] = arrays[key]
    return out


def run_many(task_ids: Sequence[str], candidate_ids: Sequence[str], **kwargs) -> None:
    for task_id in task_ids:
        for candidate_id in candidate_ids:
            table = run_candidate(task_id, candidate_id, **kwargs)
            print(
                f"{task_id:>10} {candidate_id:<16} dev={table.dev_bacc.mean():.3f} "
                f"holdout={table.holdout_bacc.mean():.3f} ± {table.holdout_bacc.std():.3f} "
                f"(n={len(table)})",
                flush=True,
            )


def export_for_gpu(task_ids: Sequence[str], preset: str = "erp", out_dir: Path | None = None) -> Path:
    """Write stimulus-window epochs (microvolts, float16) for the GPU script."""
    paths = ProjectPaths.from_env("research", create_dirs=False)
    out_dir = out_dir or paths.artifacts_dir / "research" / "_export"
    out_dir.mkdir(parents=True, exist_ok=True)
    for task_id in task_ids:
        dataset = load_stimulus_dataset(task_id, preset)
        scale = 1e6 if np.nanmax(np.abs(dataset.X)) < 1e-2 else 1.0  # volts -> microvolts
        np.savez(
            out_dir / f"{task_id}.npz",
            X=(dataset.X * scale).astype(np.float16),
            y=dataset.y.astype(np.int8),
            subjects=np.asarray(dataset.subjects, dtype=str),
            trial_indices=dataset.trial_indices,
            sfreq=np.float64(dataset.sfreq),
            channel_names=np.asarray(dataset.channel_names, dtype=str),
            class_ids=np.asarray(dataset.class_ids, dtype=str),
        )
        print(f"exported {task_id}: {dataset.X.shape} -> {out_dir / (task_id + '.npz')}", flush=True)
    for name in ("deep_pooled.py", "kaggle_notebook.ipynb"):
        source = paths.project_dir / "gpu" / name
        (out_dir / name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return out_dir


def import_gpu_results(source: Path) -> list[Path]:
    """Copy ``<task>/<run>.{csv,npz,json}`` files produced by the GPU script into the results tree."""
    copied = []
    for csv_path in Path(source).glob("*/*.csv"):
        task_id = csv_path.parent.name
        for suffix in (".csv", ".npz", ".json"):
            src = csv_path.with_suffix(suffix)
            if src.exists():
                dst = results_dir(task_id) / src.name
                dst.write_bytes(src.read_bytes())
                copied.append(dst)
    return copied
