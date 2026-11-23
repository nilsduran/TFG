"""Pooled multi-subject training with Euclidean Alignment and target calibration.

For each target subject the model is trained on

* every *other* subject's trials (all of them; they never touch the target's
  evaluation), each aligned with its own Euclidean Alignment reference, plus
* the target's own training trials, aligned with a reference estimated
  without labels from the target's development period only.

The target's final 20% is never used for alignment, selection or training.
This is the "pooled + EA + calibration" setting that Junqueira et al. (2024)
found to beat both per-subject and zero-shot transfer.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.linalg import fractional_matrix_power
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from ..splits import SplitPlan, development_splits
from .harness import load_stimulus_dataset, results_dir
from .pipelines import CANDIDATES
from .stats import balanced_accuracy, permutation_null


def ea_reference(X: np.ndarray) -> np.ndarray:
    """Inverse square root of the mean trial covariance (Euclidean Alignment)."""
    centered = X - X.mean(axis=-1, keepdims=True)
    covariances = np.einsum("nct,ndt->ncd", centered, centered) / X.shape[-1]
    mean_cov = covariances.mean(axis=0)
    mean_cov += 1e-6 * np.trace(mean_cov) / len(mean_cov) * np.eye(len(mean_cov))
    return np.real(fractional_matrix_power(mean_cov, -0.5)).astype(X.dtype)


def ea_align(X: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.einsum("cd,ndt->nct", reference, X)


def _fit_predict(estimator, X_train, y_train, X_test):
    model = clone(estimator).fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    aligned = np.zeros((len(X_test), 2))
    aligned[:, np.asarray(model.classes_, dtype=int)] = proba
    return aligned


def split_stateless_head(estimator):
    """Split off a leading per-trial covariance step so it can be computed once per trial.

    Pooled training otherwise copies ~13k raw epochs per target and per worker, which
    exhausts memory; covariance matrices are ~3x smaller and need computing only once.
    """
    if isinstance(estimator, Pipeline) and type(estimator.steps[0][1]).__name__ == "Covariances":
        return estimator.steps[0][1], Pipeline(estimator.steps[1:])
    return None, estimator


def _evaluate_target(estimator, X_sources, y_sources, X_target, y_target, target_repeat, plan, head=None):
    splits, holdout = development_splits(y_target, plan)
    if not splits or len(np.unique(y_target[holdout])) < 2:
        return None
    n_dev = len(y_target) - len(holdout)
    reference = ea_reference(X_target[:n_dev])  # unsupervised, development trials only
    X_aligned = ea_align(X_target, reference)
    if head is not None:
        X_aligned = head.transform(X_aligned)

    def train_and_predict(train_idx, test_idx):
        X_train = np.concatenate([X_sources] + [X_aligned[train_idx]] * target_repeat)
        y_train = np.concatenate([y_sources] + [y_target[train_idx]] * target_repeat)
        return _fit_predict(estimator, X_train, y_train, X_aligned[test_idx])

    dev_proba = np.full((n_dev, 2), np.nan)
    with threadpool_limits(1):
        for train, test in splits:
            dev_proba[test] = train_and_predict(train, test)
        holdout_proba = train_and_predict(np.arange(n_dev), holdout)
    covered = ~np.isnan(dev_proba[:, 0])
    return {
        "dev_bacc": balanced_accuracy(y_target[:n_dev][covered], dev_proba[covered].argmax(1), 2),
        "holdout_bacc": balanced_accuracy(y_target[holdout], holdout_proba.argmax(1), 2),
        "holdout_auc": float(roc_auc_score(y_target[holdout], holdout_proba[:, 1])),
        "n_dev": n_dev,
        "n_holdout": len(holdout),
        "holdout_y": y_target[holdout],
        "holdout_proba": holdout_proba,
        "dev_y": y_target[:n_dev],
        "dev_proba": dev_proba,
    }


def run_pooled(
    task_id: str,
    candidate_id: str,
    *,
    target_repeat: int = 1,
    n_jobs: int = 4,
    force: bool = False,
    n_permutations: int = 1000,
    plan: SplitPlan = SplitPlan(),
) -> pd.DataFrame:
    candidate = CANDIDATES[candidate_id]
    run_id = f"pooled_ea_{candidate_id}_x{target_repeat}"
    out_dir = results_dir(task_id)
    table_path = out_dir / f"{run_id}.csv"
    if table_path.exists() and not force:
        return pd.read_csv(table_path)

    dataset = load_stimulus_dataset(task_id, candidate.preset)
    X_all = dataset.X.astype(np.float32)
    y_all = dataset.y.astype(int)
    subjects = dataset.subject_ids
    aligned = {}
    for subject in subjects:
        mask = dataset.subjects == subject
        aligned[subject] = ea_align(X_all[mask], ea_reference(X_all[mask]))
    head, estimator = split_stateless_head(candidate.build(dataset.sfreq))
    if head is not None:
        aligned = {s: head.transform(X).astype(np.float32) for s, X in aligned.items()}

    def job(target):
        sources = [s for s in subjects if s != target]
        X_sources = np.concatenate([aligned[s] for s in sources])
        y_sources = np.concatenate([y_all[dataset.subjects == s] for s in sources])
        mask = dataset.subjects == target
        return target, _evaluate_target(
            estimator, X_sources, y_sources, X_all[mask], y_all[mask], target_repeat, plan, head
        )

    results = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(job)(s) for s in subjects)
    rows, arrays = [], {}
    for index, (subject, outcome) in enumerate(results):
        if outcome is None:
            continue
        null = permutation_null(
            outcome["holdout_y"], outcome["holdout_proba"].argmax(1), 2, n_permutations, seed=index
        )
        rows.append(
            {
                "subject": subject,
                "dev_bacc": outcome["dev_bacc"],
                "holdout_bacc": outcome["holdout_bacc"],
                "holdout_auc": outcome["holdout_auc"],
                "holdout_perm_p": float(np.mean(null >= null[0])),
                "n_dev": outcome["n_dev"],
                "n_holdout": outcome["n_holdout"],
            }
        )
        arrays[f"{subject}__null"] = null
        for key in ("holdout_y", "holdout_proba", "dev_y", "dev_proba"):
            arrays[f"{subject}__{key}"] = outcome[key]
    table = pd.DataFrame(rows)
    np.savez_compressed(out_dir / f"{run_id}.npz", **arrays)
    table.to_csv(table_path, index=False)
    (out_dir / f"{run_id}.json").write_text(
        json.dumps(
            {
                "candidate": run_id,
                "base_candidate": candidate_id,
                "preset": candidate.preset,
                "description": f"Pooled EA training of {candidate_id}; target trials repeated x{target_repeat}",
                "n_subjects": len(table),
            },
            indent=2,
        )
    )
    return table
