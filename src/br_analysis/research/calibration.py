"""How much data does a new person need? A calibration curve for pooled Riemann decoding.

For each target subject, a classifier is trained on the other 46 subjects only (each
aligned with its own Euclidean Alignment reference). The target is then scored on its
untouched final 20 % of trials under two kinds of calibration, both drawn only from the
start of the target's session:

* ``align``: the first k trials, **unlabelled**, are used only to compute the EA reference
  (zero-shot: no target label is ever used);
* ``align+labels``: the same k trials are also added, with labels, to the training set.

k runs over 1, 5, 10 and 20 trials, and 10 %, 20 %, 50 % and 80 % of the session.
``none`` is the no-alignment baseline. Everything is done in covariance space (EA of a
covariance is R C R), with the tangent space taken at the identity, which is where EA puts
each subject's mean. This is a fast approximation of the full Riemann pipeline.

    python -m br_analysis.research.calibration c_vs_ip
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .harness import load_stimulus_dataset, results_dir

TRIALS = [1, 5, 10, 20]
FRACTIONS = [0.1, 0.2, 0.5, 0.8]
HOLDOUT = 0.2


def _covariances(X: np.ndarray) -> np.ndarray:
    centered = X - X.mean(axis=-1, keepdims=True)
    return np.einsum("nct,ndt->ncd", centered, centered, optimize=True) / X.shape[-1]


def _inv_sqrt(C: np.ndarray) -> np.ndarray:
    C = C + 1e-6 * np.trace(C) / len(C) * np.eye(len(C))
    w, V = np.linalg.eigh(C)
    return (V / np.sqrt(w)) @ V.T


def _tangent(C: np.ndarray) -> np.ndarray:
    """Upper triangle of logm(C) (off-diagonals x sqrt 2), batched."""
    n = C.shape[-1]
    C = C + 1e-4 * np.trace(C, axis1=1, axis2=2)[:, None, None] / n * np.eye(n)
    w, V = np.linalg.eigh(C)
    L = np.einsum("nij,nj,nkj->nik", V, np.log(w), V, optimize=True)
    iu = np.triu_indices(n)
    weights = np.where(iu[0] == iu[1], 1.0, np.sqrt(2))
    return (L[:, iu[0], iu[1]] * weights).astype(np.float32)


def _align(C: np.ndarray, reference_covs: np.ndarray) -> np.ndarray:
    R = _inv_sqrt(reference_covs.mean(0))
    return np.einsum("ij,njk,kl->nil", R, C, R, optimize=True)


def _lda() -> LinearDiscriminantAnalysis:
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5])


def run(task_id: str) -> pd.DataFrame:
    dataset = load_stimulus_dataset(task_id, "broadband")
    subjects = list(dataset.subject_ids)
    covs, labels = {}, {}
    for s in subjects:
        mask = dataset.subjects == s
        covs[s] = _covariances(dataset.X[mask].astype(np.float64))
        labels[s] = dataset.y[mask].astype(int)
    aligned = {s: _tangent(_align(covs[s], covs[s])) for s in subjects}  # sources: own full-session EA
    raw = {s: _tangent(covs[s]) for s in subjects}

    rows = []
    for target in subjects:
        y = labels[target]
        n = len(y)
        cut = n - int(np.ceil(HOLDOUT * n))
        hold = np.arange(cut, n)
        if len(np.unique(y[hold])) < 2:
            continue
        sources = [s for s in subjects if s != target]
        Xs = np.concatenate([aligned[s] for s in sources])
        ys = np.concatenate([labels[s] for s in sources])
        base = _lda().fit(Xs, ys)

        def score(model, Xh, condition, k):
            proba = model.predict_proba(Xh)[:, list(model.classes_).index(1)]
            rows.append({"subject": target, "condition": condition, "k": k,
                         "bacc": balanced_accuracy_score(y[hold], (proba > 0.5).astype(int)),
                         "auc": roc_auc_score(y[hold], proba)})

        unaligned = _lda().fit(np.concatenate([raw[s] for s in sources]), ys)
        score(unaligned, raw[target][hold], "none", 0)
        for k in TRIALS + [int(f * n) for f in FRACTIONS]:
            k = min(k, cut)
            calib = np.arange(k)
            Xt = _tangent(_align(covs[target], covs[target][calib]))
            score(base, Xt[hold], "align", k)
            if len(np.unique(y[calib])) == 2:
                model = _lda().fit(np.concatenate([Xs, Xt[calib]]), np.concatenate([ys, y[calib]]))
                score(model, Xt[hold], "align+labels", k)
        print(target, flush=True)

    df = pd.DataFrame(rows)
    df["k_label"] = df.apply(lambda r: _k_label(r, df), axis=1)
    out = results_dir(task_id) / "_calibration.csv"
    df.to_csv(out, index=False)
    return df


def _k_label(row, df) -> str:
    return "none" if row.condition == "none" else str(row.k)


def summarise(task_id: str) -> dict:
    df = pd.read_csv(results_dir(task_id) / "_calibration.csv")
    # map each subject's k back to the requested step (trials or fraction)
    steps = []
    for (subject, condition), g in df[df.condition != "none"].groupby(["subject", "condition"], sort=False):
        names = [f"{t} trials" for t in TRIALS] + [f"{int(f * 100)} %" for f in FRACTIONS]
        g = g.reset_index(drop=True)
        if condition == "align+labels" and len(g) < len(names):  # steps without both classes were skipped
            names = names[len(names) - len(g):]
        steps.append(g.assign(step=names[:len(g)]))
    tidy = pd.concat(steps + [df[df.condition == "none"].assign(step="none")])
    zero = tidy[tidy.condition == "none"].set_index("subject")
    summary = {}
    for (condition, step), g in tidy.groupby(["condition", "step"], sort=False):
        g = g.set_index("subject")
        common = g.index.intersection(zero.index)
        diff = g.loc[common, "bacc"] - zero.loc[common, "bacc"]
        summary[f"{condition} | {step}"] = {
            "n": int(len(g)), "bacc": round(float(g.bacc.mean()), 4), "auc": round(float(g.auc.mean()), 4),
            "p_vs_chance": float(wilcoxon(g.bacc - 0.5).pvalue),
            "gain_vs_none": round(float(diff.mean()), 4), "p_vs_none": float(wilcoxon(diff).pvalue) if len(diff) > 5 else None,
        }
    tidy.to_csv(results_dir(task_id) / "_calibration_tidy.csv", index=False)
    return summary


SOURCES = [0, 1, 2, 5, 10, 20, 46]
TARGET_DATA = {"zero-shot": (0.8, False), "20 % labelled": (0.2, True), "80 % labelled": (0.8, True)}


def _repeats(n_sources: int) -> int:
    return 1 if n_sources in (0, 46) else 5 if n_sources <= 5 else 3


def sources_curve(task_id: str, shard: tuple[int, int] = (0, 1)) -> pd.DataFrame | None:
    """The other axis: how many *other* subjects does the pooled model need?

    For each target and each number of source subjects N (random draws, repeated for small
    N), train on those N subjects plus the target's calibration data and score the
    target's final 20 %. The target's calibration is either the development 80 % unlabelled
    (alignment only), or the first 20 % / 80 % of the session with labels. N = 0 is the
    per-subject model (own labelled data only, still in the EA frame).
    """
    dataset = load_stimulus_dataset(task_id, "broadband")
    subjects = list(dataset.subject_ids)
    covs, labels = {}, {}
    for s in subjects:
        mask = dataset.subjects == s
        covs[s] = _covariances(dataset.X[mask].astype(np.float64))
        labels[s] = dataset.y[mask].astype(int)
    aligned = {s: _tangent(_align(covs[s], covs[s])) for s in subjects}
    del dataset

    cache = results_dir(task_id) / "_calibration_sources"
    cache.mkdir(exist_ok=True)

    def one_target(t_index: int, target: str) -> list[dict]:
        done = cache / f"{target}.csv"
        if done.exists():  # resumable: each target is written as soon as it finishes
            return pd.read_csv(done).to_dict("records")
        y = labels[target]
        n = len(y)
        cut = n - int(np.ceil(HOLDOUT * n))
        hold = np.arange(cut, n)
        if len(np.unique(y[hold])) < 2:
            return []
        others = [s for s in subjects if s != target]
        rng = np.random.default_rng(1000 + t_index)
        out = []
        for condition, (fraction, use_labels) in TARGET_DATA.items():
            k = min(int(fraction * n), cut)
            calib = np.arange(k)
            Xt = _tangent(_align(covs[target], covs[target][calib]))
            for n_src in SOURCES:
                if n_src == 0 and not use_labels:
                    continue
                for rep in range(_repeats(n_src)):
                    chosen = list(rng.choice(others, size=n_src, replace=False)) if n_src else []
                    X_parts = [aligned[s] for s in chosen]
                    y_parts = [labels[s] for s in chosen]
                    if use_labels:
                        X_parts.append(Xt[calib])
                        y_parts.append(y[calib])
                    y_train = np.concatenate(y_parts)
                    if len(np.unique(y_train)) < 2:  # e.g. own data only, calibration with one class
                        continue
                    model = _lda().fit(np.concatenate(X_parts), y_train)
                    proba = model.predict_proba(Xt[hold])[:, list(model.classes_).index(1)]
                    out.append({"subject": target, "target_data": condition, "n_sources": n_src, "rep": rep,
                                "bacc": balanced_accuracy_score(y[hold], (proba > 0.5).astype(int)),
                                "auc": roc_auc_score(y[hold], proba)})
        pd.DataFrame(out).to_csv(done, index=False)
        print(target, flush=True)
        return out

    index, count = shard  # run as several processes: --shard 0/3, 1/3, 2/3, then once without
    rows = [one_target(i, s) for i, s in enumerate(subjects) if i % count == index]
    if count > 1:
        return None
    df = pd.DataFrame([r for chunk in rows for r in chunk])
    df.to_csv(results_dir(task_id) / "_calibration_sources.csv", index=False)
    return df


def summarise_sources(task_id: str) -> dict:
    df = pd.read_csv(results_dir(task_id) / "_calibration_sources.csv")
    per_subject = df.groupby(["target_data", "n_sources", "subject"])[["bacc", "auc"]].mean().reset_index()
    summary = {}
    for (condition, n_src), g in per_subject.groupby(["target_data", "n_sources"]):
        summary[f"{condition} | N={n_src}"] = {
            "n": int(len(g)), "bacc": round(float(g.bacc.mean()), 4),
            "sem": round(float(g.bacc.std(ddof=1) / np.sqrt(len(g))), 4), "auc": round(float(g.auc.mean()), 4),
            "p_vs_chance": float(wilcoxon(g.bacc - 0.5).pvalue)}
    per_subject.to_csv(results_dir(task_id) / "_calibration_sources_tidy.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks", nargs="+")
    parser.add_argument("--sources", action="store_true", help="run the number-of-subjects curve instead")
    parser.add_argument("--shard", default="0/1", help="i/n: this process handles targets i, i+n, ...")
    args = parser.parse_args()
    for task in args.tasks:
        if args.sources:
            index, count = map(int, args.shard.split("/"))
            if sources_curve(task, (index, count)) is None:
                continue
            print(json.dumps(summarise_sources(task), indent=1))
        else:
            run(task)
            print(json.dumps(summarise(task), indent=1))


if __name__ == "__main__":
    main()
