"""Cheap, broad exploration of IM vs IP (mixed vs single-colour percept under conflict).

Features are extracted once per incongruent trial straight from the FIF epochs
(-1.24 to 2.0 s, 500 Hz) and cached; every probe is then a pooled, class-balanced
logistic regression that takes seconds. Selection never touches the holdout:

* per subject, incongruent trials in chronological order; holdout = last 20%;
* probes train on the first 60% of every subject (pooled) and are scored on
  trials 60-80% (validation), as mean per-subject AUC with a Wilcoxon test;
* features are z-scored per subject with statistics from that subject's
  development trials only (unsupervised, like Euclidean alignment).

Windows: ``pre`` (-0.5-0 s) and ``post`` (1.5-2.0 s) exist only to see whether
widening the window would be worth it; the target is the stimulus (0-1.5 s).

    python -m br_analysis.research.im_explore features
    python -m br_analysis.research.im_explore probe
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ..config import ProjectPaths

INCONGRUENT = (3, 4, 5, 6, 7, 8)
MIXED = (7, 8)
CONFIG_B = (4, 6, 8)  # codes come in two eye/colour configurations: A = (3,5,7), B = (4,6,8)
WINDOWS = {"pre": (-0.5, 0.0), "w1": (0.0, 0.5), "w2": (0.5, 1.0), "w3": (1.0, 1.5), "post": (1.5, 2.0)}
STIM_WINDOWS = ("w1", "w2", "w3")
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta1": (13, 20), "beta2": (20, 30), "gamma": (30, 45), "hgamma": (55, 95)}
TRAIN_END, DEV_END = 0.6, 0.8


def cache_dir() -> Path:
    paths = ProjectPaths.from_env("research", create_dirs=False)
    out = paths.artifacts_dir / "_cache" / "im_features"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ----------------------------------------------------------------------------- features


def behaviour(codes: np.ndarray) -> dict[str, np.ndarray]:
    """Per incongruent trial: label, stimulus configuration and response history."""
    idx = np.flatnonzero(np.isin(codes, INCONGRUENT))
    y = np.isin(codes[idx], MIXED).astype(int)
    cfg = np.isin(codes[idx], CONFIG_B).astype(int)
    prev_any_mixed = np.r_[0, np.isin(codes[:-1], MIXED)][idx].astype(int)
    prev_inc_mixed = np.r_[0, y[:-1]]
    prev_cfg_same = np.r_[0, (cfg[1:] == cfg[:-1]).astype(int)]
    return {"index": idx, "y": y, "cfg": cfg, "prev_any_mixed": prev_any_mixed,
            "prev_inc_mixed": prev_inc_mixed, "prev_cfg_same": prev_cfg_same}


def _cov_oas(X: np.ndarray) -> np.ndarray:
    from pyriemann.estimation import Covariances

    return Covariances(estimator="oas").transform(X.astype(np.float64))


def extract_subject(path: Path, out: Path) -> str:
    import mne

    mne.set_log_level("ERROR")
    epochs = mne.read_epochs(path, preload=False)
    codes = epochs.events[:, 2]
    beh = behaviour(codes)
    X = (epochs[beh["index"]].get_data() * 1e6).astype(np.float32)  # µV, incongruent trials only
    X -= X.mean(axis=1, keepdims=True)  # common average reference
    sfreq, times = epochs.info["sfreq"], epochs.times

    def sl(window):
        return (times >= window[0] - 1e-9) & (times < window[1] - 1e-9)

    # band power (Welch on each 0.5 s window, 2 Hz resolution)
    bp = np.zeros((len(X), len(WINDOWS), X.shape[1], len(BANDS)), dtype=np.float32)
    for w, window in enumerate(WINDOWS.values()):
        freqs, psd = welch(X[..., sl(window)], fs=sfreq, nperseg=int(0.5 * sfreq), axis=-1)
        for b, (lo, hi) in enumerate(BANDS.values()):
            bp[:, w, :, b] = np.log10(psd[..., (freqs >= lo) & (freqs < hi)].mean(-1) + 1e-12)

    # ERP: 0.3-20 Hz, 50 ms bins from -0.5 to 2.0 s
    erp_x = sosfiltfilt(butter(4, [0.3, 20], btype="band", fs=sfreq, output="sos"), X, axis=-1)
    edges = np.arange(-0.5, 2.0001, 0.05)
    erp = np.stack([erp_x[..., sl((a, b))].mean(-1) for a, b in zip(edges[:-1], edges[1:])], axis=-1).astype(np.float32)

    # covariance per window on 1-40 Hz data (for tangent-space features)
    cov_x = sosfiltfilt(butter(4, [1, 40], btype="band", fs=sfreq, output="sos"), X, axis=-1)
    covs = np.stack([_cov_oas(cov_x[..., sl(window)]) for window in WINDOWS.values()], axis=1).astype(np.float32)

    np.savez_compressed(out / f"{path.stem}.npz", bp=bp, erp=erp, erp_times=edges[:-1], covs=covs,
                        channels=np.array(epochs.ch_names), **beh)
    return path.stem


def extract_all(n_jobs: int = 3, force: bool = False) -> None:
    paths = ProjectPaths.from_env("research", create_dirs=False)
    out = cache_dir()
    todo = [p for p in sorted(paths.data_dir.glob("*.fif")) if force or not (out / f"{p.stem}.npz").exists()]
    for name in Parallel(n_jobs=n_jobs)(delayed(extract_subject)(p, out) for p in todo):
        print("features", name, flush=True)


# ----------------------------------------------------------------------------- probes


def load_features() -> dict[str, dict[str, np.ndarray]]:
    out = {}
    for f in sorted(cache_dir().glob("*.npz")):
        d = dict(np.load(f))
        n = len(d["y"])
        cut_tr, cut_dev = int(np.floor(TRAIN_END * n)), int(np.floor(DEV_END * n))
        parts = (slice(0, cut_tr), slice(cut_tr, cut_dev), slice(cut_dev, n))
        if any(len(np.unique(d["y"][p])) < 2 for p in parts):
            continue
        d["parts"] = parts
        out[f.stem] = d
    return out


def tangent_features(d: dict, windows) -> np.ndarray:
    """Tangent space of each window's covariances at the subject's development Riemannian mean."""
    from pyriemann.utils.mean import mean_riemann
    from pyriemann.utils.tangentspace import tangent_space

    dev = slice(0, d["parts"][1].stop)
    names = list(WINDOWS)
    blocks = []
    for w in windows:
        c = d["covs"][:, names.index(w)].astype(np.float64)
        blocks.append(tangent_space(c, mean_riemann(c[dev])))
    return np.concatenate(blocks, axis=1)


def behaviour_matrix(d: dict) -> np.ndarray:
    return np.column_stack([d["cfg"], d["prev_inc_mixed"], d["prev_any_mixed"], d["prev_cfg_same"],
                            d["cfg"] * d["prev_inc_mixed"]]).astype(float)


def zscore_dev(F: np.ndarray, d: dict) -> np.ndarray:
    dev = slice(0, d["parts"][1].stop)
    mu, sd = F[dev].mean(0), F[dev].std(0) + 1e-6
    return (F - mu) / sd


def probe(data: dict, featurize, *, C: float = 0.01, subjects=None, extra=None, part: int = 1) -> dict:
    """Pooled logistic regression; returns mean per-subject AUC on ``part`` (1 = validation)."""
    subjects = subjects or list(data)
    feats = {s: zscore_dev(featurize(data[s]), data[s]) for s in subjects}
    if extra is not None:
        feats = {s: np.column_stack([feats[s], extra(data[s])]) for s in subjects}
    # validation probes train on trials 0-60%; the final holdout check (part=2) trains on 0-80%
    train = {s: data[s]["parts"][0] if part == 1 else slice(0, data[s]["parts"][1].stop) for s in subjects}
    X_tr = np.concatenate([feats[s][train[s]] for s in subjects])
    y_tr = np.concatenate([data[s]["y"][train[s]] for s in subjects])
    model = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(X_tr, y_tr)
    aucs, baccs = {}, {}
    for s in subjects:
        p = data[s]["parts"][part]
        proba = model.predict_proba(feats[s][p])[:, 1]
        y = data[s]["y"][p]
        aucs[s] = roc_auc_score(y, proba)
        # balanced accuracy at the subject's own training prior (threshold-free of pooled prior)
        thr = np.quantile(proba, 1 - y.mean()) if 0 < y.mean() < 1 else 0.5
        pred = (proba > thr).astype(int)
        baccs[s] = np.mean([np.mean(pred[y == c] == c) for c in (0, 1)])
    a = np.array(list(aucs.values()))
    return {"auc": a.mean(), "auc_sem": a.std(ddof=1) / np.sqrt(len(a)),
            "p": wilcoxon(a - 0.5, alternative="greater").pvalue, "aucs": aucs, "bacc": np.mean(list(baccs.values()))}


def probe_suite(data: dict) -> pd.DataFrame:
    names = list(WINDOWS)
    band_names = list(BANDS)
    def bp(windows, bands=None, chans=None):
        def f(d):
            x = d["bp"][:, [names.index(w) for w in windows]]
            if bands:
                x = x[..., [band_names.index(b) for b in bands]]
            if chans is not None:
                x = x[:, :, chans]
            return x.reshape(len(x), -1)
        return f

    def erp(t0, t1):
        def f(d):
            m = (d["erp_times"] >= t0 - 1e-9) & (d["erp_times"] < t1 - 1e-9)
            return d["erp"][..., m].reshape(len(d["erp"]), -1)
        return f

    first = next(iter(data.values()))
    chans = [str(c) for c in first["channels"]]
    posterior = [i for i, c in enumerate(chans) if c.startswith(("P", "O"))]
    frontal = [i for i, c in enumerate(chans) if c.startswith(("Fp", "AF"))]

    probes = {
        # ---- stimulus window (the target)
        "bp_stim_all": (bp(STIM_WINDOWS), 0.01),
        **{f"bp_{w}": (bp([w]), 0.01) for w in STIM_WINDOWS},
        **{f"bp_stim_{b}": (bp(STIM_WINDOWS, [b]), 0.01) for b in band_names},
        "bp_stim_alpha_post": (bp(STIM_WINDOWS, ["alpha"], posterior), 0.05),
        "bp_stim_hgamma_frontal": (bp(STIM_WINDOWS, ["hgamma"], frontal), 0.05),
        "erp_0-1.5": (erp(0, 1.5), 0.001),
        "erp_0.1-0.5": (erp(0.1, 0.5), 0.005),
        "erp_0.5-1.5": (erp(0.5, 1.5), 0.002),
        **{f"ts_{w}": ((lambda w: lambda d: tangent_features(d, [w]))(w), 0.005) for w in STIM_WINDOWS},
        "ts_stim": (lambda d: tangent_features(d, STIM_WINDOWS), 0.002),
        # ---- window-expansion checks (not the target)
        "x_bp_pre": (bp(["pre"]), 0.01),
        "x_bp_post": (bp(["post"]), 0.01),
        "x_erp_pre": (erp(-0.5, 0), 0.005),
        "x_erp_post": (erp(1.5, 2.0), 0.005),
        "x_ts_pre": (lambda d: tangent_features(d, ["pre"]), 0.005),
        "x_ts_post": (lambda d: tangent_features(d, ["post"]), 0.005),
    }
    rows = []
    beh = probe(data, behaviour_matrix, C=1.0)
    rows.append({"probe": "behaviour_only", "auc": beh["auc"], "sem": beh["auc_sem"], "p": beh["p"], "bacc": beh["bacc"]})
    print(f"{'behaviour_only':<26} auc={beh['auc']:.3f}±{beh['auc_sem']:.3f} p={beh['p']:.2g}", flush=True)
    for name, (featurize, C) in probes.items():
        r = probe(data, featurize, C=C)
        h = probe(data, featurize, C=C, extra=behaviour_matrix)
        rows.append({"probe": name, "auc": r["auc"], "sem": r["auc_sem"], "p": r["p"], "bacc": r["bacc"],
                     "auc_with_behaviour": h["auc"], "gain_over_behaviour": h["auc"] - beh["auc"],
                     "p_gain": wilcoxon(np.array(list(h["aucs"].values())) - np.array(list(beh["aucs"].values())), alternative="greater").pvalue})
        print(f"{name:<26} auc={r['auc']:.3f}±{r['auc_sem']:.3f} p={r['p']:.2g} | +behaviour {h['auc']:.3f} (gain {h['auc'] - beh['auc']:+.3f}, p={rows[-1]['p_gain']:.2g})", flush=True)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m br_analysis.research.im_explore")
    parser.add_argument("command", choices=["features", "probe"])
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args(argv)
    if args.command == "features":
        extract_all(args.jobs)
    else:
        data = load_features()
        print(f"{len(data)} subjects", flush=True)
        table = probe_suite(data)
        paths = ProjectPaths.from_env("research", create_dirs=False)
        out = paths.artifacts_dir / "research" / "im_vs_ip" / "_explore_probes.csv"
        table.to_csv(out, index=False)
        print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
