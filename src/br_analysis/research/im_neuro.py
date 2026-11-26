"""Neuroscience of the conflict *resolution* (IM vs IP), next to conflict detection (C vs IP).

Not a decoder: these are group-level, interpretable analyses that ask *where* and *when*
the brain differs, and what explains who reports "mixed".

1. **Spectral maps.** Per subject, Cohen's d for each window x band x channel; group
   sign-flip cluster permutation over the joint (window, band, channel) lattice, so each
   contrast is one family-wise test. C vs IP is the positive control.
2. **ERPs.** IM vs IP grand averages in three regions, with a temporal cluster test.
3. **Behaviour.** Serial dependence of the percept (against a block-shuffle null that keeps
   slow drift), drift across the session, eye dominance.
4. **Individual differences.** Across subjects, the mixed-report rate against eye-dominance
   strength, individual alpha frequency, posterior alpha power and conflict decodability.

    python -m br_analysis.research.im_neuro extract    # ~15 s per subject, cached
    python -m br_analysis.research.im_neuro analyse    # results -> artifacts/research/im_neuro/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import welch

from ..config import ProjectPaths
from .im_explore import BANDS
from .im_explore import cache_dir as im_cache_dir

CONGRUENT, IP_CODES, IM_CODES = (1, 2), (3, 4, 5, 6), (7, 8)
IP_RED, IP_GREEN, CFG_A = (3, 4), (5, 6), (3, 5, 7)
WINDOWS = {"pre": (-0.5, 0.0), "w1": (0.0, 0.5), "w2": (0.5, 1.0), "w3": (1.0, 1.5)}
ROIS = {
    "occipital": ["O1", "Oz", "O2", "PO3", "POz", "PO4"],
    "parietal": ["CP1", "CPz", "CP2", "P1", "Pz", "P2"],
    "frontal": ["F1", "Fz", "F2", "FC1", "FCz", "FC2"],
}
POSTERIOR = ["P3", "P1", "Pz", "P2", "P4", "PO3", "POz", "PO4", "O1", "Oz", "O2"]


def _paths() -> ProjectPaths:
    return ProjectPaths.from_env("research", create_dirs=False)


def trait_cache() -> Path:
    out = _paths().artifacts_dir / "_cache" / "im_neuro"
    out.mkdir(parents=True, exist_ok=True)
    return out


def results_dir() -> Path:
    out = _paths().artifacts_dir / "research" / "im_neuro"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ----------------------------------------------------------------------------- extraction


def extract_subject(path: Path, out: Path) -> None:
    import mne

    mne.set_log_level("ERROR")
    epochs = mne.read_epochs(path, preload=False)
    codes = epochs.events[:, 2]
    X = (epochs.get_data() * 1e6).astype(np.float32)
    X -= X.mean(axis=1, keepdims=True)  # common average reference
    sfreq, times = epochs.info["sfreq"], epochs.times

    bp = np.zeros((len(X), len(WINDOWS), X.shape[1], len(BANDS)), dtype=np.float32)
    for w, (a, b) in enumerate(WINDOWS.values()):
        sl = (times >= a - 1e-9) & (times < b - 1e-9)
        freqs, psd = welch(X[..., sl], fs=sfreq, nperseg=int(0.5 * sfreq), axis=-1)
        for k, (lo, hi) in enumerate(BANDS.values()):
            bp[:, w, :, k] = np.log10(psd[..., (freqs >= lo) & (freqs < hi)].mean(-1) + 1e-12)

    # subject-level spectrum (trait): pre-stimulus segment, zero-padded, averaged over trials
    pre = X[..., times < 0]
    freqs, psd = welch(pre, fs=sfreq, nperseg=pre.shape[-1], nfft=4 * int(sfreq), axis=-1)
    keep = (freqs >= 1) & (freqs <= 45)
    np.savez_compressed(out / f"{path.stem}.npz", codes=codes, bp=bp, spec=psd[..., keep].mean(0),
                        spec_freqs=freqs[keep], channels=np.array(epochs.ch_names))


def extract_all(force: bool = False) -> None:
    out = trait_cache()
    for p in sorted(_paths().data_dir.glob("*.fif")):
        if force or not (out / f"{p.stem}.npz").exists():
            extract_subject(p, out)
            print("traits", p.stem, flush=True)


def load_traits() -> dict[str, dict]:
    return {f.stem: dict(np.load(f)) for f in sorted(trait_cache().glob("*.npz"))}


def channel_info(channels):
    import mne

    info = mne.create_info(list(channels), 500.0, "eeg")
    info.set_montage("standard_1005")
    return info


# ----------------------------------------------------------------------------- statistics


def cohen_d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    pooled = np.sqrt(((len(a) - 1) * a.var(0, ddof=1) + (len(b) - 1) * b.var(0, ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean(0) - b.mean(0)) / pooled


def spectral_maps(traits: dict, n_permutations: int = 2000) -> dict:
    """Per contrast: group t over subjects' Cohen's d, and joint cluster test."""
    import mne

    channels = next(iter(traits.values()))["channels"]
    info = channel_info(channels)
    ch_adj, _ = mne.channels.find_ch_adjacency(info, "eeg")
    adjacency = mne.stats.combine_adjacency(len(WINDOWS), len(BANDS), ch_adj)
    out = {}
    for name, pos, neg in (("c_vs_ip", CONGRUENT, IP_CODES), ("im_vs_ip", IM_CODES, IP_CODES)):
        ds = []
        for d in traits.values():
            a, b = d["bp"][np.isin(d["codes"], pos)], d["bp"][np.isin(d["codes"], neg)]
            if len(a) >= 10 and len(b) >= 10:
                ds.append(cohen_d(a.astype(np.float64), b.astype(np.float64)))
        D = np.stack(ds)  # (subjects, windows, channels, bands)
        D = D.transpose(0, 1, 3, 2)  # -> (subjects, windows, bands, channels)
        n = len(D)
        threshold = stats.t.ppf(1 - 0.005, n - 1)  # cluster-forming p < 0.01, two-tailed
        t_obs, clusters, p_clusters, _ = mne.stats.permutation_cluster_1samp_test(
            D.reshape(n, -1), threshold=threshold, n_permutations=n_permutations, adjacency=adjacency,
            tail=0, out_type="mask", seed=0, verbose=False)
        mask = np.zeros(t_obs.shape, bool)
        summary = []
        for c, p in zip(clusters, p_clusters):
            if p < 0.05:
                mask |= c
            idx = np.unravel_index(np.flatnonzero(c), D.shape[1:])
            summary.append({"p": float(p), "size": int(c.sum()), "sign": float(np.sign(t_obs[c].mean())),
                            "windows": sorted({list(WINDOWS)[i] for i in idx[0]}),
                            "bands": sorted({list(BANDS)[i] for i in idx[1]})})
        out[name] = {"t": t_obs.reshape(D.shape[1:]), "mask": mask.reshape(D.shape[1:]),
                     "d_mean": D.mean(0), "n": n, "clusters": sorted(summary, key=lambda s: s["p"])[:8]}
    out["channels"] = channels
    return out


def erp_roi(n_permutations: int = 2000) -> dict:
    """IM vs IP grand-average ERPs per region, with temporal cluster tests (incongruent cache)."""
    import mne

    rows = {roi: ([], []) for roi in ROIS}
    times = None
    for f in sorted(im_cache_dir().glob("*.npz")):
        d = np.load(f)
        times = d["erp_times"] + 0.025
        chans = list(d["channels"])
        for roi, names in ROIS.items():
            sig = d["erp"][:, [chans.index(c) for c in names]].mean(1)
            rows[roi][0].append(sig[d["y"] == 1].mean(0))
            rows[roi][1].append(sig[d["y"] == 0].mean(0))
    out = {"times": times}
    for roi, (im, ip) in rows.items():
        im, ip = np.array(im), np.array(ip)
        diff = im - ip
        thr = stats.t.ppf(1 - 0.025, len(diff) - 1)
        t_obs, clusters, p_clusters, _ = mne.stats.permutation_cluster_1samp_test(
            diff, threshold=thr, n_permutations=n_permutations, tail=0, out_type="indices", seed=0, verbose=False)
        clusters = [np.isin(np.arange(len(times)), np.asarray(c[0])) for c in clusters]
        sig = np.zeros(len(times), bool)
        for c, p in zip(clusters, p_clusters):
            if p < 0.05:
                sig |= c
        out[roi] = {"im": im, "ip": ip, "sig": sig,
                    "clusters": [(float(times[c].min()), float(times[c].max()), float(p)) for c, p in zip(clusters, p_clusters)]}
    return out


# ----------------------------------------------------------------------------- behaviour


def lagged_effect(y: np.ndarray, lag: int) -> float:
    prev, cur = y[:-lag], y[lag:]
    if prev.min() == prev.max():
        return np.nan
    return cur[prev == 1].mean() - cur[prev == 0].mean()


def block_shuffle(y: np.ndarray, rng, block: int = 20) -> np.ndarray:
    out = y.copy()
    for start in range(0, len(y), block):
        seg = out[start:start + block]
        rng.shuffle(seg)
    return out


def behaviour_tables(traits: dict, max_lag: int = 15, n_null: int = 200) -> dict:
    rng = np.random.default_rng(0)
    subj_rows, lag_rows, drift_rows = [], [], []
    for subject, d in traits.items():
        codes = d["codes"]
        inc = codes[np.isin(codes, IP_CODES + IM_CODES)]
        y = np.isin(inc, IM_CODES).astype(int)
        ip = codes[np.isin(codes, IP_CODES)]
        ip_cfg_a = np.isin(ip, CFG_A)
        red = np.isin(ip, IP_RED)
        p_red_a = red[ip_cfg_a].mean() if ip_cfg_a.any() else np.nan
        p_red_b = red[~ip_cfg_a].mean() if (~ip_cfg_a).any() else np.nan
        cfg_inc = np.isin(inc, CFG_A)
        subj_rows.append({
            "subject": subject, "n_inc": len(y), "mixed_rate": y.mean(),
            "eye_dominance": abs(p_red_a - p_red_b), "colour_bias": abs((p_red_a + p_red_b) / 2 - 0.5),
            "mixed_rate_cfg_a": y[cfg_inc].mean(), "mixed_rate_cfg_b": y[~cfg_inc].mean(),
            "serial_lag1": lagged_effect(y, 1),
        })
        for lag in range(1, max_lag + 1):
            null = [lagged_effect(block_shuffle(y, rng), lag) for _ in range(n_null)]
            lag_rows.append({"subject": subject, "lag": lag, "observed": lagged_effect(y, lag),
                             "null": np.nanmean(null)})
        for k, chunk in enumerate(np.array_split(y, 10)):
            drift_rows.append({"subject": subject, "decile": k + 1, "mixed_rate": chunk.mean()})
    return {"subjects": pd.DataFrame(subj_rows), "lags": pd.DataFrame(lag_rows), "drift": pd.DataFrame(drift_rows)}


# ----------------------------------------------------------------------------- balance model

REPORT_CODES = {"A": {"red": 3, "green": 5, "mixed": 7}, "B": {"red": 4, "green": 6, "mixed": 8}}


def _ordered_probs(s: float, theta: float) -> np.ndarray:
    """P(green), P(mixed), P(red) when a latent balance s + N(0, 1) is cut at -theta and +theta."""
    from scipy.stats import norm

    lo, hi = norm.cdf(-theta - s), norm.cdf(theta - s)
    return np.clip(np.array([lo, hi - lo, 1 - hi]), 1e-9, 1)


def balance_model(traits: dict) -> pd.DataFrame:
    """Per subject: one latent interocular balance per configuration, one shared 'mixed' band.

    The two configurations swap which eye sees which colour. If the eye advantage and the
    colour advantage add, one configuration is more balanced than the other, and the model
    predicts more "mixed" reports there. Fitted by maximum likelihood (3 parameters) and
    compared with the saturated multinomial (4 parameters): likelihood ratio, 1 df.
    """
    from scipy.optimize import minimize

    rows = []
    for subject, d in traits.items():
        counts = {c: np.array([np.sum(d["codes"] == REPORT_CODES[c][k]) for k in ("green", "mixed", "red")])
                  for c in REPORT_CODES}

        def nll(params):
            s_a, s_b, log_theta = params
            theta = np.exp(log_theta)
            return -sum((counts[c] * np.log(_ordered_probs(s, theta))).sum()
                        for c, s in (("A", s_a), ("B", s_b)))

        best = min((minimize(nll, x0, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 4000})
                    for x0 in ([0, 0, 0], [1, -1, 0], [-1, 1, 0], [0.5, 0.5, -1])), key=lambda r: r.fun)
        saturated = -sum((n * np.log(np.clip(n / n.sum(), 1e-9, 1))).sum() for n in counts.values())
        s_a, s_b, log_theta = best.x
        lr = max(0.0, 2 * (best.fun - saturated))
        rate = {c: counts[c][1] / counts[c].sum() for c in counts}
        pred = {c: _ordered_probs(s, np.exp(log_theta))[1] for c, s in (("A", s_a), ("B", s_b))}
        rows.append({"subject": subject, "s_a": s_a, "s_b": s_b, "theta": float(np.exp(log_theta)),
                     "eye_advantage": (s_a - s_b) / 2, "colour_advantage": (s_a + s_b) / 2,
                     "mixed_a": rate["A"], "mixed_b": rate["B"], "pred_mixed_a": pred["A"], "pred_mixed_b": pred["B"],
                     "lr": lr, "p_misfit": float(stats.chi2.sf(lr, 1)),
                     "pure_imbalance_a": abs(counts["A"][2] / max(1, counts["A"][0] + counts["A"][2]) - 0.5),
                     "pure_imbalance_b": abs(counts["B"][2] / max(1, counts["B"][0] + counts["B"][2]) - 0.5),
                     "n_a": int(counts["A"].sum()), "n_b": int(counts["B"].sum())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- traits


def alpha_traits(traits: dict) -> pd.DataFrame:
    """Individual alpha frequency (centre of gravity above the 1/f fit) and posterior alpha power."""
    rows = []
    for subject, d in traits.items():
        chans = list(d["channels"])
        f = d["spec_freqs"]
        spec = d["spec"][[chans.index(c) for c in POSTERIOR]].mean(0)
        fit = (f >= 2) & (f <= 40) & ~((f >= 6) & (f <= 15))
        slope, intercept = np.polyfit(np.log10(f[fit]), np.log10(spec[fit]), 1)
        resid = np.log10(spec) - (intercept + slope * np.log10(f))
        band = (f >= 7) & (f <= 14)
        w = np.clip(resid[band], 0, None)
        iaf = float((f[band] * w).sum() / w.sum()) if w.sum() > 0 else np.nan
        rows.append({"subject": subject, "iaf": iaf, "alpha_peak_height": float(resid[band].max()),
                     "posterior_alpha": float(np.log10(spec[(f >= 8) & (f <= 13)].mean())), "aperiodic_slope": slope})
    return pd.DataFrame(rows)


def individual_differences(subjects: pd.DataFrame, alpha: pd.DataFrame, n_perm: int = 10000) -> pd.DataFrame:
    paths = _paths()
    dec = pd.read_csv(paths.artifacts_dir / "research" / "c_vs_ip" / "scr_ens_finalists.csv")[["subject", "holdout_auc"]]
    dec = dec.rename(columns={"holdout_auc": "conflict_decodability"})
    table = subjects.merge(alpha, on="subject").merge(dec, on="subject", how="left")
    rng = np.random.default_rng(0)
    rows = []
    for predictor in ("eye_dominance", "iaf", "alpha_peak_height", "posterior_alpha", "conflict_decodability", "serial_lag1"):
        sub = table[["mixed_rate", predictor]].dropna()
        rho = stats.spearmanr(sub["mixed_rate"], sub[predictor]).statistic
        x = sub[predictor].rank().to_numpy()
        yv = sub["mixed_rate"].rank().to_numpy()
        null = np.array([np.corrcoef(x, rng.permutation(yv))[0, 1] for _ in range(n_perm)])
        rows.append({"predictor": predictor, "n": len(sub), "rho": rho,
                     "p_perm": (np.sum(np.abs(null) >= abs(rho)) + 1) / (n_perm + 1)})
    res = pd.DataFrame(rows)
    order = np.argsort(res["p_perm"].to_numpy())
    holm = np.empty(len(res))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(res) - rank) * res["p_perm"].iloc[i]))
        holm[i] = running
    res["p_holm"] = holm
    table.to_csv(results_dir() / "subject_table.csv", index=False)
    return res


# ----------------------------------------------------------------------------- driver


def analyse() -> None:
    traits = load_traits()
    out = results_dir()
    maps = spectral_maps(traits)
    np.savez_compressed(out / "spectral_maps.npz", channels=maps["channels"],
                        **{f"{c}_{k}": maps[c][k] for c in ("c_vs_ip", "im_vs_ip") for k in ("t", "mask", "d_mean")})
    summary = {c: {"n": maps[c]["n"], "clusters": maps[c]["clusters"]} for c in ("c_vs_ip", "im_vs_ip")}

    erp = erp_roi()
    np.savez_compressed(out / "erp_roi.npz", times=erp["times"],
                        **{f"{r}_{k}": erp[r][k] for r in ROIS for k in ("im", "ip", "sig")})
    summary["erp_clusters"] = {r: erp[r]["clusters"] for r in ROIS}

    beh = behaviour_tables(traits)
    for name, df in beh.items():
        df.to_csv(out / f"behaviour_{name}.csv", index=False)
    lags = beh["lags"].assign(excess=lambda d: d.observed - d.null)
    summary["serial"] = {int(k): {"excess": float(g.excess.mean()),
                                  "p": float(stats.wilcoxon(g.excess.dropna()).pvalue)}
                         for k, g in lags.groupby("lag")}
    s = beh["subjects"]
    summary["config"] = {"a": float(s.mixed_rate_cfg_a.mean()), "b": float(s.mixed_rate_cfg_b.mean()),
                         "p": float(stats.wilcoxon(s.mixed_rate_cfg_a - s.mixed_rate_cfg_b).pvalue)}
    drift = beh["drift"].groupby("decile").mixed_rate.mean()
    first, last = beh["drift"].query("decile <= 2"), beh["drift"].query("decile >= 9")
    summary["drift"] = {"by_decile": drift.round(4).tolist(),
                        "p_first_vs_last": float(stats.wilcoxon(
                            first.groupby("subject").mixed_rate.mean() - last.groupby("subject").mixed_rate.mean()).pvalue)}

    bal = balance_model(traits)
    bal.to_csv(out / "balance_model.csv", index=False)
    diff_obs = bal.mixed_a - bal.mixed_b
    diff_pred = bal.pred_mixed_a - bal.pred_mixed_b
    imbalance = (bal.s_a.abs() - bal.s_b.abs())  # > 0: configuration A less balanced
    summary["balance"] = {
        "total_lr": float(bal.lr.sum()), "df": len(bal),
        "p_global_misfit": float(stats.chi2.sf(bal.lr.sum(), len(bal))),
        "subjects_misfit_p05": int((bal.p_misfit < 0.05).sum()),
        "r_pred_vs_obs_config_diff": float(stats.pearsonr(diff_pred, diff_obs).statistic),
        "rho_imbalance_vs_mixed_diff": float(stats.spearmanr(imbalance, diff_obs).statistic),
        # non-circular version: imbalance from the pure reports only (disjoint trials from the mixed ones)
        "rho_pure_imbalance_vs_mixed_diff": float(stats.spearmanr(bal.pure_imbalance_a - bal.pure_imbalance_b, diff_obs).statistic),
        "p_pure_imbalance_vs_mixed_diff": float(stats.spearmanr(bal.pure_imbalance_a - bal.pure_imbalance_b, diff_obs).pvalue),
        "frac_subjects_direction_ok": float(np.mean(np.sign(bal.pure_imbalance_a - bal.pure_imbalance_b) == -np.sign(diff_obs))),
        "mean_abs_config_diff": float(diff_obs.abs().mean()),
        "rho_theta_vs_mixed_rate": float(stats.spearmanr(bal.theta, s.set_index("subject").loc[bal.subject, "mixed_rate"]).statistic),
    }

    ind = individual_differences(s, alpha_traits(traits))
    ind.to_csv(out / "individual_differences.csv", index=False)
    summary["individual"] = ind.round(4).to_dict(orient="records")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=1, default=float))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["extract", "analyse"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    extract_all(args.force) if args.command == "extract" else analyse()


if __name__ == "__main__":
    main()
