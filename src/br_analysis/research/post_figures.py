"""Figures for the further analyses, in one visual system (light and dark variants).

    python -m br_analysis.research.post_figures      # -> docs/images/post_tfg/*.png

Colour roles (validated categorical slots 1-3, diverging blue <-> red around a neutral grey):
conflict detection (C vs IP) = blue, conflict resolution (IM vs IP) = orange,
behaviour = aqua. Everything else is ink and hairlines.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from ..config import ProjectPaths  # noqa: E402

THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7", "mid": "#f0efec",
        "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "red": "#e34948",
        "blue_dark": "#104281", "red_dark": "#a32a2a", "dot_alpha": 0.3,
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835", "mid": "#383835",
        "blue": "#3987e5", "orange": "#d95926", "aqua": "#199e70", "red": "#e66767",
        "blue_dark": "#86b6ef", "red_dark": "#f2a3a3", "dot_alpha": 0.5,
    },
}
T: dict = THEMES["light"]


def out_dir() -> Path:
    paths = ProjectPaths.from_env("research", create_dirs=False)
    out = paths.project_dir / "docs" / "images" / "post_tfg"
    out.mkdir(parents=True, exist_ok=True)
    return out


def research_dir() -> Path:
    return ProjectPaths.from_env("research", create_dirs=False).artifacts_dir / "research"


@contextmanager
def theme(name: str):
    global T
    T = THEMES[name]
    rc = {
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
        "figure.facecolor": T["surface"], "axes.facecolor": T["surface"], "savefig.facecolor": T["surface"],
        "text.color": T["ink"], "axes.labelcolor": T["ink2"], "axes.edgecolor": T["axis"],
        "xtick.color": T["muted"], "ytick.color": T["muted"], "xtick.labelcolor": T["ink2"], "ytick.labelcolor": T["ink2"],
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": T["grid"], "grid.linewidth": 0.8,
        "axes.axisbelow": True, "xtick.major.size": 0, "ytick.major.size": 0,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.titlepad": 10,
        "axes.labelsize": 9.5, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.frameon": False,
        "legend.fontsize": 9, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    }
    with plt.rc_context(rc):
        yield T


def diverging() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("bmr", [T["blue_dark"], T["blue"], T["mid"], T["red"], T["red_dark"]])


# Poster mode: no in-figure headline, exact physical size (inches = CSS px / 96 on an A3 page), 300 dpi.
POSTER: dict | None = None
POSTER_LAYOUT = {  # name -> (size in inches, top margin)
    "fig1": ((10.7, 3.15), 0.88), "fig2": ((10.7, 3.65), 0.90), "fig3": ((6.55, 3.1), 0.88),
}


def size(name: str, default):
    return POSTER_LAYOUT[name][0] if POSTER is not None else default


def fs(points: float) -> float:
    return points * 0.78 if POSTER is not None else points


def top(name: str, default: float) -> float:
    return POSTER_LAYOUT[name][1] if POSTER is not None else default


def headline(fig, title: str, subtitle: str, y: float = 0.985) -> None:
    if POSTER is not None:
        return
    fig.text(0.012, y, title, fontsize=14, fontweight="semibold", color=T["ink"], va="top")
    fig.text(0.012, y - 0.055, subtitle, fontsize=10, color=T["ink2"], va="top")


def save(fig, name: str, variant: str) -> Path:
    if POSTER is not None:
        path = POSTER["dir"] / f"poster_{name}.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        return path
    path = out_dir() / f"{name}{'' if variant == 'light' else '_dark'}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def bootstrap_ci(values: np.ndarray, n: int = 5000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = values[~np.isnan(values)]
    means = rng.choice(values, (n, len(values))).mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def ringed_dot(ax, x, y, color, size=46, z=5):
    ax.scatter(x, y, s=size, color=color, edgecolor=T["surface"], linewidth=2 if POSTER is None else 1.1, zorder=z)


# ----------------------------------------------------------------------------- figure 1


LADDER = {
    "c_vs_ip": [
        ("riemann", "Riemann\nper subject"),
        ("pooled_ea_riemann_x1", "Riemann\npooled + EA"),
        ("gpu_eegnet_ea_aug_ft10", "EEGNet\npooled"),
        ("scr_aux3", "+ IM as\n3rd class"),
        ("scr_ens_finalists", "Ensemble of\nfinalists"),
    ],
    "im_vs_ip": [
        ("riemann", "Riemann\nper subject"),
        ("pooled_ea_riemann_x1", "Riemann\npooled + EA"),
        ("gpu_eegnet_ea_aug_ft10", "EEGNet\npooled"),
        ("scr_ens_finalists", "Ensemble of\nfinalists"),
        ("behaviour", "Behaviour\nonly"),
    ],
}


def _scores(task: str, run: str) -> np.ndarray:
    base = research_dir() / task
    if run == "behaviour":
        return pd.read_csv(base / "_behaviour_baselines.csv")["config+history"].to_numpy()
    return pd.read_csv(base / f"{run}.csv")["holdout_bacc"].to_numpy()


def fig_ladder(variant: str) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=size("fig1", (11, 4.9)), sharey=True, gridspec_kw={"wspace": 0.08})
    fig.subplots_adjust(left=0.065, right=0.99, top=top("fig1", 0.80), bottom=0.15 if POSTER is None else 0.19)
    headline(fig, "Pooling subjects lifts conflict detection; conflict resolution stays at chance",
             "Balanced accuracy on each subject's untouched final 20 % of trials. Dots: subjects. Bars: mean with 95 % bootstrap CI.")
    rng = np.random.default_rng(1)
    for ax, (task, title, color) in zip(axes, (("c_vs_ip", "C vs IP  ·  was there a conflict?", T["blue"]),
                                               ("im_vs_ip", "IM vs IP  ·  how was it resolved?", T["orange"]))):
        means = []
        for i, (run, label) in enumerate(LADDER[task]):
            vals = _scores(task, run)
            c = T["aqua"] if run == "behaviour" else color
            ax.scatter(i + rng.uniform(-0.17, 0.17, len(vals)), vals, s=9, color=c, alpha=T["dot_alpha"], linewidth=0, zorder=2)
            lo, hi = bootstrap_ci(vals)
            m = float(np.nanmean(vals))
            means.append(m)
            ax.plot([i, i], [lo, hi], color=T["ink"], lw=1.6, zorder=4)
            ringed_dot(ax, i, m, c, size=70)
            ax.text(i + 0.12, m, f"{m * 100:.1f}", fontsize=9.5, fontweight="semibold", color=T["ink"], va="center", zorder=6)
        ax.axhline(0.5, color=T["muted"], lw=1, zorder=1)
        ax.text(-0.5, 0.503, "chance", fontsize=8.5, color=T["muted"], ha="left", va="bottom")
        ax.set_xticks(range(len(means)), [label for _, label in LADDER[task]])
        ax.set_xlim(-0.55, len(means) - 0.45)
        ax.set_title(title)
        if task == "c_vs_ip":
            y = 0.875
            for (a, b, text) in ((0, 1, f"pooling  +{(means[1] - means[0]) * 100:.1f}"),
                                 (1, 4, f"deep model  +{(means[4] - means[1]) * 100:.1f}")):
                ax.annotate("", (a + 0.05, y), (b - 0.05, y),
                            arrowprops={"arrowstyle": "-", "color": T["ink2"], "lw": 1, "shrinkA": 0, "shrinkB": 0})
                for xx in (a + 0.05, b - 0.05):
                    ax.plot([xx, xx], [y - 0.008, y], color=T["ink2"], lw=1)
                ax.text((a + b) / 2, y + 0.006, text, ha="center", va="bottom", fontsize=9, color=T["ink2"])
        else:
            ax.text(4.45, 0.875, "no EEG: stimulus\nconfiguration + history", ha="right", va="bottom", fontsize=8.5, color=T["ink2"])
    axes[0].set_ylabel("Balanced accuracy (holdout)")
    axes[0].set_ylim(0.3, 0.93)
    axes[0].set_yticks(np.arange(0.3, 0.81, 0.1))
    axes[0].yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f} %")
    return save(fig, "fig1_decoding_ladder", variant)


# ----------------------------------------------------------------------------- figure 2


WINDOW_LABELS = {"pre": "−0.5–0 s\nbefore", "w1": "0–0.5 s", "w2": "0.5–1.0 s", "w3": "1.0–1.5 s"}
BAND_LABELS = ["δ 1–4", "θ 4–8", "α 8–13", "β 13–20", "β 20–30", "γ 30–45", "γ 55–95"]


def _neuro() -> Path:
    return research_dir() / "im_neuro"


def _recolour_head(ax) -> None:
    for line in ax.lines:
        if line.get_marker() in (None, "None", "", " "):
            line.set_color(T["ink2"])
            line.set_linewidth(0.9)
    for patch in ax.patches:
        patch.set_edgecolor(T["ink2"])


def fig_maps(variant: str) -> Path:
    import mne

    from .im_neuro import channel_info

    maps = np.load(_neuro() / "spectral_maps.npz")
    info = channel_info(maps["channels"])
    alpha = 2
    vmax = float(np.ceil(np.abs(maps["c_vs_ip_t"][:, alpha]).max()))
    fig = plt.figure(figsize=size("fig2", (11, 5.6)))
    headline(fig, "Conflict leaves a clear alpha–beta footprint; its resolution leaves none",
             "Group t of per-subject effect sizes (n = 47). Dots: sensors in a significant cluster (joint window × band × "
             "sensor permutation, p < 0.05). Red: less power under conflict.")
    grid = fig.add_gridspec(2, 6, left=0.075, right=0.985, top=top("fig2", 0.80), bottom=0.08 if POSTER is None else 0.1, wspace=0.08, hspace=0.25,
                            width_ratios=[1, 1, 1, 1, 0.12, 1.55])
    cmap = diverging()
    rows = (("c_vs_ip", "C vs IP", "congruent − conflict"), ("im_vs_ip", "IM vs IP", "mixed − single colour"))
    for r, (key, name, sub) in enumerate(rows):
        t, mask = maps[f"{key}_t"], maps[f"{key}_mask"]
        for w, wkey in enumerate(WINDOW_LABELS):
            ax = fig.add_subplot(grid[r, w])
            mne.viz.plot_topomap(t[w, alpha], info, axes=ax, show=False, cmap=cmap, vlim=(-vmax, vmax),
                                 mask=mask[w, alpha], contours=0, sensors=False,
                                 mask_params={"marker": "o", "markerfacecolor": T["ink"], "markeredgecolor": T["surface"],
                                              "markeredgewidth": 0.6, "markersize": 3.6, "linestyle": "none"})
            _recolour_head(ax)
            if r == 0:
                ax.set_title(WINDOW_LABELS[wkey], loc="center", fontsize=9.5, fontweight="normal", color=T["ink2"], pad=4)
            if w == 0:
                ax.text(-0.32, 0.5, f"{name}\n", transform=ax.transAxes, rotation=90, ha="center", va="center",
                        fontsize=11, fontweight="semibold", color=T["ink"])
                ax.text(-0.2, 0.5, sub, transform=ax.transAxes, rotation=90, ha="center", va="center",
                        fontsize=8.5, color=T["ink2"])
        # right: number of significant sensors per band x window
        ax = fig.add_subplot(grid[r, 5])
        counts = mask.sum(-1).T  # bands x windows
        seq = LinearSegmentedColormap.from_list("seq", [T["mid"], T["blue"] if key == "c_vs_ip" else T["orange"]])
        ax.imshow(counts, cmap=seq, vmin=0, vmax=60, aspect="auto")
        for (b, w), v in np.ndenumerate(counts):
            if v:
                ax.text(w, b, str(v), ha="center", va="center", fontsize=8,
                        color="#ffffff" if v > 30 else T["ink"])
        ax.set_xticks(range(4), ["pre", "0–0.5", "0.5–1", "1–1.5"] if r == 1 else [""] * 4)
        ax.set_yticks(range(7), BAND_LABELS, fontsize=8)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, 4), minor=True)
        ax.set_yticks(np.arange(-0.5, 7), minor=True)
        ax.grid(which="minor", color=T["surface"], linewidth=2)
        ax.tick_params(which="minor", length=0)
        if r == 0:
            ax.set_title("Significant sensors (of 60), all bands", loc="left", fontsize=9.5, fontweight="normal",
                         color=T["ink2"], pad=4)
        if counts.sum() == 0:
            ax.text(1.5, 3, "none", ha="center", va="center", fontsize=10, color=T["ink2"])
    fig.text(0.075 + 0.62 * 0.5, 0.025, "Alpha band (8–13 Hz) maps", ha="center", fontsize=9, color=T["ink2"])
    cax = fig.add_axes([0.075, 0.035, 0.12, 0.012])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-vmax, vmax), cmap=cmap), cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.set_ticks([-vmax, 0, vmax])
    cb.ax.tick_params(labelsize=8, colors=T["ink2"], length=0)
    cax.text(-0.04, 0.5, "t", transform=cax.transAxes, ha="right", va="center", fontsize=9, color=T["ink2"])
    return save(fig, "fig2_spectral_maps", variant)


# ----------------------------------------------------------------------------- figure 3


def fig_balance(variant: str) -> Path:
    from scipy.stats import norm, spearmanr

    bal = pd.read_csv(_neuro() / "balance_model.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=size("fig3", (11, 4.6)), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.22 if POSTER is None else 0.3})
    fig.subplots_adjust(left=0.04 if POSTER is None else 0.02, right=0.985, top=top("fig3", 0.78), bottom=0.13 if POSTER is None else 0.16)
    headline(fig, "A one-variable model of the report: mixed percepts happen when the eyes are balanced",
             "Each configuration swaps which eye sees which colour. If eye and colour advantages add, one configuration is "
             "more balanced, and should give more “mixed” reports.")

    # left: schematic of the ordered model
    x = np.linspace(-4, 4, 600)
    theta = 0.9
    ax1.axvspan(-theta, theta, color=T["orange"], alpha=0.10, lw=0)
    for s, name, ls in ((0.25, "balanced configuration", "-"), (1.6, "unbalanced configuration", "-")):
        y = norm.pdf(x, s)
        c = T["orange"] if s < 1 else T["ink2"]
        ax1.plot(x, y, color=c, lw=2)
        inside = (x > -theta) & (x < theta)
        ax1.fill_between(x[inside], 0, y[inside], color=c, alpha=0.18 if s < 1 else 0.10, lw=0)
        p_mixed = norm.cdf(theta - s) - norm.cdf(-theta - s)
        peak = norm.pdf(0)
        ax1.text(-theta - 0.12 if s < 1 else s + 0.35, peak + 0.015, f"{name}\nP(mixed) = {p_mixed:.0%}", fontsize=fs(8.8),
                 color=T["ink"], va="bottom", ha="right" if s < 1 else "left")
    for edge in (-theta, theta):
        ax1.axvline(edge, color=T["ink2"], lw=1)
    ax1.text(0, -0.055, "“mixed”", ha="center", fontsize=fs(9.5), color=T["ink"], fontweight="semibold")
    ax1.text(-2.5, -0.055, "“colour 1”", ha="center", fontsize=fs(9.5), color=T["ink2"])
    ax1.text(2.5, -0.055, "“colour 2”", ha="center", fontsize=fs(9.5), color=T["ink2"])
    ax1.text(theta + 0.05, 0.53, "θ", fontsize=fs(10), color=T["ink2"])
    ax1.text(-theta - 0.05, 0.53, "−θ", fontsize=fs(10), color=T["ink2"], ha="right")
    ax1.set_ylim(-0.09, 0.62)
    ax1.set_xlim(-4, 4)
    ax1.set_yticks([])
    ax1.set_xticks([])
    ax1.grid(False)
    ax1.spines["left"].set_visible(False)
    ax1.set_title("Model: latent balance s + noise, cut at ±θ" if POSTER is None else "Model: balance s + noise, cut at ±θ")
    ax1.set_xlabel("latent balance s = eye advantage + colour advantage", fontsize=fs(9))

    # right: test on independent trials
    dx = (bal.pure_imbalance_a - bal.pure_imbalance_b) * 100
    dy = (bal.mixed_a - bal.mixed_b) * 100
    lim_x = float(np.ceil(dx.abs().max() / 10) * 10)
    lim_y = float(np.ceil(dy.abs().max() / 10) * 10)
    ax2.fill_between([0, lim_x], -lim_y, 0, color=T["orange"], alpha=0.07, lw=0)
    ax2.fill_between([-lim_x, 0], 0, lim_y, color=T["orange"], alpha=0.07, lw=0)
    ax2.axhline(0, color=T["axis"], lw=1)
    ax2.axvline(0, color=T["axis"], lw=1)
    slope, intercept = np.polyfit(dx, dy, 1)
    xs = np.array([-lim_x, lim_x])
    ax2.plot(xs, intercept + slope * xs, color=T["ink2"], lw=1.4)
    ringed_dot(ax2, dx, dy, T["orange"], size=42 if POSTER is None else 22)
    rho, p = spearmanr(dx, dy)
    ok = np.mean(np.sign(dx) == -np.sign(dy))
    ax2.text(0.98, 0.97, f"Spearman ρ = {rho:.2f}, p = {p:.3f}\n{ok:.0%} of subjects in the predicted quadrants".replace("-", "−"),
             transform=ax2.transAxes, ha="right", va="top", fontsize=fs(9), color=T["ink"])
    ax2.text(lim_x * 0.97, -lim_y * 0.93, "predicted", ha="right", fontsize=fs(8.5), color=T["ink2"])
    ax2.text(-lim_x * 0.97, lim_y * 0.88, "predicted", ha="left", fontsize=fs(8.5), color=T["ink2"])
    ax2.set_xlim(-lim_x, lim_x)
    ax2.set_ylim(-lim_y, lim_y)
    ax2.grid(True, axis="both")
    ax2.set_xlabel("Imbalance, config. A − B (pure reports only, points)" if POSTER is None else "Imbalance A − B (pure reports only)")
    ax2.set_ylabel("“Mixed” rate, config. A − B (points)" if POSTER is None else "“Mixed” rate A − B (points)")
    ax2.set_title("Test on separate trials: more imbalance → fewer “mixed”" if POSTER is None else "Test on separate trials")
    return save(fig, "fig3_balance_model", variant)


# ----------------------------------------------------------------------------- figure 4


def fig_behaviour(variant: str) -> Path:
    subjects = pd.read_csv(_neuro() / "behaviour_subjects.csv")
    lags = pd.read_csv(_neuro() / "behaviour_lags.csv").assign(excess=lambda d: d.observed - d.null)
    drift = pd.read_csv(_neuro() / "behaviour_drift.csv")
    ind = pd.read_csv(_neuro() / "individual_differences.csv")
    table = pd.read_csv(_neuro() / "subject_table.csv")

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.0), gridspec_kw={"width_ratios": [1, 1.1, 1.1, 1.0], "wspace": 0.38})
    fig.subplots_adjust(left=0.04, right=0.985, top=0.74, bottom=0.17)
    pos = axes[3].get_position()
    axes[3].set_position([pos.x0 + 0.06, pos.y0, pos.width - 0.06, pos.height])
    headline(fig, "Who says “mixed” is a trait; when they say it depends on the last two trials",
             "47 subjects. Error bars: 95 % bootstrap CI across subjects.", y=0.985)

    ax = axes[0]
    rates = np.sort(subjects.mixed_rate.to_numpy()) * 100
    ax.hlines(np.arange(len(rates)), 0, rates, color=T["grid"], lw=1)
    ringed_dot(ax, rates, np.arange(len(rates)), T["orange"], size=22)
    ax.set_yticks([])
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.set_xlim(0, 100)
    ax.set_xlabel("“Mixed” reports (% of conflict trials)")
    ax.set_title("A 19–92 % spread")
    ax.text(rates[0] - 3, 0, f"{rates[0]:.0f} %", va="center", ha="right", fontsize=8.5, color=T["ink2"])
    ax.text(rates[-1] - 3, len(rates) - 1, f"{rates[-1]:.0f} %", va="center", ha="right", fontsize=8.5, color=T["ink2"])

    ax = axes[1]
    g = lags.groupby("lag").excess
    mean = g.mean() * 100
    ci = np.array([bootstrap_ci(v.dropna().to_numpy() * 100) for _, v in g])
    ax.axhline(0, color=T["axis"], lw=1)
    ax.fill_between(mean.index, ci[:, 0], ci[:, 1], color=T["orange"], alpha=0.12, lw=0)
    ax.plot(mean.index, mean, color=T["orange"], lw=2)
    ringed_dot(ax, mean.index[:2], mean.iloc[:2], T["orange"], size=40)
    ax.text(2.4, mean.iloc[1] + 0.4, f"+{mean.iloc[0]:.0f} points after\na “mixed” trial", fontsize=8.5, color=T["ink2"])
    ax.set_xlabel("Trials back (conflict trials)")
    ax.set_ylabel("Extra P(“mixed”), points")
    ax.set_title("Perceptual inertia")
    ax.set_xticks([1, 5, 10, 15])

    ax = axes[2]
    g = drift.groupby("decile").mixed_rate
    mean = g.mean() * 100
    ci = np.array([bootstrap_ci(v.to_numpy() * 100) for _, v in g])
    ax.fill_between(mean.index, ci[:, 0], ci[:, 1], color=T["orange"], alpha=0.12, lw=0)
    ax.plot(mean.index, mean, color=T["orange"], lw=2)
    ax.set_xticks([1, 5, 10], ["start", "middle", "end"])
    ax.set_xlabel("Session (tenths of conflict trials)")
    ax.set_ylabel("“Mixed” reports (%)")
    ax.set_title("Rises early, then plateaus")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}")

    ax = axes[3]
    names = {"eye_dominance": "Eye dominance", "iaf": "Alpha frequency", "alpha_peak_height": "Alpha peak height",
             "posterior_alpha": "Posterior alpha power", "conflict_decodability": "Conflict decodability",
             "serial_lag1": "Perceptual inertia"}
    ind = ind.iloc[::-1].reset_index(drop=True)
    rng = np.random.default_rng(0)
    for i, row in ind.iterrows():
        sub = table[["mixed_rate", row.predictor]].dropna()
        boots = []
        for _ in range(2000):
            idx = rng.integers(0, len(sub), len(sub))
            boots.append(sub.iloc[idx].corr(method="spearman").iloc[0, 1])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        ax.plot([lo, hi], [i, i], color=T["ink"], lw=1.6)
        ringed_dot(ax, row.rho, i, T["orange"], size=40)
    ax.axvline(0, color=T["axis"], lw=1)
    ax.set_yticks(range(len(ind)), [names[p] for p in ind.predictor], fontsize=8.5)
    ax.set_xlim(-0.6, 0.6)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.set_xlabel("Spearman ρ with “mixed” rate")
    ax.set_title("Not explained by traits")
    return save(fig, "fig4_behaviour", variant)


# ----------------------------------------------------------------------------- figure 5


def fig_erp(variant: str) -> Path:
    erp = np.load(_neuro() / "erp_roi.npz")
    t = erp["times"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), sharey=True, gridspec_kw={"wspace": 0.08})
    fig.subplots_adjust(left=0.06, right=0.985, top=0.72, bottom=0.16)
    headline(fig, "Evoked responses: mixed and single-colour percepts overlap almost everywhere",
             "Grand average over 47 subjects (0.3–20 Hz, common average). Orange band: temporal cluster with p < 0.05 "
             "(uncorrected for the three regions). The large deflections sit at the stimulus edges.")
    for ax, roi in zip(axes, ("occipital", "parietal", "frontal")):
        im, ip, sig = erp[f"{roi}_im"] / 1e6, erp[f"{roi}_ip"] / 1e6, erp[f"{roi}_sig"]  # cache is µV x 1e6
        ax.axvspan(0, 1.5, color=T["grid"], alpha=0.35, lw=0)
        for s in np.flatnonzero(sig):
            ax.axvspan(t[s] - 0.025, t[s] + 0.025, color=T["orange"], alpha=0.18, lw=0)
        ax.axhline(0, color=T["axis"], lw=0.8)
        ax.plot(t, ip.mean(0), color=T["ink2"], lw=2, label="single colour (IP)")
        ax.plot(t, im.mean(0), color=T["orange"], lw=2, label="mixed (IM)")
        ax.set_title(roi.capitalize())
        ax.set_xlim(-0.5, 1.95)
        ax.set_xticks([0, 0.5, 1, 1.5])
        ax.set_xlabel("Time from stimulus onset (s)")
        ax.text(0.02, 0.04, "stimulus on screen", transform=ax.transAxes, fontsize=8, color=T["muted"]) if roi == "occipital" else None
    axes[0].set_ylabel("Amplitude (µV)")
    axes[2].legend(loc="upper right", handlelength=1.4)
    return save(fig, "fig5_erp", variant)


FIGURES = [fig_ladder, fig_maps, fig_balance, fig_behaviour, fig_erp]


def fig_poster_inertia(variant: str) -> Path:
    lags = pd.read_csv(_neuro() / "behaviour_lags.csv").assign(excess=lambda d: d.observed - d.null)
    fig, ax = plt.subplots(figsize=(3.85, 3.1))
    fig.subplots_adjust(left=0.17, right=0.97, top=0.88, bottom=0.16)
    g = lags[lags.lag <= 10].groupby("lag").excess
    mean = g.mean() * 100
    ci = np.array([bootstrap_ci(v.dropna().to_numpy() * 100) for _, v in g])
    ax.axhline(0, color=T["axis"], lw=1)
    ax.fill_between(mean.index, ci[:, 0], ci[:, 1], color=T["orange"], alpha=0.12, lw=0)
    ax.plot(mean.index, mean, color=T["orange"], lw=2)
    ringed_dot(ax, mean.index[:2], mean.iloc[:2], T["orange"], size=34)
    ax.text(2.5, mean.iloc[1] + 0.3, f"+{mean.iloc[0]:.0f} points after\na “mixed” trial", fontsize=8, color=T["ink2"])
    ax.set_xticks([1, 2, 4, 6, 8, 10])
    ax.set_xlabel("Conflict trials back")
    ax.set_ylabel("Extra P(“mixed”), points")
    ax.set_title("Perceptual inertia")
    return save(fig, "inertia", variant)


def _slope_panel(ax, before, after, labels, title, note):
    up = after > before
    for b, a, u in zip(before, after, up):
        ax.plot([0, 1], [b, a], color=T["orange"] if u else T["muted"], alpha=0.45 if u else 0.5, lw=1)
    mb, ma = float(np.mean(before)), float(np.mean(after))
    ax.plot([0, 1], [mb, ma], color=T["ink"], lw=2.4, zorder=5)
    ringed_dot(ax, [0, 1], [mb, ma], T["ink"], size=40, z=6)
    ax.text(-0.06, mb, f"{mb:.0f} %", ha="right", va="center", fontsize=9, fontweight="semibold", color=T["ink"])
    ax.text(1.06, ma, f"{ma:.0f} %", ha="left", va="center", fontsize=9, fontweight="semibold", color=T["ink"])
    ax.set_xticks([0, 1], labels)
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.text(0.5, 0.03, note, transform=ax.transAxes, ha="center", va="bottom", fontsize=8, color=T["ink2"])


def fig_poster_behaviour(variant: str) -> Path:
    """Two paired comparisons a reader can take in at a glance: one line per person."""
    from scipy.stats import wilcoxon

    from .im_neuro import IM_CODES, IP_CODES, load_traits

    bal = pd.read_csv(_neuro() / "balance_model.csv")
    bal = bal[bal.pure_imbalance_a != bal.pure_imbalance_b]
    a_balanced = bal.pure_imbalance_a < bal.pure_imbalance_b
    more = np.where(a_balanced, bal.mixed_a, bal.mixed_b) * 100
    less = np.where(a_balanced, bal.mixed_b, bal.mixed_a) * 100

    after_pure, after_mixed = [], []
    for d in load_traits().values():
        c = d["codes"]
        y = np.isin(c[np.isin(c, IP_CODES + IM_CODES)], IM_CODES).astype(int)
        after_mixed.append(y[1:][y[:-1] == 1].mean() * 100)
        after_pure.append(y[1:][y[:-1] == 0].mean() * 100)
    after_pure, after_mixed = np.array(after_pure), np.array(after_mixed)

    fig, axes = plt.subplots(1, 2, figsize=(10.7, 3.0), sharey=True, gridspec_kw={"wspace": 0.18})
    fig.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.17)
    _slope_panel(axes[0], less, more, ["less balanced\nconfiguration", "more balanced\nconfiguration"],
                 "Balanced eyes → more “mixed”",
                 f"{np.mean(more > less):.0%} of people go up · p = {wilcoxon(more - less).pvalue:.2f}")
    _slope_panel(axes[1], after_pure, after_mixed, ["after a\nsingle-colour report", "after a\n“mixed” report"],
                 "The last report sticks",
                 f"{np.mean(after_mixed > after_pure):.0%} of people go up · p < 10⁻¹¹")
    axes[0].set_ylabel("“Mixed” reports (%)")
    axes[0].yaxis.set_major_formatter(lambda v, _: f"{v:.0f}")
    return save(fig, "behaviour_slopes", variant)


def poster(directory: Path) -> None:
    """Poster versions of figures 1-3 plus the inertia panel, light theme, at A3 physical size."""
    global POSTER
    POSTER = {"dir": directory}
    directory.mkdir(parents=True, exist_ok=True)
    small = {"font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5, "xtick.labelsize": 8,
             "ytick.labelsize": 8, "legend.fontsize": 8, "axes.titlepad": 7}
    try:
        with theme("light"), plt.rc_context(small):
            for make in (fig_ladder, fig_maps, fig_poster_behaviour):
                print(make("light"))
    finally:
        POSTER = None


def main() -> None:
    import sys

    if len(sys.argv) > 2 and sys.argv[1] == "--poster":
        poster(Path(sys.argv[2]))
        return
    for variant in ("light", "dark"):
        with theme(variant):
            for make in FIGURES:
                print(make(variant))


if __name__ == "__main__":
    main()
