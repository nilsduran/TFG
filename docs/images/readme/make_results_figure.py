"""Regenerate the README results figure from the thesis final-holdout table."""

from pathlib import Path

import matplotlib.pyplot as plt

# Final chronological-holdout balanced accuracy (mean, SD across subjects), thesis Table "balanced-accuracy".
RESULTS = [
    ("Logistic regression · FCz theta", 49.6, 6.7, 50.6, 8.6),
    ("Logistic regression · Oz alpha", 53.6, 8.2, 49.4, 8.0),
    ("Riemannian covariance + LDA", 58.7, 8.3, 49.3, 6.1),
    ("Experiment-guided classical ML", 56.9, 8.1, 50.9, 8.1),
    ("SVC · cluster features", 56.2, 6.3, 52.7, 6.2),
    ("EEGNet · agnostic", 53.2, 8.2, 49.5, 5.8),
    ("EEGNet · clustering", 49.6, 8.0, 48.5, 7.5),
    ("ST-GNN", 52.0, 7.3, 51.4, 5.6),
    ("ST-EEGFormer (pretrained)", 52.5, 8.3, 50.7, 5.8),
    ("LaBraM (pretrained)", 51.9, 6.8, 48.2, 9.6),
]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "text.color": INK})
fig, ax = plt.subplots(figsize=(9.2, 5.4), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

rows = list(reversed(RESULTS))
for i, (name, c_mean, c_sd, im_mean, im_sd) in enumerate(rows):
    for offset, mean, sd, color in ((0.17, c_mean, c_sd, BLUE), (-0.17, im_mean, im_sd, ORANGE)):
        ax.plot([mean - sd, mean + sd], [i + offset] * 2, color=color, lw=2, alpha=0.35, solid_capstyle="round")
        ax.plot(mean, i + offset, "o", ms=8, color=color, mec=SURFACE, mew=2, zorder=3)
    weight = "bold" if name.startswith("Riemannian") else "normal"
    ax.text(c_mean + c_sd + 0.6, i + 0.17, f"{c_mean:.1f}", va="center", fontsize=9, color=MUTED, fontweight=weight)

ax.axvline(50, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
ax.text(50.3, len(rows) - 0.35, "chance", ha="left", va="bottom", fontsize=9, color=MUTED)
ax.axhspan(4.5, 9.5, color=GRID, alpha=0.35, zorder=0, lw=0)
ax.text(70.4, 9.2, "classical", ha="right", va="center", fontsize=9, color=MUTED, style="italic")
ax.text(70.4, 4.2, "deep / pretrained", ha="right", va="center", fontsize=9, color=MUTED, style="italic")

ax.set_yticks(range(len(rows)), [r[0] for r in rows])
ax.set_xlim(38, 71)
ax.set_ylim(-0.6, len(rows) - 0.1)
ax.set_xlabel("Balanced accuracy on the untouched final 20 % of trials (%) · mean ± 1 SD across subjects", color=MUTED)
ax.xaxis.grid(True, color=GRID, lw=0.8)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(colors=MUTED, length=0)
ax.tick_params(axis="y", colors=INK)

handles = [
    plt.Line2D([], [], marker="o", ls="", ms=8, color=BLUE, label="C vs IP  — is there interocular conflict?"),
    plt.Line2D([], [], marker="o", ls="", ms=8, color=ORANGE, label="IM vs IP  — mixed vs single-colour percept"),
]
ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.32, 1.0), ncol=2, frameon=False, fontsize=9.5)
fig.tight_layout()
fig.savefig(Path(__file__).with_name("decoding_results.png"), dpi=200, facecolor=SURFACE)
