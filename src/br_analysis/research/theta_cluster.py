"""Channel x time cluster tests for C vs IP band power, all 47 subjects, stimulus window 0-1.5 s.

Re-runs the thesis' 2-D cluster search under the current protocol (no subject selection):

* fronto-medial theta restricted to the article ROI (Fz, FCz, Cz), as in the thesis;
* theta and alpha over the whole scalp, for reference.

Cluster-forming thresholds p = 0.01 and 0.05 (two-sided), 5000 sign-flip permutations.

    python -m br_analysis.research.theta_cluster
"""

from __future__ import annotations

import json

import mne
import numpy as np
from scipy import stats

from ..cluster_analysis import ARTICLE_ROIS, subject_condition_means
from ..config import ProjectPaths
from ..power import load_power_dataset
from ..reduction import channel_adjacency
from ..tasks import get_task


def _test(effects: np.ndarray, channels: list[str], times: np.ndarray, p_forming: float) -> list[dict]:
    """effects: (subjects, channels, times) of C - IP."""
    threshold = float(stats.t.ppf(1 - p_forming / 2, df=len(effects) - 1))
    adjacency = channel_adjacency(channels)
    t_obs, clusters, p_values, _ = mne.stats.spatio_temporal_cluster_1samp_test(
        effects.transpose(0, 2, 1), adjacency=adjacency, threshold=threshold, tail=0,
        n_permutations=5000, seed=42, out_type="indices", verbose=False)
    out = []
    for (time_idx, chan_idx), p in sorted(zip(clusters, p_values), key=lambda c: c[1])[:3]:
        sign = float(np.sign(t_obs[time_idx, chan_idx].mean()))
        out.append({"p": round(float(p), 4), "sign_C_minus_IP": sign, "points": int(len(time_idx)),
                    "channels": sorted({channels[c] for c in chan_idx}),
                    "t_from": round(float(times[time_idx].min()), 3), "t_to": round(float(times[time_idx].max()), 3)})
    return out


def main() -> None:
    paths = ProjectPaths.from_env("research", create_dirs=False)
    dataset = load_power_dataset(paths, get_task("c_vs_ip"), verbose=False)
    channels, times = list(dataset.channel_names), np.asarray(dataset.times)
    result = {"n_subjects": len(dataset.subject_ids), "times": [float(times[0]), float(times[-1])]}
    for band in ("theta", "alpha"):
        means, _ = subject_condition_means(dataset, band)
        effects = means[:, 0] - means[:, 1]
        for p_forming in (0.01, 0.05):
            result[f"{band} | whole scalp | p<{p_forming}"] = _test(effects, channels, times, p_forming)
            if band == "theta":
                roi = [channels.index(c) for c in ARTICLE_ROIS["theta"]]
                result[f"theta | Fz FCz Cz | p<{p_forming}"] = _test(effects[:, roi], list(ARTICLE_ROIS["theta"]), times, p_forming)
    out = paths.artifacts_dir / "research" / "theta_cluster.json"
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
