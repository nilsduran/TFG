"""Summaries of research runs: group statistics, prevalence and paired gains."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .harness import load_predictions, results_dir
from .stats import group_summary, paired_comparison, prevalence_inference


def available_runs(task_id: str) -> list[str]:
    return sorted(path.stem for path in results_dir(task_id).glob("*.csv") if not path.stem.startswith("_"))


def summarize_task(task_id: str, reference: str = "riemann", n_second_level: int = 100_000) -> pd.DataFrame:
    directory = results_dir(task_id)
    reference_table = (
        pd.read_csv(directory / f"{reference}.csv").set_index("subject")
        if (directory / f"{reference}.csv").exists()
        else None
    )
    rows = []
    for run_id in available_runs(task_id):
        table = pd.read_csv(directory / f"{run_id}.csv").set_index("subject")
        predictions = load_predictions(task_id, run_id)
        first_level = np.stack([predictions[s]["null"] for s in table.index])
        prevalence = prevalence_inference(first_level, n_second_level=n_second_level)
        holdout = group_summary(table.holdout_bacc.values)
        metadata_path = directory / f"{run_id}.json"
        description = json.loads(metadata_path.read_text()).get("description", "") if metadata_path.exists() else ""
        row = {
            "run": run_id,
            "dev_bacc": table.dev_bacc.mean(),
            "holdout_bacc": holdout["mean"],
            "holdout_sd": holdout["sd"],
            "holdout_auc": table.holdout_auc.mean(),
            "above_chance": f"{holdout['above_chance']}/{holdout['n']}",
            "subjects_p05": int(np.sum(table.holdout_perm_p < 0.05)),
            "wilcoxon_p": holdout["wilcoxon_p"],
            "prevalence_gamma0": prevalence.gamma0,
            "prevalence_global_p": prevalence.global_p,
            "description": description,
        }
        if reference_table is not None and run_id != reference:
            common = table.index.intersection(reference_table.index)
            gain = paired_comparison(table.loc[common, "holdout_bacc"], reference_table.loc[common, "holdout_bacc"])
            dev_gain = paired_comparison(table.loc[common, "dev_bacc"], reference_table.loc[common, "dev_bacc"])
            row.update(
                {
                    "gain_vs_ref": gain["mean_gain"],
                    "gain_p": gain["wilcoxon_p"],
                    "dev_gain_vs_ref": dev_gain["mean_gain"],
                    "dev_gain_p": dev_gain["wilcoxon_p"],
                }
            )
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("dev_bacc", ascending=False)
    summary.to_csv(directory / "_summary.csv", index=False)
    return summary


def format_summary(summary: pd.DataFrame) -> str:
    columns = [
        "run",
        "dev_bacc",
        "holdout_bacc",
        "holdout_sd",
        "holdout_auc",
        "above_chance",
        "subjects_p05",
        "prevalence_gamma0",
        "gain_vs_ref",
        "gain_p",
    ]
    present = [c for c in columns if c in summary]
    return summary[present].to_string(index=False, float_format=lambda v: f"{v:.3f}")
