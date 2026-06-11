from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

OOF_CACHE_VERSION = "stratified_group_kfold_holdout_v1"
OOF_CV_STRATEGY = "stratified_group_kfold_development_only"

MODEL_ORDER = [
    "LR_theta",
    "LR_alpha",
    "Riemann",
    "Experiment_Guided",
    "EEGNet_Agnostic",
    "EEGNet_Clustering",
    "ST_GNN",
    "ST_EEGFormer",
    "LaBraM",
]

MODEL_LABELS = {
    "LR_theta": "LR theta (FCz)",
    "LR_alpha": "LR alpha (Oz)",
    "Riemann": "Riemann TS-LDA",
    "Experiment_Guided": "Experiment-guided",
    "EEGNet_Agnostic": "EEGNet agnostic",
    "EEGNet_Clustering": "EEGNet clustering",
    "ST_GNN": "ST-GNN",
    "ST_EEGFormer": "ST-EEGFormer",
    "LaBraM": "LaBraM",
}

MODEL_FAMILIES = {
    "LR_theta": "targeted classical",
    "LR_alpha": "targeted classical",
    "Riemann": "covariance classical",
    "Experiment_Guided": "hypothesis-guided",
    "EEGNet_Agnostic": "deep CNN",
    "EEGNet_Clustering": "deep CNN / cluster-restricted",
    "ST_GNN": "graph deep learning",
    "ST_EEGFormer": "transformer",
    "LaBraM": "foundation model",
}

MODEL_COLORS = {
    "LR_theta": "#2878b5",
    "LR_alpha": "#8bbbd9",
    "Riemann": "#d95f02",
    "Experiment_Guided": "#7a5230",
    "EEGNet_Agnostic": "#1b9e77",
    "EEGNet_Clustering": "#66a61e",
    "ST_GNN": "#b2182b",
    "ST_EEGFormer": "#756bb1",
    "LaBraM": "#c51b7d",
}

ALL_CHANNELS = [
    "Fp1", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
]

THETA_CHANNELS = ["Fp1", "Fp2", "Fz", "F1", "F2", "FCz", "FC1", "FC2", "Cz"]
ALPHA_CHANNELS = ["Pz", "P1", "P2", "P3", "P4", "P7", "P8", "Oz", "O1", "O2", "PO3", "PO4", "POz"]
CLUSTER_CHANNELS = [
    "Fz", "FCz", "FC3", "FC1", "FC2", "C3", "C1", "Cz", "C2",
    "CP3", "CP1", "CPz", "CP2", "CP4", "P5", "P3", "P1", "Pz",
    "P2", "P4", "P6", "P8", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
]


@dataclass(frozen=True)
class ExplainabilityContext:
    task_tag: str
    display_name: str
    contrast_type: str
    hypothesis: str
    interpretation_default: str
    class_0: str
    class_1: str
    paths: object
    task: object


def build_context(task_tag: str, paths, task) -> ExplainabilityContext:
    if task_tag == "c_vs_ip":
        return ExplainabilityContext(
            task_tag=task_tag,
            display_name="Congruent vs Incongruent Pure",
            contrast_type="stimulus conflict vs resolved pure percept",
            hypothesis=(
                "Evidence is expected to be weak but, if real, should align with distributed "
                "conflict-related covariance plus frontal theta and posterior alpha structure."
            ),
            interpretation_default=(
                "The strongest thesis claim should come from methods that are above chance, stable "
                "across subjects, and compatible with sensor-level frontal theta / posterior alpha hypotheses."
            ),
            class_0=task.class_0,
            class_1=task.class_1,
            paths=paths,
            task=task,
        )
    if task_tag == "im_vs_ip":
        return ExplainabilityContext(
            task_tag=task_tag,
            display_name="Incongruent Mixed vs Incongruent Pure",
            contrast_type="perceptual outcome within incongruent stimulation",
            hypothesis=(
                "This is the stricter contrast: physical incongruence is shared, so any decodable "
                "signal is expected to be subtler, more variable, and more vulnerable to confounds."
            ),
            interpretation_default=(
                "Interpret IM vs IP as exploratory unless a model is clearly above chance and its "
                "evidence converges with the C vs IP pattern without relying on stale or missing caches."
            ),
            class_0=task.class_0,
            class_1=task.class_1,
            paths=paths,
            task=task,
        )
    raise ValueError(f"Unsupported task_tag: {task_tag}")


def cache_paths(paths, task_tag: str) -> dict[str, Path]:
    if task_tag == "c_vs_ip":
        lr_dir = "c_vs_i"
        riemann_dir = "c_vs_i_riemann"
        experiment_dir = "experiment_guided_config_c_ip"
        files = {
            "LR_theta": "oof_predictions_c_vs_i_FCz_theta_Oz_alpha.npz",
            "LR_alpha": "oof_predictions_c_vs_i_FCz_theta_Oz_alpha.npz",
            "Riemann": "oof_predictions_c_vs_i_riemann_fused.npz",
            "Experiment_Guided": "oof_predictions_c_vs_ip_experiment_guided.npz",
            "EEGNet_Agnostic": "oof_predictions_c_vs_ip_eegnet_agnostic.npz",
            "EEGNet_Clustering": "oof_predictions_c_vs_ip_eegnet_clustering.npz",
            "ST_GNN": "oof_predictions_c_vs_ip_stgnn.npz",
            "ST_EEGFormer": "oof_predictions_c_vs_ip_steegformer.npz",
            "LaBraM": "oof_predictions_c_vs_ip_labram.npz",
        }
    elif task_tag == "im_vs_ip":
        lr_dir = "im_vs_ip"
        riemann_dir = "im_vs_ip_riemann"
        experiment_dir = "experiment_guided_config_im_ip"
        files = {
            "LR_theta": "oof_predictions_im_vs_ip_FCz_theta_Oz_alpha.npz",
            "LR_alpha": "oof_predictions_im_vs_ip_FCz_theta_Oz_alpha.npz",
            "Riemann": "oof_predictions_im_vs_ip_riemann_fused.npz",
            "Experiment_Guided": "oof_predictions_im_vs_ip_experiment_guided.npz",
            "EEGNet_Agnostic": "oof_predictions_im_vs_ip_eegnet_agnostic.npz",
            "EEGNet_Clustering": "oof_predictions_im_vs_ip_eegnet_clustering.npz",
            "ST_GNN": "oof_predictions_im_vs_ip_stgnn.npz",
            "ST_EEGFormer": "oof_predictions_im_vs_ip_steegformer.npz",
            "LaBraM": "oof_predictions_im_vs_ip_labram.npz",
        }
    else:
        raise ValueError(f"Unsupported task_tag: {task_tag}")

    return {
        "LR_theta": paths.analysis_root / lr_dir / "cache" / files["LR_theta"],
        "LR_alpha": paths.analysis_root / lr_dir / "cache" / files["LR_alpha"],
        "Riemann": paths.analysis_root / riemann_dir / "cache" / files["Riemann"],
        "Experiment_Guided": paths.analysis_root / experiment_dir / "cache" / files["Experiment_Guided"],
        "EEGNet_Agnostic": paths.cache_dir / files["EEGNet_Agnostic"],
        "EEGNet_Clustering": paths.cache_dir / files["EEGNet_Clustering"],
        "ST_GNN": paths.cache_dir / files["ST_GNN"],
        "ST_EEGFormer": paths.cache_dir / files["ST_EEGFormer"],
        "LaBraM": paths.cache_dir / files["LaBraM"],
    }


def final_holdout_paths(paths, task_tag: str) -> dict[str, Path]:
    if task_tag == "c_vs_ip":
        return {
            "Experiment_Guided": paths.analysis_root / "experiment_guided_config_c_ip" / "tables" / "experiment_guided_config_c_ip_final_holdout_ONCE.csv",
            "EEGNet_Agnostic": paths.analysis_root / "eegnet" / "tables" / "final_holdout_c_vs_ip_eegnet.csv",
            "EEGNet_Clustering": paths.analysis_root / "eegnet" / "tables" / "final_holdout_c_vs_ip_eegnet.csv",
            "ST_GNN": paths.analysis_root / "c_vs_i_stgnn" / "tables" / "final_holdout_c_vs_ip_stgnn.csv",
            "ST_EEGFormer": paths.analysis_root / "c_vs_i_steegformer" / "tables" / "final_holdout_c_vs_ip_steegformer.csv",
            "LaBraM": paths.analysis_root / "c_vs_i_labram" / "tables" / "final_holdout_c_vs_ip_labram.csv",
        }
    if task_tag == "im_vs_ip":
        return {
            "Experiment_Guided": paths.analysis_root / "experiment_guided_config_im_ip" / "tables" / "experiment_guided_config_im_ip_final_holdout_ONCE.csv",
            "EEGNet_Agnostic": paths.analysis_root / "eegnet_im_vs_ip" / "tables" / "final_holdout_im_vs_ip_eegnet.csv",
            "EEGNet_Clustering": paths.analysis_root / "eegnet_im_vs_ip" / "tables" / "final_holdout_im_vs_ip_eegnet.csv",
            "ST_GNN": paths.analysis_root / "im_vs_i_stgnn" / "tables" / "final_holdout_im_vs_ip_stgnn.csv",
            "ST_EEGFormer": paths.analysis_root / "im_vs_i_steegformer" / "tables" / "final_holdout_im_vs_ip_steegformer.csv",
            "LaBraM": paths.analysis_root / "im_vs_i_labram" / "tables" / "final_holdout_im_vs_ip_labram.csv",
        }
    raise ValueError(f"Unsupported task_tag: {task_tag}")


def safe_subject_ids(task) -> list[str]:
    path = Path(task.selected_subjects_csv)
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if "subject" not in df.columns:
        return []
    return df["subject"].dropna().astype(str).tolist()


def oof_cache_status(path: Path) -> tuple[bool, str]:
    if not Path(path).exists():
        return False, "missing"
    try:
        with np.load(path, allow_pickle=True) as data:
            if "_cache_version" not in data:
                return False, "legacy cache without _cache_version"
            version = str(np.asarray(data["_cache_version"]).item())
            if version != OOF_CACHE_VERSION:
                return False, f"cache version {version} != {OOF_CACHE_VERSION}"
            strategy = str(np.asarray(data["_cv_strategy"]).item()) if "_cv_strategy" in data else "unknown"
            if strategy != OOF_CV_STRATEGY:
                return False, f"unexpected cv strategy {strategy}"
    except Exception as exc:
        return False, f"could not read metadata: {exc}"
    return True, "ok"


def prob_key(model_name: str, subject: str) -> str:
    if model_name == "LR_theta":
        return f"{subject}_FCz-theta_prob"
    if model_name == "LR_alpha":
        return f"{subject}_Oz-alpha_prob"
    if model_name == "Riemann":
        return f"{subject}_fused_prob"
    return f"{subject}_prob"


def load_per_subject_accuracy(paths, task, task_tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    subjects = safe_subject_ids(task)
    caches = cache_paths(paths, task_tag)
    status_rows = []
    rows = []
    for model_name, cache_file in caches.items():
        fresh, reason = oof_cache_status(cache_file)
        status_rows.append(
            {
                "Model": model_name,
                "OOF cache": cache_file.name,
                "OOF status": reason,
                "OOF usable": fresh,
                "Source": str(cache_file),
            }
        )

    for sub in subjects:
        row = {"subject": sub}
        for model_name, cache_file in caches.items():
            fresh, _ = oof_cache_status(cache_file)
            if not fresh:
                continue
            with np.load(cache_file, allow_pickle=True) as data:
                yt_key = f"{sub}_y_true"
                yp_key = prob_key(model_name, sub)
                if yt_key not in data or yp_key not in data:
                    continue
                y_true = np.asarray(data[yt_key])
                y_prob = np.asarray(data[yp_key], dtype=float)
                valid = ~np.isnan(y_prob)
                if valid.any():
                    row[model_name] = float(((y_prob[valid] >= 0.5).astype(int) == y_true[valid]).mean())
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["subject"] + MODEL_ORDER)
    present = [c for c in MODEL_ORDER if c in df.columns]
    if present:
        df = df.dropna(subset=present, how="all")
    return df, pd.DataFrame(status_rows)


def load_oof_predictions(paths, task, task_tag: str) -> dict[str, dict[str, np.ndarray]]:
    subjects = safe_subject_ids(task)
    out: dict[str, dict[str, np.ndarray]] = {}
    for model_name, cache_file in cache_paths(paths, task_tag).items():
        fresh, _ = oof_cache_status(cache_file)
        if not fresh:
            continue
        y_true_all = []
        y_prob_all = []
        with np.load(cache_file, allow_pickle=True) as data:
            for sub in subjects:
                yt_key = f"{sub}_y_true"
                yp_key = prob_key(model_name, sub)
                if yt_key not in data or yp_key not in data:
                    continue
                y_true = np.asarray(data[yt_key])
                y_prob = np.asarray(data[yp_key], dtype=float)
                valid = ~np.isnan(y_prob)
                y_true_all.append(y_true[valid])
                y_prob_all.append(y_prob[valid])
        if y_true_all:
            out[model_name] = {
                "y_true": np.concatenate(y_true_all),
                "y_prob": np.concatenate(y_prob_all),
            }
    return out


def load_final_holdout_summary(paths, task_tag: str) -> pd.DataFrame:
    rows = []
    classical = paths.analysis_root / "cross_model_comparison" / "tables" / f"final_holdout_classical_baselines_{task_tag}.csv"
    task_specific = paths.analysis_root / f"cross_model_comparison_{task_tag}" / "tables" / f"final_holdout_classical_baselines_{task_tag}.csv"
    for model in ["LR_theta", "LR_alpha", "Riemann"]:
        source = task_specific if task_specific.exists() else classical
        if source.exists():
            df = pd.read_csv(source)
            if "Model" in df.columns:
                vals = pd.to_numeric(df.loc[df["Model"].eq(model), "Accuracy"], errors="coerce")
                rows.append({"Model": model, "Final holdout accuracy": vals.mean(), "Final holdout status": "ok", "Final holdout source": str(source)})
                continue
        rows.append({"Model": model, "Final holdout accuracy": np.nan, "Final holdout status": "missing_final_holdout", "Final holdout source": str(source)})

    for model, source in final_holdout_paths(paths, task_tag).items():
        if not source.exists():
            rows.append({"Model": model, "Final holdout accuracy": np.nan, "Final holdout status": "missing_final_holdout", "Final holdout source": str(source)})
            continue
        df = pd.read_csv(source)
        if "Model" in df.columns:
            df["Model"] = df["Model"].replace({"ST-GNN": "ST_GNN", "ST-EEGFormer": "ST_EEGFormer"})
            vals = pd.to_numeric(df.loc[df["Model"].eq(model), "Accuracy"], errors="coerce")
            if vals.dropna().empty and model == "EEGNet_Clustering":
                vals = pd.to_numeric(df.loc[df["Model"].astype(str).str.contains("Cluster", case=False, na=False), "Accuracy"], errors="coerce")
            if vals.dropna().empty and model == "EEGNet_Agnostic":
                vals = pd.to_numeric(df.loc[df["Model"].astype(str).str.contains("Agnostic", case=False, na=False), "Accuracy"], errors="coerce")
        elif "Accuracy" in df.columns:
            vals = pd.to_numeric(df["Accuracy"], errors="coerce")
        else:
            vals = pd.Series(dtype=float)
        rows.append({"Model": model, "Final holdout accuracy": vals.mean(), "Final holdout status": "ok", "Final holdout source": str(source)})
    return pd.DataFrame(rows)


def _normalize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    total = np.abs(arr).sum()
    return arr / total if total > 0 else arr


def _channel_vector(channels: list[str], weights: list[float]) -> np.ndarray:
    values = np.zeros(len(ALL_CHANNELS), dtype=float)
    lookup = {name.upper(): i for i, name in enumerate(ALL_CHANNELS)}
    for ch, weight in zip(channels, weights):
        idx = lookup.get(ch.upper())
        if idx is not None:
            values[idx] += float(weight)
    return _normalize(values)


def canonical_attribution_priors() -> dict[str, dict[str, object]]:
    return {
        "LR_theta": {
            "source": "model prior / feature definition",
            "availability": "available_feature_definition",
            "spatial": _channel_vector(["FCz"], [1.0]),
            "temporal_peak": 0.75,
            "temporal_width": 0.25,
            "stability": "high_by_design",
            "pattern": "single frontal-midline theta feature",
        },
        "LR_alpha": {
            "source": "model prior / feature definition",
            "availability": "available_feature_definition",
            "spatial": _channel_vector(["Oz"], [1.0]),
            "temporal_peak": 0.75,
            "temporal_width": 0.25,
            "stability": "high_by_design",
            "pattern": "single posterior alpha feature",
        },
        "Riemann": {
            "source": "model prior / covariance channel set",
            "availability": "available_feature_definition",
            "spatial": _channel_vector(THETA_CHANNELS + ALPHA_CHANNELS, [1.0] * (len(THETA_CHANNELS) + len(ALPHA_CHANNELS))),
            "temporal_peak": 0.75,
            "temporal_width": 0.75,
            "stability": "structural_not_fold_estimated",
            "pattern": "distributed theta/alpha covariance over frontal and posterior sensors",
        },
        "Experiment_Guided": {
            "source": "model prior / cluster restriction",
            "availability": "available_feature_definition",
            "spatial": _channel_vector(CLUSTER_CHANNELS, [1.0] * len(CLUSTER_CHANNELS)),
            "temporal_peak": 0.75,
            "temporal_width": 0.75,
            "stability": "structural_with_circularity_risk",
            "pattern": "hypothesis-guided spatiotemporal cluster restriction",
        },
    }


def load_deep_attribution_summaries(paths, task, task_tag: str) -> pd.DataFrame:
    subjects = safe_subject_ids(task)
    rows = []
    for model_name in ["EEGNet_Agnostic", "EEGNet_Clustering", "ST_GNN", "ST_EEGFormer", "LaBraM"]:
        cache_file = cache_paths(paths, task_tag)[model_name]
        fresh, reason = oof_cache_status(cache_file)
        row = {
            "Model": model_name,
            "Attribution availability": "missing_or_stale",
            "Attribution source": str(cache_file),
            "Attribution stability": "not_available",
            "Attribution note": reason,
            "Spatial vector": None,
            "Temporal vector": None,
        }
        if fresh:
            spatial = []
            temporal = []
            with np.load(cache_file, allow_pickle=True) as data:
                for sub in subjects:
                    sp_key = f"{sub}_spatial_saliency"
                    te_key = f"{sub}_temporal_saliency"
                    if sp_key in data:
                        sp = np.asarray(data[sp_key], dtype=float).ravel()
                        if len(sp) == len(ALL_CHANNELS):
                            spatial.append(_normalize(sp))
                    if te_key in data:
                        temporal.append(_normalize(np.asarray(data[te_key], dtype=float).ravel()))
            if spatial:
                sp_arr = np.vstack(spatial)
                stability = float(np.nanmean(np.corrcoef(sp_arr))) if len(sp_arr) > 1 else np.nan
                row.update(
                    {
                        "Attribution availability": "available_saliency_cache",
                        "Attribution stability": stability,
                        "Attribution note": f"{len(spatial)} subject spatial maps; {len(temporal)} temporal maps",
                        "Spatial vector": _normalize(np.nanmean(sp_arr, axis=0)),
                    }
                )
            if temporal:
                max_len = max(len(t) for t in temporal)
                padded = np.full((len(temporal), max_len), np.nan)
                for i, t in enumerate(temporal):
                    padded[i, : len(t)] = t
                row["Temporal vector"] = _normalize(np.nanmean(padded, axis=0))
        rows.append(row)
    return pd.DataFrame(rows)


def attribution_summary(paths, task, task_tag: str) -> pd.DataFrame:
    rows = []
    for model, info in canonical_attribution_priors().items():
        rows.append(
            {
                "Model": model,
                "Attribution availability": info["availability"],
                "Attribution source": info["source"],
                "Attribution stability": info["stability"],
                "Attribution note": info["pattern"],
                "Spatial vector": info["spatial"],
                "Temporal vector": None,
            }
        )
    deep = load_deep_attribution_summaries(paths, task, task_tag)
    return pd.concat([pd.DataFrame(rows), deep], ignore_index=True)


def performance_summary(df_subject: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        if model not in df_subject.columns:
            rows.append({"Model": model, "Mean accuracy": np.nan, "Std accuracy": np.nan, "N subjects": 0, "Above 0.5": False})
            continue
        vals = pd.to_numeric(df_subject[model], errors="coerce").dropna()
        rows.append(
            {
                "Model": model,
                "Mean accuracy": vals.mean() if not vals.empty else np.nan,
                "Std accuracy": vals.std() if len(vals) > 1 else np.nan,
                "N subjects": int(vals.shape[0]),
                "Above 0.5": bool(vals.mean() > 0.5) if not vals.empty else False,
            }
        )
    return pd.DataFrame(rows)


def build_evidence_table(paths, task, task_tag: str) -> tuple[pd.DataFrame, Mapping[str, object]]:
    df_subject, df_cache = load_per_subject_accuracy(paths, task, task_tag)
    perf = performance_summary(df_subject)
    final = load_final_holdout_summary(paths, task_tag)
    attr = attribution_summary(paths, task, task_tag).drop(columns=["Spatial vector", "Temporal vector"])
    table = perf.merge(final, on="Model", how="left").merge(attr, on="Model", how="left").merge(
        df_cache[["Model", "OOF status", "OOF usable"]], on="Model", how="left"
    )
    table["Family"] = table["Model"].map(MODEL_FAMILIES)
    table["Display"] = table["Model"].map(MODEL_LABELS)
    table["Reliability margin"] = table["Mean accuracy"] - 0.5
    table["Chance/permutation status"] = np.select(
        [
            table["OOF usable"].fillna(False) & (table["Reliability margin"] >= 0.04),
            table["OOF usable"].fillna(False) & (table["Reliability margin"] > 0.0),
            table["OOF usable"].fillna(False),
        ],
        ["above practical chance", "near chance / weak positive", "not above chance"],
        default="no fresh OOF evidence",
    )
    table["Confound risk level"] = table.apply(_confound_risk, axis=1)
    table["Thesis-use recommendation"] = table.apply(_recommendation, axis=1)
    table = table.sort_values(["Mean accuracy", "N subjects"], ascending=[False, False], na_position="last")
    return table, {"subjects": df_subject, "cache_status": df_cache}


def _confound_risk(row: pd.Series) -> str:
    model = row["Model"]
    if not bool(row.get("OOF usable", False)):
        return "high: missing/stale validation artifact"
    margin = row.get("Reliability margin", np.nan)
    if model == "Experiment_Guided" or model == "EEGNet_Clustering":
        return "medium-high: cluster-selection circularity risk"
    if model in {"ST_EEGFormer", "LaBraM"}:
        return "medium-high: adaptation/cache sensitivity"
    if pd.isna(margin) or margin <= 0.0:
        return "medium-high: near chance despite interpretable map risk"
    if margin < 0.04:
        return "medium: weak decoding margin"
    return "medium: EEG subject/time confounds still require caution"


def _recommendation(row: pd.Series) -> str:
    if not bool(row.get("OOF usable", False)):
        return "discard until regenerated"
    margin = row.get("Reliability margin", np.nan)
    if pd.isna(margin) or margin <= 0.0:
        return "discard as evidence; keep as diagnostic"
    if margin >= 0.04 and row["Model"] in {"Riemann", "LR_alpha", "LR_theta"}:
        return "strong/moderate thesis evidence"
    if margin >= 0.02:
        return "exploratory supporting evidence"
    return "weak exploratory evidence"


def evidence_pyramid(evidence: pd.DataFrame) -> pd.DataFrame:
    def tier(rec: str) -> str:
        if rec == "strong/moderate thesis evidence":
            return "strongest evidence"
        if rec in {"exploratory supporting evidence", "weak exploratory evidence"}:
            return "weaker evidence"
        return "suspect / do not use as support"

    out = evidence[["Display", "Model", "Mean accuracy", "Chance/permutation status", "Confound risk level", "Thesis-use recommendation"]].copy()
    out["Evidence tier"] = out["Thesis-use recommendation"].map(tier)
    return out.sort_values(["Evidence tier", "Mean accuracy"], ascending=[True, False], na_position="last")


def convergence_table(evidence: pd.DataFrame, paths, task, task_tag: str) -> pd.DataFrame:
    attr = attribution_summary(paths, task, task_tag)
    weights = evidence.set_index("Model")["Reliability margin"].clip(lower=0.0).fillna(0.0)
    rows = []
    for _, row in attr.iterrows():
        model = row["Model"]
        spatial = row["Spatial vector"]
        if spatial is None or np.isscalar(spatial):
            frontal = posterior = 0.0
        else:
            spatial = np.asarray(spatial, dtype=float)
            lookup = {name: i for i, name in enumerate(ALL_CHANNELS)}
            frontal = float(sum(spatial[lookup[ch]] for ch in THETA_CHANNELS if ch in lookup))
            posterior = float(sum(spatial[lookup[ch]] for ch in ALPHA_CHANNELS if ch in lookup))
        rows.append(
            {
                "Model": model,
                "Display": MODEL_LABELS[model],
                "Reliability weight": float(weights.get(model, 0.0)),
                "Frontal/theta-region alignment": frontal,
                "Posterior/alpha-region alignment": posterior,
                "Attribution availability": row["Attribution availability"],
                "Attribution stability": row["Attribution stability"],
            }
        )
    df = pd.DataFrame(rows)
    total_weight = df["Reliability weight"].sum()
    if total_weight > 0:
        df["Weighted convergence contribution"] = df["Reliability weight"] * (
            df["Frontal/theta-region alignment"] + df["Posterior/alpha-region alignment"]
        ) / 2.0
    else:
        df["Weighted convergence contribution"] = 0.0
    return df.sort_values("Weighted convergence contribution", ascending=False)


def synthesis_text(context: ExplainabilityContext, evidence: pd.DataFrame, convergence: pd.DataFrame) -> str:
    usable = evidence[evidence["OOF usable"].fillna(False)].copy()
    top = usable.sort_values("Mean accuracy", ascending=False).head(3)
    top_names = ", ".join(top["Display"].tolist()) if not top.empty else "no models with fresh OOF evidence"
    conv = convergence["Weighted convergence contribution"].sum()
    if context.task_tag == "c_vs_ip":
        verdict = (
            "For C vs IP, the defensible reading is a cautious positive result: the signal is weak, "
            "but the best evidence should privilege distributed covariance and simple alpha/theta baselines "
            "over visually attractive deep saliency maps."
        )
    else:
        verdict = (
            "For IM vs IP, the defensible reading is exploratory: because both classes are incongruent, "
            "performance and explainability must clear a higher bar before being interpreted as perceptual-outcome evidence."
        )
    return (
        f"{context.hypothesis}\n\n"
        f"Fresh-artifact ranking is led by {top_names}. The performance-weighted convergence score is {conv:.3f}; "
        "interpret this as a compact diagnostic of agreement with the frontal-theta/posterior-alpha sensor hypotheses, "
        "not as a source-localization result.\n\n"
        f"{verdict} {context.interpretation_default}"
    )


def save_tables(paths, task_tag: str, evidence: pd.DataFrame, pyramid: pd.DataFrame, convergence: pd.DataFrame) -> None:
    paths.tables_dir.mkdir(parents=True, exist_ok=True)
    export = evidence.copy()
    for col in export.columns:
        if export[col].dtype == object:
            export[col] = export[col].map(lambda x: str(x) if isinstance(x, Path) else x)
    export.to_csv(paths.tables_dir / f"explainability_evidence_table_{task_tag}.csv", index=False)
    pyramid.to_csv(paths.tables_dir / f"explainability_evidence_pyramid_{task_tag}.csv", index=False)
    convergence.to_csv(paths.tables_dir / f"explainability_convergence_{task_tag}.csv", index=False)


def plot_evidence_overview(evidence: pd.DataFrame, task_tag: str, *, save_path: Path | None = None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plot_df = evidence.dropna(subset=["Mean accuracy"]).copy()
    if plot_df.empty:
        print("No fresh accuracy evidence available to plot.")
        return None
    plot_df = plot_df.sort_values("Mean accuracy", ascending=True)
    colors = [MODEL_COLORS.get(m, "#777777") for m in plot_df["Model"]]
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.barh(plot_df["Display"], plot_df["Mean accuracy"], color=colors, alpha=0.88)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.2, label="nominal chance")
    ax.set_xlim(0.45, max(0.7, float(plot_df["Mean accuracy"].max()) + 0.04))
    ax.set_xlabel("Mean subject OOF accuracy")
    ax.set_title(f"Performance-weighted explainability evidence: {task_tag}", fontweight="bold")
    ax.legend(loc="lower right")
    sns.despine(fig=fig)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_convergence(convergence: pd.DataFrame, task_tag: str, *, save_path: Path | None = None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plot_df = convergence.copy()
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    width = 0.38
    x = np.arange(len(plot_df))
    ax.bar(x - width / 2, plot_df["Frontal/theta-region alignment"], width, label="frontal/theta-region")
    ax.bar(x + width / 2, plot_df["Posterior/alpha-region alignment"], width, label="posterior/alpha-region")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["Display"], rotation=35, ha="right")
    ax.set_ylabel("Attribution mass / structural prior mass")
    ax.set_title(f"Cross-model sensor-region convergence: {task_tag}", fontweight="bold")
    ax.legend()
    sns.despine(fig=fig)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
