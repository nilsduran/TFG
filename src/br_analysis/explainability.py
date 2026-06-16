from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

OOF_CACHE_VERSION = "stratified_group_kfold_holdout_v1"
OOF_CV_STRATEGY = "stratified_group_kfold_development_only"

MODEL_ORDER = [
    "LR_theta",
    "LR_alpha",
    "Riemann",
    "Experiment_Guided",
    "SVC_Clustering",
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
    "SVC_Clustering": "SVC clustering",
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
    "SVC_Clustering": "classical SVC / cluster-restricted",
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
    "SVC_Clustering": "#17becf",
    "ST_GNN": "#b2182b",
    "ST_EEGFormer": "#756bb1",
    "LaBraM": "#c51b7d",
}

RELIABLE_SUBJECT_THRESHOLD = 0.55
RELIABLE_SUBJECT_MIN_MODELS = 3
RELIABLE_SUBJECT_MIN_MODEL_FRACTION = 0.30
RELIABLE_COHORT_DEFAULT_N = 5
RELIABLE_COHORT_DIAGNOSTIC_MIN_N = 5

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

SCALP_FRONTAL = [
    "Fp1", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6"
]
SCALP_TEMPORAL = [
    "FT7", "FT8", "T7", "T8", "TP7", "TP8"
]
SCALP_PARIETAL = [
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8"
]
SCALP_OCCIPITAL = [
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"
]
CLUSTER_CHANNELS = [
    "Fz", "FCz", "FC3", "FC1", "FC2", "C3", "C1", "Cz", "C2",
    "CP3", "CP1", "CPz", "CP2", "CP4", "P5", "P3", "P1", "Pz",
    "P2", "P4", "P6", "P8", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
]
CLUSTER_CHANNELS_IM_IP = [
    "Fz", "FCz", "Cz", "TP7", "CP5", "CP3", "CP1", "P7", "P5", "P3"
]

REFERENCE_N_TIMES = 192

MODEL_CHANNELS = {
    "LR_theta": ["FCz"],
    "LR_alpha": ["Oz"],
    "Riemann": THETA_CHANNELS + ALPHA_CHANNELS,
    "Experiment_Guided": CLUSTER_CHANNELS,
    "EEGNet_Agnostic": ALL_CHANNELS,
    "EEGNet_Clustering": CLUSTER_CHANNELS,
    "SVC_Clustering": CLUSTER_CHANNELS,
    "ST_GNN": ALL_CHANNELS,
    "ST_EEGFormer": ALL_CHANNELS,
    "LaBraM": ALL_CHANNELS,
}


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
    comparison_cache_dir = paths.analysis_root / "cross_model_comparison" / "cache"
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
            "SVC_Clustering": "oof_predictions_c_vs_ip_svc_clustering.npz",
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
            "SVC_Clustering": "oof_predictions_im_vs_ip_svc_clustering.npz",
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
        "EEGNet_Agnostic": comparison_cache_dir / files["EEGNet_Agnostic"],
        "EEGNet_Clustering": comparison_cache_dir / files["EEGNet_Clustering"],
        "SVC_Clustering": comparison_cache_dir / files["SVC_Clustering"],
        "ST_GNN": comparison_cache_dir / files["ST_GNN"],
        "ST_EEGFormer": comparison_cache_dir / files["ST_EEGFormer"],
        "LaBraM": comparison_cache_dir / files["LaBraM"],
    }


def final_holdout_paths(paths, task_tag: str) -> dict[str, Path]:
    if task_tag == "c_vs_ip":
        return {
            "Experiment_Guided": paths.analysis_root / "experiment_guided_config_c_ip" / "tables" / "experiment_guided_config_c_ip_final_holdout_ONCE.csv",
            "EEGNet_Agnostic": paths.analysis_root / "eegnet" / "tables" / "final_holdout_c_vs_ip_eegnet.csv",
            "EEGNet_Clustering": paths.analysis_root / "eegnet" / "tables" / "final_holdout_c_vs_ip_eegnet.csv",
            "SVC_Clustering": paths.analysis_root / "svc_clustering" / "tables" / "svc_clustering_c_vs_ip_final_holdout_ONCE.csv",
            "ST_GNN": paths.analysis_root / "c_vs_i_stgnn" / "tables" / "final_holdout_c_vs_ip_stgnn.csv",
            "ST_EEGFormer": paths.analysis_root / "c_vs_i_steegformer" / "tables" / "final_holdout_c_vs_ip_steegformer.csv",
            "LaBraM": paths.analysis_root / "c_vs_i_labram" / "tables" / "final_holdout_c_vs_ip_labram.csv",
        }
    if task_tag == "im_vs_ip":
        return {
            "Experiment_Guided": paths.analysis_root / "experiment_guided_config_im_ip" / "tables" / "experiment_guided_config_im_ip_final_holdout_ONCE.csv",
            "EEGNet_Agnostic": paths.analysis_root / "eegnet_im_vs_ip" / "tables" / "final_holdout_im_vs_ip_eegnet.csv",
            "EEGNet_Clustering": paths.analysis_root / "eegnet_im_vs_ip" / "tables" / "final_holdout_im_vs_ip_eegnet.csv",
            "SVC_Clustering": paths.analysis_root / "svc_clustering" / "tables" / "svc_clustering_im_vs_ip_final_holdout_ONCE.csv",
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


def _balanced_binary_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    scores = []
    for label in [0, 1]:
        mask = y_true == label
        if mask.any():
            scores.append(float((y_pred[mask] == y_true[mask]).mean()))
    return float(np.mean(scores)) if scores else np.nan


def _task_label(task_tag: str) -> str:
    if " vs " in task_tag:
        return task_tag
    if "_vs_" in task_tag:
        left, right = task_tag.split("_vs_", 1)
        return f"{left.upper()} vs {right.upper()}"
    return task_tag


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
                    row[model_name] = _balanced_binary_accuracy(y_true[valid], (y_prob[valid] >= 0.5).astype(int))
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["subject"] + MODEL_ORDER)
    present = [c for c in MODEL_ORDER if c in df.columns]
    if present:
        df = df.dropna(subset=present, how="all")
    return df, pd.DataFrame(status_rows)


def select_reliable_subject_cohort(
    df_subject: pd.DataFrame,
    *,
    threshold: float = RELIABLE_SUBJECT_THRESHOLD,
    min_models_above: int = RELIABLE_SUBJECT_MIN_MODELS,
    min_model_fraction: float = RELIABLE_SUBJECT_MIN_MODEL_FRACTION,
    default_n: int = RELIABLE_COHORT_DEFAULT_N,
    diagnostic_min_n: int = RELIABLE_COHORT_DIAGNOSTIC_MIN_N,
) -> pd.DataFrame:
    """Rank and select one shared subject cohort for all model interpretations.

    The selection is deliberately model-agnostic: each subject receives one rank
    based on performance across all usable model columns. The resulting cohort is
    then reused for every model so cross-model map differences are not driven by
    different subject sets.
    """

    if df_subject.empty or "subject" not in df_subject.columns:
        return pd.DataFrame(
            columns=[
                "subject",
                "mean_accuracy",
                "n_models_available",
                "n_models_above_0_55",
                "fraction_models_above_0_55",
                "meets_reliable_subject_rule",
                "selected_rank",
                "selected_for_interpretability",
                "selected_cohort_role",
                "selection_reason",
            ]
        )

    model_cols = [model for model in MODEL_ORDER if model in df_subject.columns]
    out = df_subject.copy()
    if not model_cols:
        out["mean_accuracy"] = np.nan
        out["n_models_available"] = 0
        out["n_models_above_0_55"] = 0
        out["fraction_models_above_0_55"] = np.nan
    else:
        numeric = out[model_cols].apply(pd.to_numeric, errors="coerce")
        out[model_cols] = numeric
        out["mean_accuracy"] = numeric.mean(axis=1, skipna=True)
        out["n_models_available"] = numeric.notna().sum(axis=1).astype(int)
        out["n_models_above_0_55"] = numeric.ge(threshold).sum(axis=1).astype(int)
        out["fraction_models_above_0_55"] = np.divide(
            out["n_models_above_0_55"],
            out["n_models_available"].replace(0, np.nan),
        )

    out = out.sort_values(
        ["n_models_above_0_55", "mean_accuracy", "n_models_available", "subject"],
        ascending=[False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    out["candidate_rank"] = np.arange(1, len(out) + 1)

    out["meets_reliable_subject_rule"] = (
        (out["n_models_available"] > 0)
        & (out["mean_accuracy"] >= threshold)
        & (
            (out["n_models_above_0_55"] >= min_models_above)
            | (out["fraction_models_above_0_55"] >= min_model_fraction)
        )
    )
    reliable = out[out["meets_reliable_subject_rule"]].copy()
    if len(reliable) >= default_n:
        selected_subjects = reliable.head(default_n)["subject"].tolist()
        reason = f"top_{default_n}_reliable_subjects"
    elif len(reliable) >= diagnostic_min_n:
        selected_subjects = reliable["subject"].tolist()
        reason = f"{len(selected_subjects)}_reliable_subjects"
    elif len(reliable) > 0:
        reliable_subjects = reliable["subject"].tolist()
        supplement_n = min(diagnostic_min_n, len(out)) - len(reliable_subjects)
        supplement = out[
            ~out["subject"].astype(str).isin([str(s) for s in reliable_subjects])
        ].head(max(supplement_n, 0))["subject"].tolist()
        selected_subjects = reliable_subjects + supplement
        reason = "reliable_core_plus_low_confidence_diagnostic_subjects"
    else:
        selected_subjects = out.head(min(diagnostic_min_n, len(out)))["subject"].tolist()
        reason = "no_reliable_subjects_ranked_diagnostic_fallback"

    selected_lookup = {subject: i + 1 for i, subject in enumerate(selected_subjects)}
    out["selected_rank"] = out["subject"].map(selected_lookup)
    out["selected_for_interpretability"] = out["selected_rank"].notna()
    out["selected_cohort_role"] = np.where(
        out["selected_for_interpretability"],
        np.where(out["meets_reliable_subject_rule"], "reliable", "low_confidence_diagnostic"),
        "not_selected",
    )
    out["selection_reason"] = np.where(out["selected_for_interpretability"], reason, "not_selected")

    ordered_cols = [
        "subject",
        *model_cols,
        "mean_accuracy",
        "n_models_available",
        "n_models_above_0_55",
        "fraction_models_above_0_55",
        "meets_reliable_subject_rule",
        "candidate_rank",
        "selected_rank",
        "selected_for_interpretability",
        "selected_cohort_role",
        "selection_reason",
    ]
    return out[ordered_cols]


def selected_subject_ids(cohort: pd.DataFrame) -> list[str]:
    if cohort.empty or "selected_for_interpretability" not in cohort.columns:
        return []
    selected = cohort[cohort["selected_for_interpretability"]].copy()
    selected = selected.sort_values("selected_rank", na_position="last")
    return selected["subject"].astype(str).tolist()


def get_model_specific_subjects(
    df_subject: pd.DataFrame,
    cohort_subjects: Sequence[str],
    threshold: float,
    fallback_threshold: float | None = 0.51,
) -> dict[str, list[str]]:
    """Get a dictionary mapping each model to its valid subjects from cohort_subjects.

    A subject is valid for a model if the model's accuracy on that subject is >= threshold.
    If no subjects in cohort_subjects meet this threshold for a model, we fall back to
    subjects where accuracy is >= fallback_threshold (if provided).
    Prior-based classical models (LR_theta, LR_alpha, Riemann, etc.) are always mapped to
    all cohort_subjects since they do not depend on subject-level estimation.
    """
    mapping = {}
    for model in MODEL_ORDER:
        if model in df_subject.columns:
            valid_subs = []
            for sub in cohort_subjects:
                acc = df_subject.loc[df_subject["subject"] == sub, model].values
                if len(acc) > 0 and not pd.isna(acc[0]) and acc[0] >= threshold:
                    valid_subs.append(sub)
            if not valid_subs and fallback_threshold is not None:
                for sub in cohort_subjects:
                    acc = df_subject.loc[df_subject["subject"] == sub, model].values
                    if len(acc) > 0 and not pd.isna(acc[0]) and acc[0] >= fallback_threshold:
                        valid_subs.append(sub)
            mapping[model] = valid_subs
        else:
            mapping[model] = list(cohort_subjects)
    return mapping


def performance_summary_for_subjects(df_subject: pd.DataFrame, subjects: Sequence[str]) -> pd.DataFrame:
    if not subjects:
        return performance_summary(df_subject)
    filtered = df_subject[df_subject["subject"].astype(str).isin([str(s) for s in subjects])].copy()
    return performance_summary(filtered)


def load_oof_predictions(
    paths,
    task,
    task_tag: str,
    *,
    subjects: Sequence[str] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    subject_ids = [str(s) for s in subjects] if subjects is not None else safe_subject_ids(task)
    out: dict[str, dict[str, np.ndarray]] = {}
    for model_name, cache_file in cache_paths(paths, task_tag).items():
        fresh, _ = oof_cache_status(cache_file)
        if not fresh:
            continue
        y_true_all = []
        y_prob_all = []
        with np.load(cache_file, allow_pickle=True) as data:
            for sub in subject_ids:
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


def _normalize_spatial_density(values: np.ndarray, n_available_channels: int) -> np.ndarray:
    arr = _normalize(values)
    support_scale = max(1, int(n_available_channels)) / len(ALL_CHANNELS)
    return arr * support_scale


def _normalize_temporal_density(values: np.ndarray, reference_n_times: int = REFERENCE_N_TIMES) -> np.ndarray:
    arr = _normalize(values)
    support_scale = len(arr) / max(1, int(reference_n_times))
    return arr * support_scale


def _channel_vector(channels: Sequence[str], weights: Sequence[float]) -> np.ndarray:
    values = np.zeros(len(ALL_CHANNELS), dtype=float)
    lookup = {name.upper(): i for i, name in enumerate(ALL_CHANNELS)}
    for ch, weight in zip(channels, weights):
        idx = lookup.get(ch.upper())
        if idx is not None:
            values[idx] += float(weight)
    return _normalize_spatial_density(values, len(channels))


def _project_channel_values(channels: Sequence[str], values: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(values, dtype=float).ravel()
    if len(channels) != len(arr):
        return None
    projected = np.zeros(len(ALL_CHANNELS), dtype=float)
    lookup = {name.upper(): i for i, name in enumerate(ALL_CHANNELS)}
    for ch, value in zip(channels, arr):
        idx = lookup.get(ch.upper())
        if idx is not None:
            projected[idx] += float(value)
    return _normalize_spatial_density(projected, len(channels))


def _candidate_channels_for_model(model_name: str, n_values: int, task_tag: str = "c_vs_ip") -> list[str] | None:
    cluster_ch = CLUSTER_CHANNELS if task_tag == "c_vs_ip" else CLUSTER_CHANNELS_IM_IP
    
    if model_name in {"Experiment_Guided", "EEGNet_Clustering", "SVC_Clustering"}:
        if len(cluster_ch) == n_values:
            return list(cluster_ch)
            
    channels = MODEL_CHANNELS.get(model_name)
    if channels is not None and len(channels) == n_values:
        return list(channels)
    if n_values == len(ALL_CHANNELS):
        return list(ALL_CHANNELS)
    if n_values == len(cluster_ch):
        return list(cluster_ch)
    if model_name == "ST_EEGFormer":
        try:
            from .pretrained import ST_EEGFORMER_CHANNELS

            return list(ST_EEGFORMER_CHANNELS[:n_values])
        except Exception:
            return None
    return None


def canonical_attribution_priors(task_tag: str = "c_vs_ip") -> dict[str, dict[str, object]]:
    cluster_ch = CLUSTER_CHANNELS if task_tag == "c_vs_ip" else CLUSTER_CHANNELS_IM_IP
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
            "spatial": _channel_vector(cluster_ch, [1.0] * len(cluster_ch)),
            "temporal_peak": 0.75,
            "temporal_width": 0.75,
            "stability": "structural_with_circularity_risk",
            "pattern": "hypothesis-guided spatiotemporal cluster restriction",
        },
        "SVC_Clustering": {
            "source": "model prior / cluster restriction",
            "availability": "available_feature_definition",
            "spatial": _channel_vector(cluster_ch, [1.0] * len(cluster_ch)),
            "temporal_peak": 0.75,
            "temporal_width": 0.75,
            "stability": "structural_with_circularity_risk",
            "pattern": "SVC over the shared cluster-restricted feature space",
        },
    }


def _temporal_prior_vector(
    *,
    peak: float,
    width: float,
    n_times: int = 192,
    analysis_window: tuple[float, float] = (0.0, 1.5),
) -> np.ndarray:
    times = np.linspace(float(analysis_window[0]), float(analysis_window[1]), int(n_times))
    sigma = max(float(width), 1e-6)
    values = np.exp(-0.5 * ((times - float(peak)) / sigma) ** 2)
    return _normalize_temporal_density(values, n_times)


def _padded_temporal_average(vectors: list[np.ndarray]) -> np.ndarray | None:
    if not vectors:
        return None
    max_len = max(len(t) for t in vectors)
    padded = np.full((len(vectors), max_len), np.nan)
    for i, t in enumerate(vectors):
        padded[i, : len(t)] = t
    return _normalize_temporal_density(np.nanmean(padded, axis=0), max_len)


def load_deep_attribution_summaries(
    paths,
    task,
    task_tag: str,
    *,
    subjects: Sequence[str] | dict[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    rows = []
    for model_name in ["EEGNet_Agnostic", "EEGNet_Clustering", "ST_GNN", "ST_EEGFormer", "LaBraM"]:
        if isinstance(subjects, dict):
            subject_ids = [str(s) for s in subjects.get(model_name, [])]
        else:
            subject_ids = [str(s) for s in subjects] if subjects is not None else safe_subject_ids(task)
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
                for sub in subject_ids:
                    sp_key = f"{sub}_spatial_saliency"
                    te_key = f"{sub}_temporal_saliency"
                    if sp_key in data:
                        sp = np.asarray(data[sp_key], dtype=float).ravel()
                        channels = _candidate_channels_for_model(model_name, len(sp), task_tag)
                        projected = _project_channel_values(channels, sp) if channels is not None else None
                        if projected is not None:
                            spatial.append(projected)
                    if te_key in data:
                        te = np.asarray(data[te_key], dtype=float).ravel()
                        temporal.append(_normalize_temporal_density(te, max(len(te), 1)))
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
                row["Temporal vector"] = _padded_temporal_average(temporal)
            if fresh and not spatial and not temporal:
                row.update(
                    {
                        "Attribution availability": "prediction_only_no_saliency_export",
                        "Attribution stability": "not_available",
                        "Attribution note": (
                            "fresh OOF predictions are available, but this model cache does not export "
                            "spatial_saliency/temporal_saliency arrays"
                        ),
                    }
                )
        rows.append(row)
    return pd.DataFrame(rows)


def attribution_summary(
    paths,
    task,
    task_tag: str,
    *,
    subjects: Sequence[str] | None = None,
) -> pd.DataFrame:
    rows = []
    for model, info in canonical_attribution_priors(task_tag).items():
        rows.append(
            {
                "Model": model,
                "Attribution availability": info["availability"],
                "Attribution source": info["source"],
                "Attribution stability": info["stability"],
                "Attribution note": info["pattern"],
                "Spatial vector": info["spatial"],
                "Temporal vector": _temporal_prior_vector(
                    peak=float(info["temporal_peak"]),
                    width=float(info["temporal_width"]),
                    analysis_window=task.analysis_window,
                ),
            }
        )
    deep = load_deep_attribution_summaries(paths, task, task_tag, subjects=subjects)
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
    threshold = 0.53 if task_tag == "im_vs_ip" else 0.55
    cohort = select_reliable_subject_cohort(df_subject, threshold=threshold)
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
    table["Model"] = pd.Categorical(table["Model"], categories=MODEL_ORDER, ordered=True)
    table = table.sort_values("Model").reset_index(drop=True)
    return table, {"subjects": df_subject, "cache_status": df_cache, "selected_cohort": cohort}


def selected_cohort_model_evidence(
    evidence: pd.DataFrame,
    df_subject: pd.DataFrame,
    subjects: Sequence[str],
) -> pd.DataFrame:
    """Recompute model accuracy summaries on the selected shared cohort."""

    selected_perf = performance_summary_for_subjects(df_subject, subjects)
    selected_perf = selected_perf.rename(
        columns={
            "Mean accuracy": "Selected cohort mean accuracy",
            "Std accuracy": "Selected cohort std accuracy",
            "N subjects": "Selected cohort N subjects",
            "Above 0.5": "Selected cohort above 0.5",
        }
    )
    out = evidence.merge(
        selected_perf[
            [
                "Model",
                "Selected cohort mean accuracy",
                "Selected cohort std accuracy",
                "Selected cohort N subjects",
                "Selected cohort above 0.5",
            ]
        ],
        on="Model",
        how="left",
    )
    out["Selected cohort reliability weight"] = (
        out["Selected cohort mean accuracy"].astype(float) - 0.5
    ).clip(lower=0.0).fillna(0.0)
    return out


def _confound_risk(row: pd.Series) -> str:
    model = row["Model"]
    if not bool(row.get("OOF usable", False)):
        return "high: missing/stale validation artifact"
    margin = row.get("Reliability margin", np.nan)
    if model in {"Experiment_Guided", "EEGNet_Clustering", "SVC_Clustering"}:
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


def model_interpretation_table(evidence: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in evidence.sort_values("Model", key=lambda s: s.map({m: i for i, m in enumerate(MODEL_ORDER)})).iterrows():
        model = row["Model"]
        selected_acc = row.get("Selected cohort mean accuracy", np.nan)
        weight = row.get("Selected cohort reliability weight", np.nan)
        attr = str(row.get("Attribution availability", "unknown"))
        if pd.isna(selected_acc):
            evidence_read = "No selected-cohort validation estimate is available in this runtime."
        elif float(selected_acc) <= 0.5:
            evidence_read = "Do not interpret as evidence; selected-cohort decoding is at or below chance."
        elif float(selected_acc) < RELIABLE_SUBJECT_THRESHOLD:
            evidence_read = "Weak diagnostic only; selected-cohort decoding is above chance but below the reliability threshold."
        else:
            evidence_read = "Interpretable with caution; selected-cohort decoding clears the reliability threshold."

        if attr == "available_feature_definition":
            attribution_read = "Structural/feature-defined attribution, normalized by model channel support."
        elif attr == "available_saliency_cache":
            attribution_read = "Cached model saliency is available and projected into the shared channel layout."
        elif attr == "prediction_only_no_saliency_export":
            attribution_read = "Predictions are available, but this cache does not export saliency arrays; no DL map should be overread."
        else:
            attribution_read = "Attribution artifact is missing or stale."

        rows.append(
            {
                "Model": MODEL_LABELS.get(model, model),
                "Family": row.get("Family", ""),
                "Selected cohort accuracy": selected_acc,
                "Reliability weight": weight,
                "Attribution status": attr,
                "Evidence interpretation": evidence_read,
                "Attribution interpretation": attribution_read,
            }
        )
    return pd.DataFrame(rows)


def convergence_table(
    evidence: pd.DataFrame,
    paths,
    task,
    task_tag: str,
    *,
    subjects: Sequence[str] | None = None,
    weight_col: str = "Selected cohort reliability weight",
    four_regions: bool = False,
) -> pd.DataFrame:
    attr = attribution_summary(paths, task, task_tag, subjects=subjects)
    if weight_col in evidence.columns:
        weights = evidence.set_index("Model")[weight_col].clip(lower=0.0).fillna(0.0)
    else:
        weights = evidence.set_index("Model")["Reliability margin"].clip(lower=0.0).fillna(0.0)
    rows = []
    lookup = {name.upper(): i for i, name in enumerate(ALL_CHANNELS)}
    for _, row in attr.iterrows():
        model = row["Model"]
        spatial = row["Spatial vector"]
        if spatial is None or np.isscalar(spatial):
            frontal = temporal = parietal = occipital = 0.0
        else:
            spatial = np.asarray(spatial, dtype=float)
            total = np.abs(spatial).sum()
            if total > 0:
                spatial = spatial / total
            frontal = float(sum(spatial[lookup[ch.upper()]] for ch in SCALP_FRONTAL if ch.upper() in lookup))
            temporal = float(sum(spatial[lookup[ch.upper()]] for ch in SCALP_TEMPORAL if ch.upper() in lookup))
            parietal = float(sum(spatial[lookup[ch.upper()]] for ch in SCALP_PARIETAL if ch.upper() in lookup))
            occipital = float(sum(spatial[lookup[ch.upper()]] for ch in SCALP_OCCIPITAL if ch.upper() in lookup))
        
        weight = float(weights.get(model, 0.0))
        r = {
            "Model": model,
            "Display": MODEL_LABELS[model],
            "Reliability weight": weight,
            "Attribution availability": row["Attribution availability"],
            "Attribution stability": row["Attribution stability"],
            "Frontal alignment": frontal,
            "Temporal alignment": temporal,
            "Parietal alignment": parietal,
            "Occipital alignment": occipital,
        }
        rows.append(r)
    df = pd.DataFrame(rows)
    total_weight = df["Reliability weight"].sum()
    if total_weight > 0:
        df["Weighted convergence contribution"] = df["Reliability weight"] * (
            df["Frontal alignment"] + df["Occipital alignment"]
        ) / 2.0
    else:
        df["Weighted convergence contribution"] = 0.0
    df["Model"] = pd.Categorical(df["Model"], categories=MODEL_ORDER, ordered=True)
    df = df.sort_values("Model").reset_index(drop=True)
    return df


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


def save_tables(
    paths,
    task_tag: str,
    evidence: pd.DataFrame,
    pyramid: pd.DataFrame,
    convergence: pd.DataFrame,
    *,
    selected_cohort: pd.DataFrame | None = None,
    consensus: pd.DataFrame | None = None,
) -> None:
    paths.tables_dir.mkdir(parents=True, exist_ok=True)
    export = evidence.copy()
    for col in export.columns:
        if export[col].dtype == object:
            export[col] = export[col].map(lambda x: str(x) if isinstance(x, Path) else x)
    export.to_csv(paths.tables_dir / f"explainability_evidence_table_{task_tag}.csv", index=False)
    pyramid.to_csv(paths.tables_dir / f"explainability_evidence_pyramid_{task_tag}.csv", index=False)
    convergence.to_csv(paths.tables_dir / f"explainability_convergence_{task_tag}.csv", index=False)
    if selected_cohort is not None:
        selected_cohort.to_csv(paths.tables_dir / f"selected_interpretability_subjects_{task_tag}.csv", index=False)
    if consensus is not None:
        consensus.to_csv(paths.tables_dir / f"explainability_consensus_diagnostic_{task_tag}.csv", index=False)


def plot_evidence_overview(evidence: pd.DataFrame, task_tag: str, *, save_path: Path | None = None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plot_df = evidence.dropna(subset=["Selected cohort mean accuracy"]).copy()
    if plot_df.empty:
        print("No fresh accuracy evidence available to plot.")
        return None
    plot_df = plot_df.sort_values("Selected cohort mean accuracy", ascending=True)
    colors = [MODEL_COLORS.get(m, "#777777") for m in plot_df["Model"]]
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.barh(plot_df["Display"], plot_df["Selected cohort mean accuracy"], color=colors, alpha=0.88)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.2, label="nominal chance")
    ax.set_xlim(0.45, max(0.6, float(plot_df["Selected cohort mean accuracy"].max()) + 0.04))
    ax.set_xlabel("Selected cohort mean subject OOF accuracy")
    ax.set_title(f"Selected-cohort explainability performance overview: {_task_label(task_tag)}", fontweight="bold")
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
    
    regions = [
        ("Frontal alignment", "Frontal region", "#2878b5"),
        ("Temporal alignment", "Temporal region", "#756bb1"),
        ("Parietal alignment", "Parietal region", "#1b9e77"),
        ("Occipital alignment", "Occipital region", "#d95f02"),
    ]

    x = np.arange(len(plot_df))
    bottom = np.zeros(len(plot_df))
    
    for col, label, color in regions:
        vals = plot_df[col].fillna(0.0).to_numpy()
        ax.bar(x, vals, bottom=bottom, label=label, color=color, alpha=0.85, width=0.55)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["Display"], rotation=30, ha="right")
    ax.set_ylabel("Attribution mass proportion (sum = 1.0)")
    ax.set_title(f"Cross-model sensor-region convergence: {_task_label(task_tag)}", fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    sns.despine(fig=fig)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def _safe_spatial_vector(value) -> np.ndarray | None:
    if value is None or np.isscalar(value):
        return None
    arr = np.asarray(value, dtype=float).ravel()
    if arr.size != len(ALL_CHANNELS):
        return None
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_temporal_vector(value) -> np.ndarray | None:
    if value is None or np.isscalar(value):
        return None
    arr = np.asarray(value, dtype=float).ravel()
    if arr.size == 0:
        return None
    target_size = 192
    if arr.size != target_size:
        x_old = np.linspace(0, 1, arr.size)
        x_new = np.linspace(0, 1, target_size)
        arr = np.interp(x_new, x_old, arr)
    arr = _normalize(arr)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _time_axis(n_times: int, task) -> np.ndarray:
    start, stop = task.analysis_window
    return np.linspace(float(start), float(stop), int(n_times))


def _cosine_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return np.nan
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else np.nan


def consensus_diagnostic(
    evidence: pd.DataFrame,
    paths,
    task,
    task_tag: str,
    *,
    subjects: Sequence[str] | dict[str, Sequence[str]],
    df_subject: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, np.ndarray | float | None]]:
    """Build exploratory weighted/unweighted consensus summaries for the shared cohort."""

    attr = attribution_summary(paths, task, task_tag, subjects=subjects)
    if "Selected cohort reliability weight" in evidence.columns:
        weights = evidence.set_index("Model")["Selected cohort reliability weight"].fillna(0.0)
    else:
        weights = evidence.set_index("Model")["Reliability margin"].clip(lower=0.0).fillna(0.0)

    spatial_rows = []
    temporal_rows = []
    summary_rows = []
    for _, row in attr.iterrows():
        model = row["Model"]
        spatial = _safe_spatial_vector(row["Spatial vector"])
        temporal = _safe_temporal_vector(row["Temporal vector"])
        weight = float(weights.get(model, 0.0))
        if spatial is not None:
            spatial_rows.append((model, weight, spatial))
        if temporal is not None:
            # Exclude prior-based models with Gaussian temporal priors and ST-EEGFormer from the temporal consensus
            if model not in {"LR_theta", "LR_alpha", "Riemann", "Experiment_Guided", "SVC_Clustering", "ST_EEGFormer"}:
                temporal_rows.append((model, weight, temporal))
        peak_time = np.nan
        if temporal is not None:
            times = _time_axis(len(temporal), task)
            peak_time = float(times[int(np.nanargmax(np.abs(temporal)))])
        summary_rows.append(
            {
                "Model": model,
                "Display": MODEL_LABELS.get(model, model),
                "Selected cohort reliability weight": weight,
                "Attribution availability": row["Attribution availability"],
                "Included in weighted consensus": bool(weight > 0 and spatial is not None),
                "Temporal peak time (s)": peak_time,
            }
        )

    def aggregate(rows: list[tuple[str, float, np.ndarray]], weighted: bool) -> np.ndarray | None:
        if not rows:
            return None
        lengths = [len(item[2]) for item in rows]
        if len(set(lengths)) > 1:
            target_size = 192
            resampled = []
            for item in rows:
                arr = item[2]
                if len(arr) == target_size:
                    resampled.append(arr)
                else:
                    x_old = np.linspace(0, 1, len(arr))
                    x_new = np.linspace(0, 1, target_size)
                    resampled.append(np.interp(x_new, x_old, arr))
            arrays = np.vstack(resampled)
        else:
            arrays = np.vstack([item[2] for item in rows])
        if weighted:
            w = np.asarray([item[1] for item in rows], dtype=float)
            if float(w.sum()) <= 0:
                return None
            return np.average(arrays, axis=0, weights=w)
        return np.nanmean(arrays, axis=0)

    weighted_spatial = aggregate(spatial_rows, weighted=True)
    weighted_temporal = aggregate(temporal_rows, weighted=True)

    if isinstance(subjects, dict):
        all_cohort_subs = set()
        for subs in subjects.values():
            all_cohort_subs.update(subs)
        flat_subjects = list(all_cohort_subs)
    else:
        flat_subjects = list(subjects)

    selected_df = df_subject[df_subject["subject"].astype(str).isin([str(s) for s in flat_subjects])].copy()
    model_cols = [model for model in MODEL_ORDER if model in selected_df.columns]
    boot_std = np.nan
    if len(selected_df) >= 5 and model_cols:
        rng = np.random.default_rng(42)
        boot_scores = []
        numeric = selected_df[model_cols].apply(pd.to_numeric, errors="coerce")
        for _ in range(500):
            idx = rng.integers(0, len(numeric), len(numeric))
            sampled = numeric.iloc[idx]
            boot_scores.append(float(np.nanmean(sampled.to_numpy())))
        boot_std = float(np.nanstd(boot_scores))

    summary = pd.DataFrame(summary_rows)
    summary["Bootstrap selected-subject mean-accuracy std"] = boot_std
    summary["Model"] = pd.Categorical(summary["Model"], categories=MODEL_ORDER, ordered=True)
    summary = summary.sort_values("Model").reset_index(drop=True)

    vectors: dict[str, np.ndarray | float | None] = {
        "weighted_spatial": weighted_spatial,
        "weighted_temporal": weighted_temporal,
        "bootstrap_std": boot_std,
    }
    return summary, vectors


def _plot_topomap(ax, values: np.ndarray, *, title: str, cmap: str, vmin: float | None = None, vlim: float | None = None):
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    xy = _channel_xy()
    if vlim is None:
        vmax = float(np.nanmax(np.abs(values))) if np.size(values) else 0.0
        vlim = vmax if vmax > 0 else 1.0
    if vmin is None:
        vmin = -vlim
    x = xy[:, 0]
    y = xy[:, 1]
    head = plt.Circle((0, 0), 1.08, color="black", fill=False, linewidth=1.0)
    ax.add_patch(head)
    ax.plot([0.0, -0.08, 0.08, 0.0], [1.08, 1.18, 1.18, 1.08], color="black", linewidth=1.0)
    try:
        triang = mtri.Triangulation(x, y)
        im = ax.tricontourf(triang, values, levels=15, cmap=cmap, vmin=vmin, vmax=vlim)
    except Exception:
        im = ax.scatter(x, y, c=values, cmap=cmap, vmin=vmin, vmax=vlim, s=80)
    ax.scatter(x, y, c="black", s=5, alpha=0.65)
    ax.set_aspect("equal")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.22)
    ax.axis("off")
    ax.set_title(title, fontsize=9)
    return im


def _channel_xy() -> np.ndarray:
    return np.asarray([_channel_position(ch) for ch in ALL_CHANNELS], dtype=float)


def _channel_position(ch_name: str) -> tuple[float, float]:
    import re

    ch = ch_name.upper()
    prefix = re.sub(r"[0-9Z]+$", "", ch)
    suffix_match = re.search(r"([0-9]+|Z)$", ch)
    suffix = suffix_match.group(1) if suffix_match else "Z"
    row_y = {
        "FP": 0.98,
        "AF": 0.78,
        "F": 0.56,
        "FT": 0.35,
        "FC": 0.30,
        "T": 0.00,
        "C": 0.00,
        "TP": -0.28,
        "CP": -0.34,
        "P": -0.58,
        "PO": -0.78,
        "O": -0.96,
    }
    y = row_y.get(prefix, 0.0)
    if suffix == "Z":
        return 0.0, y
    number = int(suffix)
    side = -1.0 if number % 2 else 1.0
    rank = {1: 0.18, 2: 0.18, 3: 0.38, 4: 0.38, 5: 0.58, 6: 0.58, 7: 0.82, 8: 0.82}.get(number, 0.95)
    if prefix in {"T", "FT", "TP"}:
        rank = min(0.98, rank + 0.08)
    if prefix in {"O", "PO"}:
        rank = min(0.78, rank)
    return side * rank, y


def plot_shared_cohort_spatial_topomaps(
    evidence: pd.DataFrame,
    paths,
    task,
    task_tag: str,
    *,
    subjects: Sequence[str],
    save_path: Path | None = None,
):
    import matplotlib.pyplot as plt

    attr = attribution_summary(paths, task, task_tag, subjects=subjects)
    attr = attr.set_index("Model")
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), dpi=150)
    spatial_vectors = [
        _safe_spatial_vector(attr.loc[model, "Spatial vector"]) if model in attr.index else None
        for model in MODEL_ORDER
    ]
    for ax, model, vector in zip(axes.ravel(), MODEL_ORDER, spatial_vectors):
        if vector is None:
            ax.axis("off")
            ax.set_title(f"{MODEL_LABELS[model]}\nno map", fontsize=9)
            continue
        weight = evidence.set_index("Model").get("Selected cohort reliability weight", pd.Series(dtype=float)).get(model, np.nan)
        title = f"{MODEL_LABELS[model]}\nw={weight:.3f}" if pd.notna(weight) else MODEL_LABELS[model]
        vmax = float(np.nanmax(vector)) if np.size(vector) and np.nanmax(vector) > 0.0 else 1.0
        _plot_topomap(ax, vector, title=title, cmap="Reds", vmin=0.0, vlim=vmax)
    fig.suptitle(f"Shared-cohort spatial importance: {_task_label(task_tag)}", fontweight="bold")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_shared_cohort_temporal_attribution(
    evidence: pd.DataFrame,
    paths,
    task,
    task_tag: str,
    *,
    subjects: Sequence[str],
    save_path: Path | None = None,
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    attr = attribution_summary(paths, task, task_tag, subjects=subjects)
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    weights = evidence.set_index("Model").get("Selected cohort reliability weight", pd.Series(dtype=float))
    for _, row in attr.iterrows():
        temporal = _safe_temporal_vector(row["Temporal vector"])
        if temporal is None:
            continue
        times = _time_axis(len(temporal), task)
        model = row["Model"]
        weight = float(weights.get(model, 0.0)) if model in weights.index else 0.0
        alpha = 0.95 if weight > 0 else 0.35
        linestyle = "-" if weight > 0 else "--"
        ax.plot(
            times,
            np.abs(temporal),
            color=MODEL_COLORS.get(model, "#777777"),
            label=f"{MODEL_LABELS.get(model, model)} (w={weight:.3f})",
            linewidth=2,
            alpha=alpha,
            linestyle=linestyle,
        )
    ax.set_title(f"Shared-cohort temporal attribution: {_task_label(task_tag)}", fontweight="bold")
    ax.set_xlabel("Time from stimulus onset (s)")
    ax.set_ylabel("Normalized importance")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    sns.despine(fig=fig)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_peak_time_difference_topomaps(
    evidence: pd.DataFrame,
    paths,
    task,
    task_tag: str,
    *,
    subjects: Sequence[str],
    save_path: Path | None = None,
):
    import matplotlib.pyplot as plt

    attr = attribution_summary(paths, task, task_tag, subjects=subjects).set_index("Model")
    model_vectors = {}
    peak_times = {}
    for model in MODEL_ORDER:
        if model not in attr.index:
            continue
        spatial = _safe_spatial_vector(attr.loc[model, "Spatial vector"])
        temporal = _safe_temporal_vector(attr.loc[model, "Temporal vector"])
        if spatial is None:
            continue
        scale = 1.0
        peak_time = np.nan
        if temporal is not None:
            idx = int(np.nanargmax(np.abs(temporal)))
            scale = float(np.abs(temporal[idx]))
            peak_time = float(_time_axis(len(temporal), task)[idx])
        model_vectors[model] = spatial * scale
        peak_times[model] = peak_time

    if not model_vectors:
        print("No spatial attribution vectors available for peak-time difference topomaps.")
        return None

    average = np.nanmean(np.vstack(list(model_vectors.values())), axis=0)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), dpi=150)
    diffs = [model_vectors[model] - average for model in model_vectors]
    vmax = max([float(np.nanmax(np.abs(d))) for d in diffs] or [1.0])
    for ax, model in zip(axes.ravel(), MODEL_ORDER):
        if model not in model_vectors:
            ax.axis("off")
            ax.set_title(f"{MODEL_LABELS[model]}\nno map", fontsize=9)
            continue
        peak = peak_times.get(model, np.nan)
        title = f"{MODEL_LABELS[model]}\npeak {peak:.2f}s dev." if pd.notna(peak) else f"{MODEL_LABELS[model]}\ndeviation"
        _plot_topomap(ax, model_vectors[model] - average, title=title, cmap="RdBu_r", vlim=vmax)
    fig.suptitle(f"Peak-time spatial deviation from average: {_task_label(task_tag)}", fontweight="bold")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_consensus_diagnostic(
    vectors: Mapping[str, np.ndarray | float | None],
    task,
    task_tag: str,
    *,
    save_path: Path | None = None,
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    weighted_spatial = _safe_spatial_vector(vectors.get("weighted_spatial"))
    weighted_temporal = _safe_temporal_vector(vectors.get("weighted_temporal"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    # Spatial plot
    ax_sp = axes[0]
    if weighted_spatial is None:
        ax_sp.axis("off")
        ax_sp.set_title("Weighted consensus spatial\nnot available")
    else:
        spatial_vmax = float(np.nanmax(weighted_spatial)) if np.nanmax(weighted_spatial) > 0.0 else 1.0
        _plot_topomap(ax_sp, weighted_spatial, title="Weighted spatial consensus", cmap="Reds", vmin=0.0, vlim=spatial_vmax)

    # Temporal plot
    ax_te = axes[1]
    if weighted_temporal is None:
        ax_te.axis("off")
        ax_te.set_title("Weighted consensus temporal\nnot available")
    else:
        ax_te.plot(_time_axis(len(weighted_temporal), task), np.abs(weighted_temporal), color="black", linewidth=2)
        ax_te.set_title("Weighted temporal consensus")
        ax_te.set_xlabel("Time from stimulus onset (s)")
        ax_te.set_ylabel("Normalized importance")

    fig.suptitle(f"Experimental consensus diagnostic: {_task_label(task_tag)}", fontweight="bold")
    sns.despine(fig=fig)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
