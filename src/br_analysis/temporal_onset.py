from __future__ import annotations

import json
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter, sosfiltfilt
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

from .artifacts import implementation_signature, stable_signature
from .config import ProjectPaths
from .data import data_inventory_signature, load_epoch_dataset
from .preprocessing import ERP_ANALYSIS_WINDOW, ERP_BASELINE
from .splits import (
    SplitPlan,
    chronological_block_labels,
    development_splits,
)
from .tasks import TASKS, get_task
from .temporal_statistics import (
    balanced_accuracy_curve,
    benjamini_yekutieli,
    first_sustained,
    hierarchical_onset_bootstrap,
    holm_adjust,
    max_t_inference,
    one_sided_sign_flip_pvalue,
    one_sided_t_pvalues,
    subject_max_t_pvalues,
)


@dataclass(frozen=True)
class TemporalOnsetConfig:
    permutations: int = 2000
    bootstrap_samples: int = 2000
    bootstrap_trial_replicates: int = 100
    pseudo_iterations: int = 100
    pseudo_folds: int = 6
    alpha: float = 0.05
    minimum_duration_seconds: float = 0.020
    baseline: tuple[float, float] = ERP_BASELINE
    search_window: tuple[float, float] = (-0.24, 1.5)
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.permutations < 1 or self.bootstrap_samples < 1:
            raise ValueError("permutations and bootstrap_samples must be positive.")
        if self.bootstrap_trial_replicates < 1:
            raise ValueError("bootstrap_trial_replicates must be positive.")
        if self.pseudo_iterations < 0 or self.pseudo_folds < 3:
            raise ValueError("Invalid pseudo-trial configuration.")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be between zero and one.")
        if self.minimum_duration_seconds <= 0:
            raise ValueError("minimum_duration_seconds must be positive.")


@dataclass
class _SubjectTemporalResult:
    subject: str
    y: np.ndarray
    trial_indices: np.ndarray
    development: np.ndarray
    holdout: np.ndarray
    splits: list[tuple[np.ndarray, np.ndarray]]
    blocks: np.ndarray
    predictions: np.ndarray
    pseudo_curve: np.ndarray | None
    X: np.ndarray
    eog_X: np.ndarray | None


_INTERPRETATIONS = {
    "c_vs_ip": (
        "Earliest decodable physical-stimulus/conflict difference; it is not "
        "by itself the onset of perceptual conflict."
    ),
    "im_vs_ip": (
        "Earliest signal predictive of the eventual mixed versus pure "
        "percept, not necessarily the onset of conscious perception."
    ),
    "ipr_vs_ipg": (
        "Earliest signal predictive of eventual red versus green dominance, "
        "not necessarily the onset of the percept."
    ),
    "cr_vs_cg": "Early decoding may primarily reflect physical colour information.",
    "r_vs_g": "Early decoding may combine physical colour and eventual percept information.",
    "c_vs_ip_vs_im": (
        "This multiclass contrast combines physical stimulus and perceptual "
        "outcome differences and therefore has no single process onset."
    ),
}


def run_temporal_onset_analysis(
    *,
    paths: ProjectPaths | None = None,
    task_ids: Sequence[str] | None = None,
    config: TemporalOnsetConfig = TemporalOnsetConfig(),
    split_plan: SplitPlan = SplitPlan(),
    n_jobs: int = 1,
    force: bool = False,
) -> Path:
    if n_jobs < 1:
        raise ValueError("n_jobs must be at least one.")
    paths = paths or ProjectPaths.from_env("temporal_onset")
    selected_tasks = list(task_ids) if task_ids is not None else list(TASKS)
    root = paths.artifacts_dir / "temporal_onset"
    root.mkdir(parents=True, exist_ok=True)
    family_id = stable_signature(
        {
            "config": asdict(config),
            "split_plan": asdict(split_plan),
            "tasks": list(TASKS),
            "data": data_inventory_signature(paths),
            "implementation": implementation_signature(
                (
                    "data.py",
                    "preprocessing.py",
                    "splits.py",
                    "tasks.py",
                    "temporal_onset.py",
                    "temporal_statistics.py",
                )
            ),
        }
    )
    summaries = []
    for index, task_id in enumerate(selected_tasks, start=1):
        print(f"[{index}/{len(selected_tasks)}] scientific temporal onset / {task_id}")
        summaries.append(
            _run_task(
                paths,
                task_id,
                config=config,
                split_plan=split_plan,
                n_jobs=n_jobs,
                family_id=family_id,
                force=force,
            )
        )
    _apply_family_correction(root, family_id, config.alpha)
    compatible_summaries = []
    for task_id in TASKS:
        path = root / task_id / "summary.json"
        if path.is_file():
            summary = json.loads(path.read_text(encoding="utf-8"))
            if summary.get("family_id") == family_id:
                compatible_summaries.append(summary)
    summaries = compatible_summaries or summaries
    pd.DataFrame(summaries).to_csv(root / "summary.csv", index=False)
    return root


def _run_task(
    paths: ProjectPaths,
    task_id: str,
    *,
    config: TemporalOnsetConfig,
    split_plan: SplitPlan,
    n_jobs: int,
    family_id: str,
    force: bool,
) -> dict[str, Any]:
    directory = paths.artifacts_dir / "temporal_onset" / task_id
    directory.mkdir(parents=True, exist_ok=True)
    summary_path = directory / "summary.json"
    task = get_task(task_id, analysis_window=ERP_ANALYSIS_WINDOW)
    run_signature = stable_signature(
        {
            "task": asdict(task),
            "config": asdict(config),
            "split_plan": asdict(split_plan),
            "data": data_inventory_signature(paths),
            "implementation": implementation_signature(
                (
                    "data.py",
                    "preprocessing.py",
                    "splits.py",
                    "tasks.py",
                    "temporal_onset.py",
                    "temporal_statistics.py",
                )
            ),
        }
    )
    decode_signature = stable_signature(
        {
            "task": asdict(task),
            "split_plan": asdict(split_plan),
            "baseline": config.baseline,
            "data": data_inventory_signature(paths),
            "implementation": implementation_signature(
                (
                    "data.py",
                    "preprocessing.py",
                    "splits.py",
                    "tasks.py",
                    "temporal_onset.py",
                )
            ),
        }
    )
    if summary_path.is_file() and not force:
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous.get("run_signature") == run_signature:
            print(f"Reusing scientific temporal onset: {directory}")
            return previous

    dataset = load_epoch_dataset(paths, task, "erp")
    primary_channels, eog_channels, frontal_channels = _channel_groups(
        dataset.channel_names
    )
    if len(primary_channels) < 2:
        raise ValueError("At least two non-EOG EEG channels are required.")
    baseline_mask = _time_mask(dataset.times, config.baseline)
    search_mask = _time_mask(dataset.times, config.search_window)
    prestim_mask = baseline_mask & (
        dataset.times < config.search_window[0] - 1e-7
    )
    if not prestim_mask.any():
        raise ValueError("The prestimulus control interval is empty.")
    minimum_points = max(
        2,
        int(np.ceil(config.minimum_duration_seconds * dataset.sfreq)) + 1,
    )
    chance = 1.0 / task.n_classes
    rng = np.random.default_rng(config.random_seed)

    subject_results, skipped = _decode_subjects(
        dataset,
        primary_channels,
        eog_channels,
        split_plan=split_plan,
        config=config,
        n_jobs=n_jobs,
        cache_directory=directory / "subject_cache",
        cache_signature=decode_signature,
        force=force,
    )
    if len(subject_results) < 2:
        raise RuntimeError("Fewer than two subjects support temporal inference.")

    subject_scores = np.stack(
        [
            balanced_accuracy_curve(
                result.y[result.development],
                result.predictions[result.development],
                n_classes=task.n_classes,
            )
            for result in subject_results
        ]
    )
    search_indices = np.flatnonzero(search_mask)
    prestim_indices = np.flatnonzero(prestim_mask)
    group = max_t_inference(
        subject_scores[:, search_indices] - chance,
        permutations=config.permutations,
        alpha=config.alpha,
        rng=rng,
    )
    prestim = max_t_inference(
        subject_scores[:, prestim_indices] - chance,
        permutations=config.permutations,
        alpha=config.alpha,
        rng=rng,
    )
    p_by = benjamini_yekutieli(group.p_uncorrected)
    prestim_p_by = benjamini_yekutieli(prestim.p_uncorrected)
    significant = group.p_fwer <= config.alpha
    onset_local = first_sustained(significant, minimum_points)
    onset_index = (
        int(search_indices[onset_local])
        if onset_local is not None
        else None
    )
    candidate_indices = (
        np.arange(
            onset_index,
            min(onset_index + minimum_points, len(dataset.times)),
        )
        if onset_index is not None
        else np.array([], dtype=int)
    )

    individual_rows = []
    subject_frame_blocks = []
    for subject_index, result in enumerate(subject_results):
        individual_p = subject_max_t_pvalues(
            result.y[result.development],
            result.predictions[result.development][:, search_indices],
            result.blocks,
            n_classes=task.n_classes,
            chance=chance,
            permutations=config.permutations,
            rng=rng,
        )
        individual_local = first_sustained(
            individual_p <= config.alpha,
            minimum_points,
        )
        individual_index = (
            int(search_indices[individual_local])
            if individual_local is not None
            else None
        )
        individual_rows.append(
            {
                "subject": result.subject,
                "detected": individual_index is not None,
                "onset_seconds": (
                    float(dataset.times[individual_index])
                    if individual_index is not None
                    else np.nan
                ),
                "censored_at_seconds": (
                    np.nan
                    if individual_index is not None
                    else float(config.search_window[1])
                ),
            }
        )
        subject_frame_blocks.append(
            pd.DataFrame(
                {
                    "subject": result.subject,
                    "time_seconds": dataset.times,
                    "balanced_accuracy": subject_scores[subject_index],
                    "chance_level": chance,
                    "p_fwer_individual": np.concatenate(
                        (
                            np.full(search_indices[0], np.nan),
                            individual_p,
                            np.full(
                                len(dataset.times) - search_indices[-1] - 1,
                                np.nan,
                            ),
                        )
                    ),
                }
            )
        )

    bootstrap_local = hierarchical_onset_bootstrap(
        [
            (
                result.y[result.development],
                result.predictions[result.development][:, search_indices],
            )
            for result in subject_results
        ],
        n_classes=task.n_classes,
        chance=chance,
        critical_t=group.critical_value,
        minimum_points=minimum_points,
        samples=config.bootstrap_samples,
        trial_replicates=config.bootstrap_trial_replicates,
        rng=rng,
    )
    bootstrap_seconds = np.full(len(bootstrap_local), np.nan)
    detected_bootstrap = np.isfinite(bootstrap_local)
    bootstrap_seconds[detected_bootstrap] = dataset.times[
        search_indices[bootstrap_local[detected_bootstrap].astype(int)]
    ]
    bootstrap_detection = float(detected_bootstrap.mean())
    bootstrap_ci = (
        np.quantile(bootstrap_seconds[detected_bootstrap], [0.025, 0.975])
        if detected_bootstrap.any()
        else np.array([np.nan, np.nan])
    )

    confirmation, control_rows = _confirm_candidate_interval(
        subject_results,
        dataset,
        candidate_indices,
        primary_channels=primary_channels,
        frontal_channels=frontal_channels,
        eog_channels=eog_channels,
        n_classes=task.n_classes,
        chance=chance,
        permutations=config.permutations,
        rng=rng,
    )
    control_summary = _summarize_controls(
        control_rows,
        chance=chance,
        permutations=config.permutations,
        rng=rng,
    )
    pseudo_frame = _pseudo_group_frame(
        subject_results,
        dataset.times,
        search_indices,
        chance=chance,
        alpha=config.alpha,
        minimum_points=minimum_points,
    )

    values = subject_scores
    sem = stats.sem(values, axis=0, nan_policy="omit")
    critical = stats.t.ppf(0.975, len(subject_results) - 1)
    means = values.mean(axis=0)
    group_frame = pd.DataFrame(
        {
            "time_seconds": dataset.times,
            "phase": np.where(
                prestim_mask,
                "prestimulus_baseline",
                np.where(search_mask, "ramp_and_stimulus", "outside_inference"),
            ),
            "balanced_accuracy_mean": means,
            "ci_low": means - critical * sem,
            "ci_high": means + critical * sem,
            "chance_level": chance,
            "t_statistic": np.nan,
            "p_uncorrected": np.nan,
            "p_fwer": np.nan,
            "p_by_fdr": np.nan,
            "significant_fwer": False,
            "significant_by_fdr": False,
        }
    )
    group_frame.loc[search_mask, "t_statistic"] = group.statistic
    group_frame.loc[search_mask, "p_uncorrected"] = group.p_uncorrected
    group_frame.loc[search_mask, "p_fwer"] = group.p_fwer
    group_frame.loc[search_mask, "p_by_fdr"] = p_by
    group_frame.loc[search_mask, "significant_fwer"] = significant
    group_frame.loc[search_mask, "significant_by_fdr"] = p_by <= config.alpha
    group_frame.loc[prestim_mask, "t_statistic"] = prestim.statistic
    group_frame.loc[prestim_mask, "p_uncorrected"] = prestim.p_uncorrected
    group_frame.loc[prestim_mask, "p_fwer"] = prestim.p_fwer
    group_frame.loc[prestim_mask, "p_by_fdr"] = prestim_p_by
    group_frame.loc[prestim_mask, "significant_fwer"] = (
        prestim.p_fwer <= config.alpha
    )
    group_frame.loc[prestim_mask, "significant_by_fdr"] = (
        prestim_p_by <= config.alpha
    )

    individual_frame = pd.DataFrame(individual_rows)
    detected_individual = individual_frame["detected"].sum()
    median_individual = _right_censored_median(
        individual_frame["onset_seconds"].to_numpy(),
        individual_frame["detected"].to_numpy(dtype=bool),
    )
    onset_seconds = (
        float(dataset.times[onset_index]) if onset_index is not None else None
    )
    summary = {
        "task_id": task_id,
        "n_subjects": len(subject_results),
        "chance_level": chance,
        "discovery_onset_seconds": onset_seconds,
        "candidate_interval_seconds": (
            [
                float(dataset.times[candidate_indices[0]]),
                float(dataset.times[candidate_indices[-1]]),
            ]
            if len(candidate_indices)
            else None
        ),
        "discovery_omnibus_p": group.omnibus_p,
        "discovery_omnibus_p_holm": None,
        "family_complete": False,
        "confirmation_balanced_accuracy": confirmation.get(
            "balanced_accuracy_mean"
        ),
        "confirmation_p": confirmation.get("p_value"),
        "confirmation_p_holm": None,
        "confirmed": False,
        "bootstrap_ci_seconds": (
            [float(bootstrap_ci[0]), float(bootstrap_ci[1])]
            if np.isfinite(bootstrap_ci).all()
            else None
        ),
        "bootstrap_detection_percent": 100.0 * bootstrap_detection,
        "onset_stable": bootstrap_detection >= 0.8,
        "subjects_with_onset": int(detected_individual),
        "subjects_with_onset_percent": float(
            100.0 * detected_individual / len(individual_frame)
        ),
        "median_individual_onset_seconds": median_individual,
        "prestimulus_omnibus_p": prestim.omnibus_p,
        "peak_group_balanced_accuracy": float(
            means[search_indices].max()
        ),
        "interpretation": _INTERPRETATIONS[task_id],
        "primary_estimand": "single-trial pointwise shrinkage LDA",
        "pseudo_trial_estimand": (
            "six averaged ERPs per class; sensitivity analysis only"
        ),
        "preprocessing_limitation": (
            "The available inputs are epoched FIF files rather than continuous "
            "recordings; filtering was applied to the full -740 to 1500 ms epoch."
        ),
        "preprocessing_metadata": dataset.preprocessing_metadata,
        "filter_sensitivity": _filter_sensitivity(
            dataset.times,
            dataset.sfreq,
            (0.1, 30.0),
        ),
        "primary_channels": len(primary_channels),
        "eog_channels": [dataset.channel_names[index] for index in eog_channels],
        "pseudo_trial_onset_seconds": (
            float(pseudo_frame["sensitivity_onset_seconds"].iloc[0])
            if (
                not pseudo_frame.empty
                and np.isfinite(
                    pseudo_frame["sensitivity_onset_seconds"].iloc[0]
                )
            )
            else None
        ),
        "control_summary": control_summary,
        "skipped_subjects": skipped,
        "config": asdict(config),
        "split_plan": asdict(split_plan),
        "family_id": family_id,
        "run_signature": run_signature,
    }

    directory.mkdir(parents=True, exist_ok=True)
    group_frame.to_csv(directory / "group_curve.csv", index=False)
    pd.concat(subject_frame_blocks, ignore_index=True).to_csv(
        directory / "subject_scores.csv",
        index=False,
    )
    individual_frame.to_csv(directory / "individual_onsets.csv", index=False)
    pd.DataFrame(confirmation.get("subjects", [])).to_csv(
        directory / "confirmation.csv",
        index=False,
    )
    pd.DataFrame(control_rows).to_csv(
        directory / "controls.csv",
        index=False,
    )
    pd.DataFrame(control_summary).to_csv(
        directory / "control_summary.csv",
        index=False,
    )
    pseudo_frame.to_csv(directory / "pseudo_trial_curve.csv", index=False)
    pd.DataFrame(
        {
            "onset_seconds": bootstrap_seconds,
            "detected": detected_bootstrap,
        }
    ).to_csv(directory / "bootstrap_onsets.csv", index=False)
    np.savez_compressed(
        directory / "development_predictions.npz",
        times=dataset.times,
        y_true=np.concatenate(
            [result.y[result.development] for result in subject_results]
        ),
        predictions=np.concatenate(
            [
                result.predictions[result.development]
                for result in subject_results
            ]
        ),
        subjects=np.concatenate(
            [
                np.full(len(result.development), result.subject)
                for result in subject_results
            ]
        ),
        trial_indices=np.concatenate(
            [
                result.trial_indices[result.development]
                for result in subject_results
            ]
        ),
    )
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    _plot_task(
        group_frame,
        pd.concat(subject_frame_blocks, ignore_index=True),
        individual_frame,
        pseudo_frame,
        bootstrap_seconds,
        confirmation,
        summary,
        directory / "onset.png",
    )
    return summary


def _decode_subjects(
    dataset,
    primary_channels: Sequence[int],
    eog_channels: Sequence[int],
    *,
    split_plan: SplitPlan,
    config: TemporalOnsetConfig,
    n_jobs: int,
    cache_directory: Path,
    cache_signature: str,
    force: bool,
) -> tuple[list[_SubjectTemporalResult], list[str]]:
    cache_directory.mkdir(parents=True, exist_ok=True)
    progress = tqdm(
        total=len(dataset.subject_ids),
        desc=f"{dataset.task_id}/pointwise LDA",
        unit="subject",
        dynamic_ncols=True,
    )
    lock = Lock()

    def run(subject: str) -> tuple[_SubjectTemporalResult | None, str | None]:
        mask = dataset.subjects == subject
        X = dataset.X[mask][:, primary_channels].astype(np.float64)
        eog_X = (
            dataset.X[mask][:, eog_channels].astype(np.float64)
            if eog_channels
            else None
        )
        y = dataset.y[mask]
        trial_indices = dataset.trial_indices[mask]
        baseline_mask = _time_mask(dataset.times, config.baseline)
        X -= X[..., baseline_mask].mean(axis=-1, keepdims=True)
        if eog_X is not None:
            eog_X -= eog_X[..., baseline_mask].mean(
                axis=-1,
                keepdims=True,
            )
        splits, holdout = development_splits(y, split_plan)
        development = np.arange(len(y) - len(holdout))
        expected = np.arange(len(dataset.class_ids))
        if (
            not splits
            or not len(holdout)
            or not np.array_equal(np.unique(y[development]), expected)
            or not np.array_equal(np.unique(y[holdout]), expected)
        ):
            with lock:
                progress.update()
            return None, f"{subject}: incomplete development or chronological holdout"
        cache_path = cache_directory / f"{_safe_filename(subject)}.npz"
        cached = (
            _load_subject_cache(
                cache_path,
                signature=cache_signature,
                y=y,
                trial_indices=trial_indices,
                expected_shape=(len(y), X.shape[-1]),
                pseudo_iterations=config.pseudo_iterations,
                pseudo_folds=config.pseudo_folds,
            )
            if not force
            else None
        )
        if cached is None:
            predictions = _pointwise_predictions(
                X,
                y,
                splits,
                n_classes=len(dataset.class_ids),
            )
            pseudo_curve = (
                _pseudo_trial_curve(
                    X[development],
                    y[development],
                    n_classes=len(dataset.class_ids),
                    folds=config.pseudo_folds,
                    iterations=config.pseudo_iterations,
                    random_seed=int(
                        stable_signature(
                            {
                                "seed": config.random_seed,
                                "subject": subject,
                            }
                        )[:8],
                        16,
                    ),
                )
                if config.pseudo_iterations
                else None
            )
        else:
            predictions, pseudo_curve = cached
            if config.pseudo_iterations and pseudo_curve is None:
                pseudo_curve = _pseudo_trial_curve(
                    X[development],
                    y[development],
                    n_classes=len(dataset.class_ids),
                    folds=config.pseudo_folds,
                    iterations=config.pseudo_iterations,
                    random_seed=int(
                        stable_signature(
                            {
                                "seed": config.random_seed,
                                "subject": subject,
                            }
                        )[:8],
                        16,
                    ),
                )
        if cached is None or config.pseudo_iterations:
            _save_subject_cache(
                cache_path,
                signature=cache_signature,
                y=y,
                trial_indices=trial_indices,
                predictions=predictions,
                pseudo_curve=pseudo_curve,
                pseudo_iterations=config.pseudo_iterations,
                pseudo_folds=config.pseudo_folds,
            )
        blocks = chronological_block_labels(
            len(development),
            split_plan.temporal_blocks,
        )
        result = _SubjectTemporalResult(
            subject=subject,
            y=y,
            trial_indices=trial_indices,
            development=development,
            holdout=holdout,
            splits=list(splits),
            blocks=blocks,
            predictions=predictions,
            pseudo_curve=pseudo_curve,
            X=X,
            eog_X=eog_X,
        )
        with lock:
            progress.set_postfix_str(f"subject={subject}", refresh=False)
            progress.update()
        return result, None

    try:
        if n_jobs == 1:
            outputs = [run(subject) for subject in dataset.subject_ids]
        else:
            with threadpool_limits(limits=1):
                with ThreadPoolExecutor(
                    max_workers=min(n_jobs, len(dataset.subject_ids))
                ) as executor:
                    outputs = list(executor.map(run, dataset.subject_ids))
    finally:
        progress.close()
    return (
        [result for result, _ in outputs if result is not None],
        [reason for _, reason in outputs if reason is not None],
    )


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _load_subject_cache(
    path: Path,
    *,
    signature: str,
    y: np.ndarray,
    trial_indices: np.ndarray,
    expected_shape: tuple[int, int],
    pseudo_iterations: int,
    pseudo_folds: int,
) -> tuple[np.ndarray, np.ndarray | None] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as arrays:
            if str(arrays["signature"].item()) != signature:
                return None
            if not np.array_equal(arrays["y"], y):
                return None
            if not np.array_equal(arrays["trial_indices"], trial_indices):
                return None
            predictions = arrays["predictions"]
            if predictions.shape != expected_shape:
                return None
            pseudo = arrays["pseudo_curve"]
            pseudo_matches = (
                int(arrays["pseudo_iterations"].item())
                == pseudo_iterations
                and int(arrays["pseudo_folds"].item()) == pseudo_folds
            )
            return (
                predictions,
                pseudo if len(pseudo) and pseudo_matches else None,
            )
    except (OSError, ValueError, KeyError):
        return None


def _save_subject_cache(
    path: Path,
    *,
    signature: str,
    y: np.ndarray,
    trial_indices: np.ndarray,
    predictions: np.ndarray,
    pseudo_curve: np.ndarray | None,
    pseudo_iterations: int,
    pseudo_folds: int,
) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            signature=np.array(signature),
            y=y,
            trial_indices=trial_indices,
            predictions=predictions,
            pseudo_iterations=np.array(pseudo_iterations),
            pseudo_folds=np.array(pseudo_folds),
            pseudo_curve=(
                pseudo_curve
                if pseudo_curve is not None
                else np.array([], dtype=np.float32)
            ),
        )
    temporary.replace(path)


def _pointwise_predictions(
    X: np.ndarray,
    y: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    n_classes: int,
) -> np.ndarray:
    predictions = np.full((len(y), X.shape[-1]), -1, dtype=np.int16)
    for train, test in splits:
        predictions[test] = _oas_lda_predict(
            X[train],
            y[train],
            X[test],
            n_classes=n_classes,
        )
    return predictions


def _pseudo_trial_curve(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_classes: int,
    folds: int,
    iterations: int,
    random_seed: int,
) -> np.ndarray:
    class_indices = [np.flatnonzero(y == class_id) for class_id in range(n_classes)]
    usable = min(len(indices) for indices in class_indices)
    usable -= usable % folds
    if usable < folds:
        return np.full(X.shape[-1], np.nan)
    rng = np.random.default_rng(random_seed)
    curves = np.empty((iterations, X.shape[-1]), dtype=np.float32)
    pseudo_y = np.repeat(np.arange(n_classes), folds)
    for iteration in range(iterations):
        pseudo = []
        for indices in class_indices:
            selected = rng.permutation(indices)[:usable]
            pseudo.extend(
                X[group].mean(axis=0)
                for group in np.array_split(selected, folds)
            )
        pseudo_X = np.stack(pseudo)
        predictions = np.full(
            (len(pseudo_y), X.shape[-1]),
            -1,
            dtype=np.int16,
        )
        for fold in range(folds):
            test = np.arange(fold, len(pseudo_y), folds)
            train = np.setdiff1d(np.arange(len(pseudo_y)), test)
            predictions[test] = _oas_lda_predict(
                pseudo_X[train],
                pseudo_y[train],
                pseudo_X[test],
                n_classes=n_classes,
            )
        curves[iteration] = balanced_accuracy_curve(
            pseudo_y,
            predictions,
            n_classes=n_classes,
        )
    return curves.mean(axis=0)


def _confirm_candidate_interval(
    results: Sequence[_SubjectTemporalResult],
    dataset,
    candidate_indices: np.ndarray,
    *,
    primary_channels: Sequence[int],
    frontal_channels: Sequence[int],
    eog_channels: Sequence[int],
    n_classes: int,
    chance: float,
    permutations: int,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not len(candidate_indices):
        return {}, []
    channel_sets = {
        "primary": np.arange(len(primary_channels)),
        "without_frontal": np.array(
            [
                local
                for local, original in enumerate(primary_channels)
                if original not in frontal_channels
            ]
        ),
    }
    control_rows = []
    subject_rows = []
    primary_scores = []
    for result in results:
        for control_id, channels in channel_sets.items():
            if len(channels) < 2:
                continue
            development_scores, holdout_scores = _interval_scores(
                result.X[:, channels],
                result.y,
                result.development,
                result.holdout,
                result.splits,
                candidate_indices,
                n_classes=n_classes,
            )
            row = {
                "subject": result.subject,
                "control": control_id,
                "development_balanced_accuracy": development_scores,
                "holdout_balanced_accuracy": holdout_scores,
            }
            control_rows.append(row)
            if control_id == "primary":
                primary_scores.append(holdout_scores)
                subject_rows.append(row)
        if eog_channels:
            development_scores, holdout_scores = _interval_scores(
                result.eog_X,
                result.y,
                result.development,
                result.holdout,
                result.splits,
                candidate_indices,
                n_classes=n_classes,
            )
            control_rows.append(
                {
                    "subject": result.subject,
                    "control": "eog_only",
                    "development_balanced_accuracy": development_scores,
                    "holdout_balanced_accuracy": holdout_scores,
                }
            )
    values = np.asarray(primary_scores)
    return (
        {
            "balanced_accuracy_mean": float(values.mean()),
            "p_value": one_sided_sign_flip_pvalue(
                values - chance,
                permutations=permutations,
                rng=rng,
            ),
            "subjects": subject_rows,
        },
        control_rows,
    )


def _interval_scores(
    X: np.ndarray,
    y: np.ndarray,
    development: np.ndarray,
    holdout: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    candidate_indices: np.ndarray,
    *,
    n_classes: int,
) -> tuple[float, float]:
    development_predictions = np.full(
        (len(development), len(candidate_indices)),
        -1,
        dtype=np.int16,
    )
    holdout_predictions = np.full(
        (len(holdout), len(candidate_indices)),
        -1,
        dtype=np.int16,
    )
    for local_time, time_index in enumerate(candidate_indices):
        for train, test in splits:
            development_predictions[test, local_time] = _oas_lda_predict(
                X[train][:, :, [time_index]],
                y[train],
                X[test][:, :, [time_index]],
                n_classes=n_classes,
            )[:, 0]
        holdout_predictions[:, local_time] = _oas_lda_predict(
            X[development][:, :, [time_index]],
            y[development],
            X[holdout][:, :, [time_index]],
            n_classes=n_classes,
        )[:, 0]
    return _interval_curve_scores(
        y,
        development,
        holdout,
        development_predictions,
        holdout_predictions,
        n_classes=n_classes,
    )


def _summarize_controls(
    rows: Sequence[dict[str, Any]],
    *,
    chance: float,
    permutations: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    summaries = []
    for control, group in frame.groupby("control", sort=False):
        holdout = group["holdout_balanced_accuracy"].dropna().to_numpy()
        summaries.append(
            {
                "control": control,
                "n_subjects": len(holdout),
                "development_balanced_accuracy": float(
                    group["development_balanced_accuracy"].mean()
                ),
                "holdout_balanced_accuracy": float(np.mean(holdout)),
                "holdout_p": one_sided_sign_flip_pvalue(
                    holdout - chance,
                    permutations=permutations,
                    rng=rng,
                ),
            }
        )
    return summaries


def _oas_lda_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    n_classes: int,
) -> np.ndarray:
    mean = X_train.mean(axis=0)
    scale = X_train.std(axis=0)
    scale = np.where(scale > np.finfo(float).eps, scale, 1.0)
    train = (X_train - mean) / scale
    test = (X_test - mean) / scale
    class_means = np.stack(
        [train[y_train == class_id].mean(axis=0) for class_id in range(n_classes)]
    )
    class_covariances = []
    weighted_residuals = []
    diagonal_terms = []
    n_features = train.shape[1]
    identity = np.eye(n_features)
    for class_id in range(n_classes):
        centered = train[y_train == class_id] - class_means[class_id]
        empirical = np.einsum(
            "nct,ndt->tcd",
            centered,
            centered,
            optimize=True,
        ) / len(centered)
        trace = np.trace(empirical, axis1=1, axis2=2)
        mu = trace / n_features
        alpha = np.mean(np.square(empirical), axis=(1, 2))
        denominator = (len(centered) + 1.0) * (
            alpha - np.square(mu) / n_features
        )
        shrinkage = np.divide(
            alpha + np.square(mu),
            denominator,
            out=np.ones_like(alpha),
            where=denominator > 0,
        )
        shrinkage = np.minimum(shrinkage, 1.0)
        covariance = (
            (1.0 - shrinkage)[:, None, None] * empirical
            + shrinkage[:, None, None] * mu[:, None, None] * identity
        )
        class_covariances.append(covariance)
        residual_by_time = centered.transpose(2, 0, 1)
        residual_weight = np.sqrt(
            (1.0 - shrinkage) / (n_classes * len(centered))
        )
        weighted_residuals.append(
            residual_by_time * residual_weight[:, None, None]
        )
        diagonal_terms.append(shrinkage * mu / n_classes)
    means_by_time = class_means.transpose(2, 0, 1)
    right_hand_side = means_by_time.transpose(0, 2, 1)
    if len(y_train) < n_features:
        residual_matrix = np.concatenate(weighted_residuals, axis=1)
        diagonal = np.maximum(np.sum(diagonal_terms, axis=0), 1e-8)
        gram = np.einsum(
            "tnc,tmc->tnm",
            residual_matrix,
            residual_matrix,
            optimize=True,
        )
        dual_identity = np.eye(gram.shape[1])
        system = (
            dual_identity
            + gram / diagonal[:, None, None]
        )
        residual_rhs = np.einsum(
            "tnc,tck->tnk",
            residual_matrix,
            right_hand_side,
            optimize=True,
        )
        correction = np.linalg.solve(system, residual_rhs)
        solved = (
            right_hand_side / diagonal[:, None, None]
            - np.einsum(
                "tnc,tnk->tck",
                residual_matrix,
                correction,
                optimize=True,
            )
            / np.square(diagonal)[:, None, None]
        )
    else:
        covariance = np.mean(class_covariances, axis=0)
        covariance += identity * 1e-8
        solved = np.linalg.solve(covariance, right_hand_side)
    coefficients = solved.transpose(0, 2, 1)
    intercept = -0.5 * np.einsum(
        "tkc,tkc->tk",
        means_by_time,
        coefficients,
        optimize=True,
    )
    scores = np.einsum(
        "ntc,tkc->ntk",
        test.transpose(0, 2, 1),
        coefficients,
        optimize=True,
    )
    scores += intercept[None, :, :]
    return scores.argmax(axis=2).astype(np.int16)


def _interval_curve_scores(
    y: np.ndarray,
    development: np.ndarray,
    holdout: np.ndarray,
    development_predictions: np.ndarray,
    holdout_predictions: np.ndarray,
    *,
    n_classes: int,
) -> tuple[float, float]:
    development_curve = balanced_accuracy_curve(
        y[development],
        development_predictions,
        n_classes=n_classes,
    )
    holdout_curve = balanced_accuracy_curve(
        y[holdout],
        holdout_predictions,
        n_classes=n_classes,
    )
    return float(development_curve.mean()), float(holdout_curve.mean())


def _pseudo_group_frame(
    results: Sequence[_SubjectTemporalResult],
    times: np.ndarray,
    search_indices: np.ndarray,
    *,
    chance: float,
    alpha: float,
    minimum_points: int,
) -> pd.DataFrame:
    available = [
        result.pseudo_curve
        for result in results
        if result.pseudo_curve is not None
    ]
    if not available:
        return pd.DataFrame()
    curves = np.stack(available)
    p_values = one_sided_t_pvalues(curves[:, search_indices] - chance)
    p_by = benjamini_yekutieli(p_values)
    onset = first_sustained(p_by <= alpha, minimum_points)
    frame = pd.DataFrame(
        {
            "time_seconds": times,
            "balanced_accuracy_mean": curves.mean(axis=0),
            "chance_level": chance,
            "p_by_fdr": np.nan,
            "significant_by_fdr": False,
            "sensitivity_onset_seconds": (
                float(times[search_indices[onset]])
                if onset is not None
                else np.nan
            ),
        }
    )
    frame.loc[search_indices, "p_by_fdr"] = p_by
    frame.loc[search_indices, "significant_by_fdr"] = p_by <= alpha
    return frame


def _apply_family_correction(
    root: Path,
    family_id: str,
    alpha: float,
) -> None:
    paths = [root / task_id / "summary.json" for task_id in TASKS]
    if not all(path.is_file() for path in paths):
        return
    summaries = [
        json.loads(path.read_text(encoding="utf-8")) for path in paths
    ]
    if any(summary.get("family_id") != family_id for summary in summaries):
        return
    discovery = holm_adjust(
        [summary["discovery_omnibus_p"] for summary in summaries]
    )
    confirmation_values = [
        (
            summary["confirmation_p"]
            if summary["confirmation_p"] is not None
            else 1.0
        )
        for summary in summaries
    ]
    confirmation = holm_adjust(confirmation_values)
    for path, summary, discovery_p, confirmation_p in zip(
        paths,
        summaries,
        discovery,
        confirmation,
        strict=True,
    ):
        summary["family_complete"] = True
        summary["discovery_omnibus_p_holm"] = float(discovery_p)
        summary["confirmation_p_holm"] = (
            float(confirmation_p)
            if summary["confirmation_p"] is not None
            else None
        )
        summary["confirmed"] = bool(
            summary["discovery_onset_seconds"] is not None
            and discovery_p <= alpha
            and summary["confirmation_p"] is not None
            and confirmation_p <= alpha
            and summary["confirmation_balanced_accuracy"]
            > summary["chance_level"]
        )
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _channel_groups(
    channel_names: Sequence[str],
) -> tuple[list[int], list[int], list[int]]:
    eog = [
        index
        for index, name in enumerate(channel_names)
        if re.search(r"(EOG|HEO|VEO)", name, flags=re.IGNORECASE)
    ]
    primary = [
        index for index in range(len(channel_names)) if index not in eog
    ]
    frontal = [
        index
        for index, name in enumerate(channel_names)
        if re.match(r"^(FP|AF|F)", name, flags=re.IGNORECASE)
    ]
    return primary, eog, frontal


def _right_censored_median(
    event_times: np.ndarray,
    detected: np.ndarray,
) -> float | None:
    event_times = np.asarray(event_times, dtype=float)
    detected = np.asarray(detected, dtype=bool)
    required_events = int(np.ceil(len(detected) / 2))
    observed = np.sort(event_times[detected])
    if len(observed) < required_events:
        return None
    return float(observed[required_events - 1])


def _filter_sensitivity(
    times: np.ndarray,
    sfreq: float,
    bandpass: tuple[float, float],
) -> dict[str, float | str | None]:
    impulse = np.zeros(len(times), dtype=float)
    zero_index = int(np.argmin(np.abs(times)))
    impulse[zero_index] = 1.0
    try:
        effective_bandpass = (
            bandpass[0],
            min(bandpass[1], sfreq * 0.49),
        )
        sos = butter(
            4,
            effective_bandpass,
            btype="bandpass",
            fs=sfreq,
            output="sos",
        )
        response = sosfiltfilt(sos, impulse)
    except ValueError:
        return {
            "method": "fourth-order zero-phase Butterworth impulse response",
            "pre_response_1pct_ms": None,
            "pre_response_5pct_ms": None,
        }
    peak = float(np.max(np.abs(response)))
    result: dict[str, float | str | None] = {
        "method": "fourth-order zero-phase Butterworth impulse response",
    }
    for percentage in (1, 5):
        threshold = peak * percentage / 100.0
        pre_indices = np.flatnonzero(
            (np.arange(len(times)) < zero_index)
            & (np.abs(response) >= threshold)
        )
        result[f"pre_response_{percentage}pct_ms"] = (
            float(
                1000.0
                * (times[zero_index] - times[pre_indices[0]])
            )
            if len(pre_indices)
            else 0.0
        )
    return result


def _time_mask(
    times: np.ndarray,
    interval: tuple[float, float],
) -> np.ndarray:
    start, end = interval
    mask = (times >= start - 1e-7) & (times <= end + 1e-7)
    if not mask.any():
        raise ValueError(f"No samples found inside interval {interval}.")
    return mask


def _plot_task(
    group: pd.DataFrame,
    subject_scores: pd.DataFrame,
    individual: pd.DataFrame,
    pseudo: pd.DataFrame,
    bootstrap_seconds: np.ndarray,
    confirmation: dict[str, Any],
    summary: dict[str, Any],
    destination: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    x = group["time_seconds"].to_numpy() * 1000
    mean = group["balanced_accuracy_mean"].to_numpy()
    axes[0, 0].plot(x, mean, color="#176b87", label="Single-trial LDA")
    axes[0, 0].fill_between(
        x,
        group["ci_low"],
        group["ci_high"],
        color="#9dd5df",
        alpha=0.45,
    )
    if not pseudo.empty:
        axes[0, 0].plot(
            pseudo["time_seconds"] * 1000,
            pseudo["balanced_accuracy_mean"],
            color="#8b5e34",
            alpha=0.8,
            label="Pseudo-trial sensitivity",
        )
    significant = group["significant_fwer"].to_numpy(dtype=bool)
    axes[0, 0].scatter(
        x[significant],
        mean[significant],
        color="#c43d4f",
        s=12,
        label="max-T FWER",
    )
    axes[0, 0].axhline(
        summary["chance_level"],
        color="#444444",
        linestyle="--",
        label="Chance",
    )
    axes[0, 0].axvline(-240, color="#6c757d", linestyle=":", label="Ramp onset")
    axes[0, 0].axvline(0, color="#111111", linestyle=":", label="Full stimulus")
    axes[0, 0].set(
        title=f"{summary['task_id']}: population decoding",
        xlabel="Time (ms)",
        ylabel="Balanced accuracy",
    )
    axes[0, 0].legend(fontsize=8, ncol=2)

    heatmap = subject_scores.pivot(
        index="subject",
        columns="time_seconds",
        values="balanced_accuracy",
    )
    axes[0, 1].imshow(
        heatmap.to_numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=max(0.0, summary["chance_level"] - 0.2),
        vmax=min(1.0, summary["chance_level"] + 0.2),
        extent=(
            heatmap.columns.min() * 1000,
            heatmap.columns.max() * 1000,
            len(heatmap),
            0,
        ),
    )
    axes[0, 1].axvline(-240, color="white", linestyle=":")
    axes[0, 1].axvline(0, color="black", linestyle=":")
    axes[0, 1].set(
        title="Subject-level out-of-fold curves",
        xlabel="Time (ms)",
        ylabel="Subject",
    )

    subject_confirmation = pd.DataFrame(confirmation.get("subjects", []))
    if not subject_confirmation.empty:
        values = subject_confirmation["holdout_balanced_accuracy"].to_numpy()
        order = np.argsort(values)
        axes[1, 0].scatter(values[order], np.arange(len(values)), color="#176b87")
        axes[1, 0].axvline(
            summary["chance_level"],
            color="#9c2f2f",
            linestyle="--",
        )
        axes[1, 0].set(
            title="Independent 20% confirmation interval",
            xlabel="Mean balanced accuracy",
            ylabel="Subject",
        )
    else:
        axes[1, 0].text(
            0.5,
            0.5,
            "No discovery interval to confirm",
            ha="center",
            va="center",
        )
        axes[1, 0].set_axis_off()

    detected = bootstrap_seconds[np.isfinite(bootstrap_seconds)] * 1000
    if len(detected):
        axes[1, 1].hist(detected, bins=min(30, max(5, len(np.unique(detected)))), color="#4f8a8b")
        axes[1, 1].set(
            title=(
                "Hierarchical bootstrap onset "
                f"({100 * len(detected) / len(bootstrap_seconds):.1f}% detected)"
            ),
            xlabel="Onset (ms)",
            ylabel="Replicates",
        )
    else:
        axes[1, 1].text(0.5, 0.5, "No bootstrap detections", ha="center", va="center")
        axes[1, 1].set_axis_off()
    figure.savefig(destination, dpi=160)
    plt.close(figure)
