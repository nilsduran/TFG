from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import ttest_rel
from .features import HilbertMeanEnvelope


@dataclass(frozen=True)
class ExperimentSettings:
    task_tag: str
    random_seed: int = 42
    map_cv_max_splits: int = 5
    n_perm_surrogate: int = 50
    epochs_grid: Sequence[int] = tuple(range(20, 200, 10))
    n_repeats_epochs: int = 15
    n_jobs: int = 1
    recalculate: bool = False


@dataclass(frozen=True)
class TimeWindowGrid:
    starts: np.ndarray
    stops: np.ndarray
    centers: np.ndarray
    window_sec: float
    step_sec: float

    @classmethod
    def from_times(cls, times: np.ndarray, *, window_sec: float, step_sec: float) -> "TimeWindowGrid":
        time_step = float(np.median(np.diff(times)))
        sfreq_est = 1.0 / time_step
        win_samples = max(2, int(round(window_sec * sfreq_est)))
        step_samples = max(1, int(round(step_sec * sfreq_est)))
        starts = np.arange(0, len(times) - win_samples + 1, step_samples)
        stops = starts + win_samples
        centers = np.array([times[s:e].mean() for s, e in zip(starts, stops)], dtype=np.float32)
        return cls(starts=starts, stops=stops, centers=centers, window_sec=window_sec, step_sec=step_sec)


def choose_n_splits(y_s: np.ndarray, max_splits: int = 5) -> int:
    return max(2, min(max_splits, int(np.bincount(y_s).min())))


def temporal_stratified_splits(y_s: np.ndarray, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Forward-chaining CV splits that keep test epochs later than train epochs.

    If a specific split count yields no valid folds (e.g. because subset blocks have
    only one class), this function dynamically falls back to lower split counts (down to 2).
    If that still fails, it searches for a single split point that yields valid binary
    train and test partitions.
    
    As an ultimate fallback (e.g. for degenerate randomized permutations where one class 
    completely precedes the other), it returns a randomized stratified 2-fold split.
    """
    y_s = np.asarray(y_s)
    
    # 1. Try with decreasing number of splits down to 2
    for current_splits in range(n_splits, 1, -1):
        indices = np.arange(len(y_s))
        blocks = np.array_split(indices, current_splits + 1)
        splits: list[tuple[np.ndarray, np.ndarray]] = []
        for fold_idx in range(1, len(blocks)):
            train_idx = np.concatenate(blocks[:fold_idx])
            test_idx = blocks[fold_idx]
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            if len(np.unique(y_s[train_idx])) < 2 or len(np.unique(y_s[test_idx])) < 2:
                continue
            splits.append((train_idx, test_idx))
        if splits:
            return splits

    # 2. Fallback to a single split point search
    n = len(y_s)
    for split_point in range(n // 4, 3 * n // 4 + 1):
        train_idx = np.arange(0, split_point)
        test_idx = np.arange(split_point, n)
        if len(np.unique(y_s[train_idx])) >= 2 and len(np.unique(y_s[test_idx])) >= 2:
            return [(train_idx, test_idx)]

    # 3. Ultimate fallback: randomized stratified 2-fold split
    idx_0 = np.where(y_s == 0)[0]
    idx_1 = np.where(y_s == 1)[0]
    mid_0 = len(idx_0) // 2
    mid_1 = len(idx_1) // 2
    
    train_idx = np.concatenate([idx_0[:mid_0], idx_1[:mid_1]])
    test_idx = np.concatenate([idx_0[mid_0:], idx_1[mid_1:]])
    return [(train_idx, test_idx), (test_idx, train_idx)]


def cv_accuracy(X_feat: np.ndarray, y_s: np.ndarray, n_splits: int, model_factory: Any, *, seed: int) -> float:
    if X_feat.ndim == 1:
        X_feat = X_feat[:, None]
    accs = []
    for tr, te in temporal_stratified_splits(y_s, n_splits):
        clf = model_factory.make()
        clf.fit(X_feat[tr], y_s[tr])
        accs.append(float(np.mean(clf.predict(X_feat[te]) == y_s[te])))
    return float(np.mean(accs))


def surrogate_distribution(
    X_feat: np.ndarray,
    y_s: np.ndarray,
    n_splits: int,
    n_perm: int,
    model_factory: Any,
    *,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    perm_accs = np.empty(n_perm, dtype=np.float32)
    for p in range(n_perm):
        y_perm = rng.permutation(y_s)
        perm_accs[p] = cv_accuracy(X_feat, y_perm, n_splits, model_factory, seed=seed + p + 1)
    return perm_accs


def balanced_sample(y_s: np.ndarray, n_total: int, rng: np.random.Generator) -> np.ndarray | None:
    n2 = n_total // 2
    idx_0 = np.where(y_s == 0)[0]
    idx_1 = np.where(y_s == 1)[0]
    if len(idx_0) < n2 or len(idx_1) < n2:
        return None
    out = np.concatenate([rng.choice(idx_0, n2, replace=False), rng.choice(idx_1, n2, replace=False)])
    out.sort()
    return out


def _process_channel_subject(
    si: int,
    subj: str,
    X_all: np.ndarray,
    y_all: np.ndarray,
    subj_list_all: np.ndarray,
    channel_names: Sequence[str],
    starts: np.ndarray,
    stops: np.ndarray,
    settings: ExperimentSettings,
    model_factory: Any,
    feature_extractor: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx_s = np.where(subj_list_all == subj)[0]
    Xs, ys = X_all[idx_s], y_all[idx_s]
    real_arr = np.full((len(channel_names), len(starts)), np.nan, dtype=np.float32)
    surr_mean = np.full_like(real_arr, np.nan)
    surr_q95 = np.full_like(real_arr, np.nan)
    if len(np.unique(ys)) < 2:
        return real_arr, surr_mean, surr_q95

    n_splits = choose_n_splits(ys, max_splits=settings.map_cv_max_splits)
    base_seed = settings.random_seed + 10_000 * (si + 1)
    for ch_idx in range(len(channel_names)):
        for wi, (s, e) in enumerate(zip(starts, stops)):
            X_feat = feature_extractor.transform(Xs[:, ch_idx, s:e])
            win_seed = base_seed + wi * 1000 + ch_idx
            real_arr[ch_idx, wi] = cv_accuracy(X_feat, ys, n_splits, model_factory, seed=win_seed)
            dist = surrogate_distribution(X_feat, ys, n_splits, settings.n_perm_surrogate, model_factory, seed=win_seed + 1)
            surr_mean[ch_idx, wi] = float(dist.mean())
            surr_q95[ch_idx, wi] = float(np.quantile(dist, 0.95))
    return real_arr, surr_mean, surr_q95


def _process_tensor_subject(
    si: int,
    subj: str,
    X_dict: dict[str, np.ndarray],
    y_all: np.ndarray,
    subj_list_all: np.ndarray,
    band_names: Sequence[str],
    starts: np.ndarray,
    stops: np.ndarray,
    settings: ExperimentSettings,
    model_factory: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx_s = np.where(subj_list_all == subj)[0]
    ys = y_all[idx_s]
    real_arr = np.full((len(band_names), len(starts)), np.nan, dtype=np.float32)
    surr_mean = np.full_like(real_arr, np.nan)
    surr_q95 = np.full_like(real_arr, np.nan)
    if len(np.unique(ys)) < 2:
        return real_arr, surr_mean, surr_q95

    n_splits = choose_n_splits(ys, max_splits=settings.map_cv_max_splits)
    base_seed = settings.random_seed + 10_000 * (si + 1)
    for b_idx, band in enumerate(band_names):
        Xs_band = X_dict[band][idx_s]
        for wi, (s, e) in enumerate(zip(starts, stops)):
            X_feat = Xs_band[:, :, s:e]
            win_seed = base_seed + wi * 1000 + b_idx
            real_arr[b_idx, wi] = cv_accuracy(X_feat, ys, n_splits, model_factory, seed=win_seed)
            dist = surrogate_distribution(
                X_feat,
                ys,
                n_splits,
                min(settings.n_perm_surrogate, 500),
                model_factory,
                seed=win_seed + 1,
            )
            surr_mean[b_idx, wi] = float(dist.mean())
            surr_q95[b_idx, wi] = float(np.quantile(dist, 0.95))
    return real_arr, surr_mean, surr_q95


def _epochs_channel_worker(
    si: int,
    subj: str,
    X_all: np.ndarray,
    y_all: np.ndarray,
    subj_list_all: np.ndarray,
    channel_names: Sequence[str],
    settings: ExperimentSettings,
    model_factory: Any,
    feature_extractor: Any,
) -> list[dict[str, Any]]:
    idx_s = np.where(subj_list_all == subj)[0]
    Xs, ys = X_all[idx_s], y_all[idx_s]
    if len(np.unique(ys)) < 2:
        return []

    rows = []
    rng = np.random.default_rng(settings.random_seed + si * 9973)
    n_splits = choose_n_splits(ys, max_splits=settings.map_cv_max_splits)
    for ch_idx, ch_name in enumerate(channel_names):
        X_ch = Xs[:, ch_idx, :]
        for n_total in settings.epochs_grid:
            rows.extend(
                _epochs_rows_for_feature(
                    subj,
                    ch_name,
                    X_ch,
                    ys,
                    n_total,
                    si,
                    ch_idx,
                    rng,
                    n_splits,
                    settings,
                    model_factory,
                    lambda X: feature_extractor.transform(X),
                    min_perm=max(20, settings.n_perm_surrogate // 5),
                )
            )
    return rows


def _epochs_tensor_worker(
    si: int,
    subj: str,
    X_dict: dict[str, np.ndarray],
    y_all: np.ndarray,
    subj_list_all: np.ndarray,
    band_names: Sequence[str],
    settings: ExperimentSettings,
    model_factory: Any,
) -> list[dict[str, Any]]:
    idx_s = np.where(subj_list_all == subj)[0]
    ys = y_all[idx_s]
    if len(np.unique(ys)) < 2:
        return []

    rows = []
    rng = np.random.default_rng(settings.random_seed + si * 9973)
    n_splits = choose_n_splits(ys, max_splits=settings.map_cv_max_splits)
    for b_idx, band in enumerate(band_names):
        Xs_band = X_dict[band][idx_s]
        for n_total in settings.epochs_grid:
            rows.extend(
                _epochs_rows_for_feature(
                    subj,
                    band,
                    Xs_band,
                    ys,
                    n_total,
                    si,
                    b_idx,
                    rng,
                    n_splits,
                    settings,
                    model_factory,
                    lambda X: X,
                    min_perm=max(50, settings.n_perm_surrogate // 10),
                )
            )
    return rows


def _epochs_rows_for_feature(
    subj: str,
    label: str,
    X_source: np.ndarray,
    ys: np.ndarray,
    n_total: int,
    si: int,
    source_idx: int,
    rng: np.random.Generator,
    n_splits: int,
    settings: ExperimentSettings,
    model_factory: Any,
    feature_fn: Callable[[np.ndarray], np.ndarray],
    *,
    min_perm: int,
) -> list[dict[str, Any]]:
    if n_total % 2 != 0:
        return []
    real_accs = []
    perm_all = []
    for rep in range(settings.n_repeats_epochs):
        idx_sub = balanced_sample(ys, n_total, rng)
        if idx_sub is None:
            continue
        X_feat = feature_fn(X_source[idx_sub])
        seed_r = settings.random_seed + si * 9973 + source_idx * 997 + n_total + rep
        real_accs.append(cv_accuracy(X_feat, ys[idx_sub], n_splits, model_factory, seed=seed_r))
        perm_all.append(
            surrogate_distribution(X_feat, ys[idx_sub], n_splits, min_perm, model_factory, seed=seed_r + 1)
        )

    if not real_accs:
        return []
    perm_concat = np.concatenate(perm_all) if perm_all else np.array([0.5])
    real_mean = float(np.mean(real_accs))
    chance_mean = float(perm_concat.mean())
    chance_q95 = float(np.quantile(perm_concat, 0.95))
    return [
        {
            "subject": subj,
            "electrode": label,
            "n_epochs_total": int(n_total),
            "acc_real_mean": real_mean,
            "acc_real_std": float(np.std(real_accs)),
            "acc_chance_mean": chance_mean,
            "acc_chance_q95": chance_q95,
            "delta_acc": real_mean - chance_mean,
            "delta_acc_q95": real_mean - chance_q95,
        }
    ]


@dataclass
class ChanceLevelExperiment:
    settings: ExperimentSettings
    model_factory: Any
    feature_extractor: Any = HilbertMeanEnvelope()

    def run_epochs_grid(
        self,
        X: np.ndarray | dict[str, np.ndarray],
        y: np.ndarray,
        subjects: np.ndarray,
        labels: Sequence[str],
        *,
        cache_file: Path | None = None,
    ) -> pd.DataFrame:
        if cache_file is not None and cache_file.exists() and not self.settings.recalculate:
            print(f"Loading df_epochs from cache: {cache_file}")
            return pd.read_csv(cache_file)

        subject_ids = pd.unique(subjects).tolist()
        if isinstance(X, dict):
            all_rows = Parallel(n_jobs=self.settings.n_jobs, verbose=5)(
                delayed(_epochs_tensor_worker)(si, subj, X, y, subjects, labels, self.settings, self.model_factory)
                for si, subj in enumerate(subject_ids)
            )
        else:
            all_rows = Parallel(n_jobs=self.settings.n_jobs, verbose=5)(
                delayed(_epochs_channel_worker)(
                    si, subj, X, y, subjects, labels, self.settings, self.model_factory, self.feature_extractor
                )
                for si, subj in enumerate(subject_ids)
            )

        df_epochs = (
            pd.DataFrame([r for rows in all_rows for r in rows])
            .sort_values(["subject", "electrode", "n_epochs_total"])
            .reset_index(drop=True)
        )
        if cache_file is not None:
            df_epochs.to_csv(cache_file, index=False)
            print(f"Saved cache: {cache_file}")
        return df_epochs

    def run_time_map(
        self,
        X: np.ndarray | dict[str, np.ndarray],
        y: np.ndarray,
        subjects: np.ndarray,
        labels: Sequence[str],
        windows: TimeWindowGrid,
        *,
        cache_file: Path | None = None,
    ) -> dict[str, np.ndarray]:
        if cache_file is not None and cache_file.exists() and not self.settings.recalculate:
            print(f"Loading time map from cache: {cache_file}")
            map_data = np.load(cache_file)
            return {key: map_data[key] for key in map_data.files}

        subject_ids = pd.unique(subjects).tolist()
        if isinstance(X, dict):
            results = Parallel(n_jobs=self.settings.n_jobs)(
                delayed(_process_tensor_subject)(
                    si, subj, X, y, subjects, labels, windows.starts, windows.stops, self.settings, self.model_factory
                )
                for si, subj in enumerate(subject_ids)
            )
        else:
            results = Parallel(n_jobs=self.settings.n_jobs)(
                delayed(_process_channel_subject)(
                    si,
                    subj,
                    X,
                    y,
                    subjects,
                    labels,
                    windows.starts,
                    windows.stops,
                    self.settings,
                    self.model_factory,
                    self.feature_extractor,
                )
                for si, subj in enumerate(subject_ids)
            )

        out = {
            "real_map": np.stack([r[0] for r in results], axis=0),
            "surr_mean": np.stack([r[1] for r in results], axis=0),
            "surr_q95": np.stack([r[2] for r in results], axis=0),
        }
        if cache_file is not None:
            np.savez_compressed(cache_file, **out)
            print(f"Saved time map to cache: {cache_file}")
        return out


def paired_time_map_stats(real_map: np.ndarray, surr_mean: np.ndarray, surr_q95: np.ndarray) -> dict[str, np.ndarray]:
    n_bands, n_w = real_map.shape[1], real_map.shape[2]
    p_map = np.full((n_bands, n_w), np.nan, dtype=np.float64)
    t_map = np.full((n_bands, n_w), np.nan, dtype=np.float64)
    for b_idx in range(n_bands):
        for wi in range(n_w):
            a = real_map[:, b_idx, wi]
            b = surr_mean[:, b_idx, wi]
            valid = ~np.isnan(a) & ~np.isnan(b)
            if valid.sum() < 6:
                continue
            t_stat, p_two = ttest_rel(a[valid], b[valid])
            t_map[b_idx, wi] = t_stat
            p_map[b_idx, wi] = (p_two / 2) if t_stat > 0 else 1.0

    delta_map = real_map - surr_mean
    above_q95_map = real_map > surr_q95
    mean_delta_map = np.nanmean(delta_map, axis=0)
    hit_rate_map = np.nanmean(above_q95_map.astype(float), axis=0)
    sig_map = (p_map < 0.05) & (mean_delta_map > 0)
    robust_map = sig_map & (hit_rate_map >= 0.50)
    return {
        "p_map": p_map,
        "t_map": t_map,
        "delta_map": delta_map,
        "above_q95_map": above_q95_map,
        "mean_delta_map": mean_delta_map,
        "hit_rate_map": hit_rate_map,
        "sig_map": sig_map,
        "robust_map": robust_map,
    }
