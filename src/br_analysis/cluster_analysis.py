from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse, stats
from scipy.ndimage import uniform_filter1d

from .power import PowerDataset
from .reduction import channel_adjacency

ARTICLE_ROIS = {
    "theta": ("Fz", "FCz", "Cz"),
    "alpha": ("P7", "P8", "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"),
}


@dataclass(frozen=True)
class ClusterResult:
    band: str
    effect_maps: np.ndarray
    mean_effect: np.ndarray
    statistic: np.ndarray
    clusters: tuple[np.ndarray, ...]
    cluster_p_values: np.ndarray
    reduction_region: np.ndarray
    roi_channels: tuple[str, ...]
    roi_effect: np.ndarray
    roi_clusters: tuple[np.ndarray, ...]
    roi_cluster_p_values: np.ndarray

    @property
    def significant_cluster_count(self) -> int:
        return int(np.sum(self.cluster_p_values <= 0.05))


def subject_condition_means(
    dataset: PowerDataset,
    band: str,
) -> tuple[np.ndarray, np.ndarray]:
    dataset_class_count = len(dataset.class_ids)
    means = np.empty(
        (
            len(dataset.subject_ids),
            dataset_class_count,
            len(dataset.channel_names),
            len(dataset.times),
        ),
        dtype=np.float32,
    )
    band_power = dataset.band(band)
    for subject_index, subject in enumerate(dataset.subject_ids):
        subject_mask = dataset.subjects == subject
        for class_index in range(dataset_class_count):
            class_mask = subject_mask & (dataset.y == class_index)
            if not class_mask.any():
                raise ValueError(f"Subject {subject} has no trials for class {class_index}.")
            means[subject_index, class_index] = band_power[class_mask].mean(axis=0)
    return means, np.asarray(dataset.subject_ids)


def run_power_cluster_analysis(
    dataset: PowerDataset,
    band: str,
    *,
    n_permutations: int = 2000,
    p_uncorrected: float = 0.05,
    reduction_fraction: float = 0.05,
    min_reduction_channels: int = 3,
    min_reduction_duration: float = 0.25,
    random_seed: int = 42,
    verbose: bool = True,
) -> ClusterResult:
    if len(dataset.class_ids) != 2:
        raise ValueError("Power cluster analysis currently requires a binary task.")
    class_means, _ = subject_condition_means(dataset, band)
    effects = class_means[:, 0] - class_means[:, 1]
    threshold = float(
        stats.t.ppf(1.0 - p_uncorrected / 2.0, df=len(effects) - 1)
    )

    import mne

    adjacency = channel_adjacency(dataset.channel_names)
    statistic, raw_clusters, p_values, _ = (
        mne.stats.spatio_temporal_cluster_1samp_test(
            effects.transpose(0, 2, 1),
            adjacency=adjacency,
            threshold=threshold,
            tail=0,
            n_permutations=n_permutations,
            seed=random_seed,
            out_type="mask",
            verbose=verbose,
        )
    )
    clusters = tuple(np.asarray(mask).T for mask in raw_clusters)
    statistic = np.asarray(statistic).T
    reduction_region = connected_reduction_region(
        statistic,
        adjacency,
        dataset.times,
        target_fraction=reduction_fraction,
        min_channels=min_reduction_channels,
        min_duration=min_reduction_duration,
    )

    configured_roi = ARTICLE_ROIS[band]
    roi_channels = tuple(
        channel for channel in configured_roi if channel in dataset.channel_names
    )
    if not roi_channels:
        raise ValueError(f"None of the configured {band} ROI channels are available.")
    roi_indices = [dataset.channel_names.index(channel) for channel in roi_channels]
    roi_effect = effects[:, roi_indices].mean(axis=1)
    _, raw_roi_clusters, roi_p_values, _ = mne.stats.permutation_cluster_1samp_test(
        roi_effect,
        threshold=threshold,
        tail=0,
        n_permutations=n_permutations,
        seed=random_seed,
        out_type="mask",
        verbose=verbose,
    )
    roi_clusters = tuple(
        _as_time_mask(cluster, len(dataset.times))
        for cluster in raw_roi_clusters
    )
    return ClusterResult(
        band=band,
        effect_maps=effects,
        mean_effect=effects.mean(axis=0),
        statistic=statistic,
        clusters=clusters,
        cluster_p_values=np.asarray(p_values),
        reduction_region=reduction_region,
        roi_channels=roi_channels,
        roi_effect=roi_effect,
        roi_clusters=roi_clusters,
        roi_cluster_p_values=np.asarray(roi_p_values),
    )


def subject_reduction_regions(
    result: ClusterResult,
    channel_names: Sequence[str],
    times: np.ndarray,
    *,
    target_fraction: float = 0.05,
    min_channels: int = 3,
    min_duration: float = 0.25,
) -> np.ndarray:
    adjacency = channel_adjacency(channel_names)
    return np.stack(
        [
            connected_reduction_region(
                effect,
                adjacency,
                times,
                target_fraction=target_fraction,
                min_channels=min_channels,
                min_duration=min_duration,
            )
            for effect in result.effect_maps
        ]
    )


def jaccard_divergence(first: np.ndarray, second: np.ndarray) -> float:
    union = np.logical_or(first, second).sum()
    if union == 0:
        return 0.0
    return float(1.0 - np.logical_and(first, second).sum() / union)


def save_cluster_result(result: ClusterResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            band=result.band,
            effect_maps=result.effect_maps,
            mean_effect=result.mean_effect,
            statistic=result.statistic,
            clusters=np.stack(result.clusters)
            if result.clusters
            else np.empty((0, *result.statistic.shape), dtype=bool),
            cluster_p_values=result.cluster_p_values,
            reduction_region=result.reduction_region,
            roi_channels=np.asarray(result.roi_channels),
            roi_effect=result.roi_effect,
            roi_clusters=np.stack(result.roi_clusters)
            if result.roi_clusters
            else np.empty((0, result.roi_effect.shape[-1]), dtype=bool),
            roi_cluster_p_values=result.roi_cluster_p_values,
        )
    temporary.replace(path)


def load_cluster_result(path: Path) -> ClusterResult:
    with np.load(path, allow_pickle=False) as arrays:
        return ClusterResult(
            band=str(arrays["band"]),
            effect_maps=arrays["effect_maps"],
            mean_effect=arrays["mean_effect"],
            statistic=arrays["statistic"],
            clusters=tuple(arrays["clusters"]),
            cluster_p_values=arrays["cluster_p_values"],
            reduction_region=arrays["reduction_region"],
            roi_channels=tuple(arrays["roi_channels"].tolist()),
            roi_effect=arrays["roi_effect"],
            roi_clusters=tuple(arrays["roi_clusters"]),
            roi_cluster_p_values=arrays["roi_cluster_p_values"],
        )


def connected_reduction_region(
    statistic: np.ndarray,
    adjacency: sparse.spmatrix,
    times: np.ndarray,
    *,
    target_fraction: float = 0.05,
    min_channels: int = 3,
    min_duration: float = 0.25,
) -> np.ndarray:
    """Grow one high-effect region while preserving space-time connectivity."""
    if statistic.ndim != 2:
        raise ValueError("statistic must have shape (channels, times).")
    if not 0.0 < target_fraction <= 1.0:
        raise ValueError("target_fraction must be in (0, 1].")
    if min_channels < 1:
        raise ValueError("min_channels must be positive.")
    score = uniform_filter1d(np.abs(statistic), size=5, axis=1, mode="nearest")
    target_points = max(1, int(np.ceil(score.size * target_fraction)))
    if len(times) > 1:
        sample_period = float(np.median(np.diff(times)))
        min_time_points = max(1, int(np.ceil(min_duration / sample_period)))
    else:
        min_time_points = 1

    region = np.zeros_like(score, dtype=bool)
    seed = np.unravel_index(int(np.nanargmax(score)), score.shape)
    region[seed] = True
    neighbours = _channel_neighbours(adjacency)
    frontier = set(_point_neighbours(seed, score.shape, neighbours))

    while frontier and (
        region.sum() < target_points
        or np.sum(region.any(axis=1)) < min(min_channels, score.shape[0])
        or _maximum_time_span(region) < min(min_time_points, score.shape[1])
    ):
        point = max(frontier, key=lambda item: float(score[item]))
        frontier.remove(point)
        if region[point]:
            continue
        region[point] = True
        frontier.update(
            neighbour
            for neighbour in _point_neighbours(point, score.shape, neighbours)
            if not region[neighbour]
        )

    return _fill_temporal_gaps(region)


def _channel_neighbours(adjacency: sparse.spmatrix) -> tuple[np.ndarray, ...]:
    matrix = adjacency.tocsr()
    return tuple(
        matrix.indices[matrix.indptr[index] : matrix.indptr[index + 1]]
        for index in range(matrix.shape[0])
    )


def _point_neighbours(
    point: tuple[int, int],
    shape: tuple[int, int],
    channel_neighbours: Sequence[np.ndarray],
) -> set[tuple[int, int]]:
    channel, time = point
    neighbours = {(int(other), time) for other in channel_neighbours[channel]}
    if time > 0:
        neighbours.add((channel, time - 1))
    if time + 1 < shape[1]:
        neighbours.add((channel, time + 1))
    return neighbours


def _maximum_time_span(region: np.ndarray) -> int:
    occupied = np.flatnonzero(region.any(axis=0))
    return 0 if not len(occupied) else int(occupied[-1] - occupied[0] + 1)


def _fill_temporal_gaps(region: np.ndarray) -> np.ndarray:
    filled = region.copy()
    for channel in range(len(filled)):
        occupied = np.flatnonzero(filled[channel])
        if len(occupied) > 1:
            filled[channel, occupied[0] : occupied[-1] + 1] = True
    return filled


def _as_time_mask(cluster: object, n_times: int) -> np.ndarray:
    if isinstance(cluster, tuple):
        indices = np.asarray(cluster[0])
        mask = np.zeros(n_times, dtype=bool)
        mask[indices] = True
        return mask
    array = np.asarray(cluster)
    if array.dtype == bool and array.shape == (n_times,):
        return array
    mask = np.zeros(n_times, dtype=bool)
    mask[array.astype(int)] = True
    return mask
