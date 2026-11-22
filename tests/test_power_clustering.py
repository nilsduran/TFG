import numpy as np
from scipy import sparse

from br_analysis.cluster_analysis import connected_reduction_region
from br_analysis.power import PowerConfig, compute_baseline_corrected_power


def test_power_is_baseline_corrected_and_keeps_bands_separate() -> None:
    sfreq = 100.0
    times = np.arange(-0.74, 1.501, 1.0 / sfreq)
    theta = np.sin(2 * np.pi * 6.0 * times)
    alpha = np.sin(2 * np.pi * 10.0 * times)
    signal = theta + alpha
    signal[times >= 0.0] += 2.0 * alpha[times >= 0.0]
    X = signal[None, None].astype(np.float32)

    power, power_times = compute_baseline_corrected_power(
        X,
        times,
        sfreq,
        PowerConfig(step_seconds=0.02),
    )

    assert power.shape[:3] == (1, 2, 1)
    assert power.shape[-1] == len(power_times)
    theta_mean = float(power[0, 0, 0].mean())
    alpha_mean = float(power[0, 1, 0].mean())
    assert alpha_mean > theta_mean


def test_reduction_region_has_minimum_extent() -> None:
    statistic = np.zeros((4, 20), dtype=float)
    statistic[1, 8:12] = 5.0
    statistic[3, 0] = 20.0
    adjacency = sparse.csr_matrix(
        np.array(
            [
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [0, 1, 1, 1],
                [0, 0, 1, 1],
            ]
        )
    )
    times = np.arange(20) * 0.05

    region = connected_reduction_region(
        statistic,
        adjacency,
        times,
        target_fraction=0.1,
        min_channels=3,
        min_duration=0.2,
    )

    assert region.sum() >= 8
    assert region.any(axis=1).sum() >= 3
    occupied_times = np.flatnonzero(region.any(axis=0))
    assert occupied_times[-1] - occupied_times[0] + 1 >= 4
