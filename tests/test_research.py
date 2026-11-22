import numpy as np

from br_analysis.research.harness import evaluate_subject
from br_analysis.research.pipelines import CANDIDATES, BinnedAmplitudes, TimeDelayEmbedding
from br_analysis.research.pooled import ea_align, ea_reference
from br_analysis.research.stats import permutation_null, prevalence_inference


def _synthetic(n=200, effect=0.6, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.4).astype(int)
    X = rng.standard_normal((n, 8, 128))
    X[y == 1, :3, 40:90] += effect
    return X, y


def test_prevalence_detects_universal_effect_and_rejects_null():
    rng = np.random.default_rng(0)
    null_only = np.array([permutation_null(rng.integers(0, 2, 60), rng.integers(0, 2, 60), 2, 200, seed=i) for i in range(20)])
    assert prevalence_inference(null_only, 5000).gamma0 == 0.0
    strong = null_only.copy()
    strong[:, 0] = 0.75
    result = prevalence_inference(strong, 5000)
    assert result.global_p < 0.05 and result.gamma0 > 0.5


def test_holdout_is_last_twenty_percent_and_scored_once():
    X, y = _synthetic()
    outcome = evaluate_subject(CANDIDATES["erp_lda"].build(128.0), X, y, "S")
    assert outcome.n_holdout == 40 and outcome.n_dev == 160
    np.testing.assert_array_equal(outcome.holdout_y, y[160:])
    assert outcome.holdout_bacc > 0.7


def test_transformers_shapes():
    X = np.zeros((4, 3, 100))
    assert TimeDelayEmbedding(order=3, lag=2).transform(X).shape == (4, 9, 96)
    assert BinnedAmplitudes(0.1, 100.0).transform(X).shape == (4, 30)


def test_euclidean_alignment_whitens_mean_covariance():
    rng = np.random.default_rng(1)
    mixing = rng.standard_normal((6, 6))
    X = np.einsum("cd,ndt->nct", mixing, rng.standard_normal((50, 6, 400)))
    aligned = ea_align(X, ea_reference(X))
    centered = aligned - aligned.mean(-1, keepdims=True)
    mean_cov = np.einsum("nct,ndt->cd", centered, centered) / (50 * 400)
    np.testing.assert_allclose(mean_cov, np.eye(6), atol=5e-3)
