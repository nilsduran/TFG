from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ExplanationResult:
    values: np.ndarray
    base_values: np.ndarray
    samples: np.ndarray
    class_ids: tuple[str, ...]
    method: str
    feature_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.values.shape[0] != len(self.samples):
            raise ValueError("SHAP values and samples must contain the same number of observations.")
        if self.values.shape[1] != len(self.class_ids):
            raise ValueError("The second SHAP axis must correspond to classes.")


def stratified_background(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    per_class: int = 20,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    indices: list[np.ndarray] = []
    for class_id in np.unique(y_train):
        class_indices = np.flatnonzero(y_train == class_id)
        count = min(per_class, len(class_indices))
        indices.append(rng.choice(class_indices, size=count, replace=False))
    selected = np.sort(np.concatenate(indices))
    return X_train[selected], y_train[selected]


def _normalize_shap_values(
    values: Any,
    *,
    n_samples: int,
    n_classes: int,
) -> np.ndarray:
    if isinstance(values, list):
        normalized = np.stack([np.asarray(class_values) for class_values in values], axis=1)
    else:
        normalized = np.asarray(values)
        if normalized.shape[0] != n_samples:
            raise ValueError("Unexpected SHAP sample axis.")
        if normalized.ndim >= 3 and normalized.shape[-1] == n_classes:
            normalized = np.moveaxis(normalized, -1, 1)
        elif n_classes == 1:
            normalized = normalized[:, None]
        elif normalized.ndim < 2 or normalized.shape[1] != n_classes:
            raise ValueError(f"Cannot identify class axis in SHAP shape {normalized.shape}.")
    if normalized.shape[:2] != (n_samples, n_classes):
        raise ValueError(f"Normalized SHAP shape is invalid: {normalized.shape}.")
    return normalized


def explain_tabular_model(
    model,
    *,
    background: np.ndarray,
    samples: np.ndarray,
    class_ids: Sequence[str],
    feature_names: Sequence[str] | None = None,
    algorithm: str = "permutation",
    max_evaluations: int | None = None,
) -> ExplanationResult:
    import shap

    explainer = shap.Explainer(
        model.predict_proba,
        masker=background,
        algorithm=algorithm,
        feature_names=list(feature_names) if feature_names is not None else None,
    )
    call_options = {}
    if max_evaluations is not None:
        call_options["max_evals"] = max_evaluations
    explanation = explainer(samples, **call_options)
    values = _normalize_shap_values(
        explanation.values,
        n_samples=len(samples),
        n_classes=len(class_ids),
    )
    return ExplanationResult(
        values=values,
        base_values=np.asarray(explanation.base_values),
        samples=samples,
        class_ids=tuple(class_ids),
        method=f"shap_{algorithm}",
        feature_names=tuple(feature_names) if feature_names is not None else None,
    )


def explain_torch_gradient(
    model,
    *,
    background: np.ndarray,
    samples: np.ndarray,
    class_ids: Sequence[str],
    batch_size: int = 32,
) -> ExplanationResult:
    import shap
    import torch

    parameter = next(model.parameters())
    device = parameter.device
    model.eval()
    background_tensor = torch.as_tensor(background, dtype=torch.float32, device=device)
    sample_tensor = torch.as_tensor(samples, dtype=torch.float32, device=device)
    explainer = shap.GradientExplainer(model, background_tensor, batch_size=batch_size)
    raw_values = explainer.shap_values(sample_tensor)
    values = _normalize_shap_values(
        raw_values,
        n_samples=len(samples),
        n_classes=len(class_ids),
    )
    return ExplanationResult(
        values=values,
        base_values=np.array([]),
        samples=samples,
        class_ids=tuple(class_ids),
        method="shap_gradient",
    )


def explanation_stability(
    first: ExplanationResult,
    second: ExplanationResult,
) -> float:
    first_values = np.mean(np.abs(first.values), axis=0).ravel()
    second_values = np.mean(np.abs(second.values), axis=0).ravel()
    if len(first_values) != len(second_values):
        raise ValueError("Explanation shapes do not match.")
    if np.std(first_values) == 0 or np.std(second_values) == 0:
        return float("nan")
    return float(np.corrcoef(first_values, second_values)[0, 1])


def save_explanation(result: ExplanationResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        values=result.values,
        base_values=result.base_values,
        samples=result.samples,
    )
    metadata = {
        "class_ids": list(result.class_ids),
        "method": result.method,
        "feature_names": list(result.feature_names) if result.feature_names is not None else None,
    }
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_explanation(path: Path) -> ExplanationResult:
    arrays = np.load(path)
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    feature_names = metadata.get("feature_names")
    return ExplanationResult(
        values=arrays["values"],
        base_values=arrays["base_values"],
        samples=arrays["samples"],
        class_ids=tuple(metadata["class_ids"]),
        method=metadata["method"],
        feature_names=tuple(feature_names) if feature_names is not None else None,
    )
