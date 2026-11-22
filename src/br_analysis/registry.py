from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class ModelSpec:
    id: str
    family: str
    preprocessing: str
    representation: str
    channels: tuple[str, ...] | None = None
    analysis_window: tuple[float, float] = (0.0, 1.5)


MODELS: Mapping[str, ModelSpec] = MappingProxyType(
    {
        "lr_theta": ModelSpec(
            "lr_theta",
            "sklearn",
            "band_limited",
            "theta_envelope",
            channels=("FCz",),
            analysis_window=(0.5, 1.0),
        ),
        "lr_alpha": ModelSpec(
            "lr_alpha",
            "sklearn",
            "band_limited",
            "alpha_envelope",
            channels=("Oz",),
            analysis_window=(0.5, 1.0),
        ),
        "riemann": ModelSpec("riemann", "sklearn", "broadband", "covariance"),
        "svc_agnostic": ModelSpec("svc_agnostic", "sklearn", "broadband", "band_summary"),
        "svc_clustering": ModelSpec(
            "svc_clustering",
            "sklearn",
            "band_limited",
            "parcel_band_features",
        ),
        "eegnet_agnostic": ModelSpec("eegnet_agnostic", "torch", "broadband", "raw"),
        "eegnet_clustering": ModelSpec(
            "eegnet_clustering",
            "torch",
            "band_limited",
            "spatial_parcels",
        ),
        "st_gnn": ModelSpec("st_gnn", "torch", "band_limited", "differential_entropy_graph"),
        "st_eegformer": ModelSpec("st_eegformer", "torch", "pretrained", "pretrained_tensor"),
        "labram": ModelSpec(
            "labram",
            "torch",
            "labram_pretrained",
            "pretrained_tensor",
            analysis_window=(0.0, 1.0),
        ),
    }
)


def get_model_spec(model_id: str) -> ModelSpec:
    try:
        return MODELS[model_id]
    except KeyError as exc:
        available = ", ".join(MODELS)
        raise ValueError(f"Unknown model {model_id!r}. Available models: {available}") from exc
