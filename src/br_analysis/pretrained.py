from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from .torch_models import freeze_non_lora_parameters, inject_lora_into_linears


def load_safetensors_state_dict(path: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(str(path))


def load_torch_state_dict(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        return state["model"]
    if isinstance(state, dict):
        return state
    raise TypeError(f"Unsupported checkpoint format: {type(state)!r}")


def _extract_features(backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if hasattr(backbone, "forward_encoder"):
        out = backbone.forward_encoder(x)
        if isinstance(out, tuple):
            out = out[0]
        if isinstance(out, dict):
            out = out.get("features") or out.get("x") or out.get("latent")
        if out.ndim == 3:
            out = out[:, 0]
        return out
    if hasattr(backbone, "forward_features"):
        out = backbone.forward_features(x)
        if isinstance(out, tuple):
            out = out[0]
        if isinstance(out, dict):
            out = out.get("features") or out.get("x") or out.get("latent")
        if out.ndim == 3:
            out = out[:, 0]
        return out
    if hasattr(backbone, "get_features"):
        out = backbone.get_features(x)
        if out.ndim == 3:
            out = out[:, 0]
        return out
    raise AttributeError("Backbone does not expose forward_encoder, forward_features, or get_features.")


class BackboneClassifier(nn.Module):
    """Wrap an arbitrary pretrained backbone with a linear classification head."""

    def __init__(
        self,
        backbone: nn.Module,
        n_classes: int,
        *,
        feature_dim: int | None = None,
        example_input: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        if feature_dim is None:
            with torch.no_grad():
                if example_input is None:
                    try:
                        sample = next(backbone.parameters())
                        device = sample.device
                    except StopIteration:
                        device = torch.device("cpu")
                    dummy = torch.zeros(
                        1,
                        getattr(backbone, "n_chans", 1),
                        getattr(backbone, "n_times", 128),
                        device=device,
                    )
                else:
                    dummy = example_input
                feat = _extract_features(backbone, dummy)
                feature_dim = int(feat.shape[-1])
        self.classifier = nn.Linear(feature_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = _extract_features(self.backbone, x)
        return self.classifier(feat)


def prepare_lora_backbone(
    backbone: nn.Module,
    *,
    include_substrings: tuple[str, ...] = ("attn",),
    exclude_substrings: tuple[str, ...] = ("head", "classifier", "decoder", "patch_embed", "tokenizer"),
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
    train_classifier: bool = True,
) -> nn.Module:
    inject_lora_into_linears(
        backbone,
        include_substrings=include_substrings,
        exclude_substrings=exclude_substrings,
        r=r,
        alpha=alpha,
        dropout=dropout,
    )
    freeze_non_lora_parameters(backbone, train_classifier=train_classifier)
    return backbone


def load_backbone_weights(backbone: nn.Module, checkpoint_path: Path, *, strict: bool = False) -> nn.Module:
    if checkpoint_path.suffix == ".safetensors":
        state = load_safetensors_state_dict(checkpoint_path)
    else:
        state = load_torch_state_dict(checkpoint_path)
    backbone.load_state_dict(state, strict=strict)
    return backbone


@dataclass(frozen=True)
class STEEGFormerConfig:
    repo_path: Path
    variant: str = "small"
    weights_file: str = "ST-EEGFormer_small_encoder.safetensors"
    weight_kind: str = "encoder"
    n_classes: int = 2
    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05


@dataclass(frozen=True)
class LabramConfig:
    pretrained_repo: str = "braindecode/labram-pretrained"
    n_classes: int = 2
    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
