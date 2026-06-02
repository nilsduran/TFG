from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
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


# Standard 142-channel configuration used by ST-EEGFormer (TUH EEG dataset).
ST_EEGFORMER_CHANNELS = [
    "Fp1",
    "Fp2",
    "Fpz",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "AF7",
    "AF3",
    "AFz",
    "AF4",
    "AF8",
    "FC5",
    "FC1",
    "FC2",
    "FC6",
    "FT7",
    "FT8",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "TP9",
    "TP10",
    "CP5",
    "CP1",
    "CP2",
    "CP6",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "PO9",
    "PO10",
    "O1",
    "Oz",
    "O2",
    "Iz",
    "AFp9",
    "AFp10",
    "F9",
    "F10",
    "FT9",
    "FT10",
    "TP7",
    "TP8",
    "F5",
    "F1",
    "F2",
    "F6",
    "FC3",
    "FCz",
    "FC4",
    "C5",
    "C1",
    "C2",
    "C6",
    "CP3",
    "CPz",
    "CP4",
    "P5",
    "P1",
    "P2",
    "P6",
    "PO7",
    "PO8",
    # Extended 10-10 system to reach 142 channels
    "POz",
    "PO3",
    "PO4",
    "POO9",
    "POO10",
    "O9",
    "O10",
]
# Ensure we have close to 142 channels; pad list to match expected size
while len(ST_EEGFORMER_CHANNELS) < 142:
    ST_EEGFORMER_CHANNELS.append(f"X{len(ST_EEGFORMER_CHANNELS)}")


def align_channels_to_steegformer(
    X: np.ndarray,
    current_channel_names: list[str],
    target_channels: list[str] | None = None,
) -> np.ndarray:
    """
    Align EEG channels from current shape (N, C, T) to target shape (N, 142, T).

    Channels present in current_channel_names are placed at their correct indices
    in the target channel list. Missing channels are zero-filled.

    Args:
        X: numpy array of shape (N_samples, N_channels, N_timepoints)
        current_channel_names: list of channel names (length = X.shape[1])
        target_channels: list of 142 target channel names (default: ST_EEGFORMER_CHANNELS)

    Returns:
        Aligned numpy array of shape (N_samples, len(target_channels), N_timepoints)
    """
    if target_channels is None:
        target_channels = ST_EEGFORMER_CHANNELS

    n_samples, n_current_chans, n_times = X.shape
    n_target_chans = len(target_channels)

    X_aligned = np.zeros((n_samples, n_target_chans, n_times), dtype=X.dtype)

    # Normalize channel names to uppercase for case-insensitive matching
    current_names_upper = [ch.upper() for ch in current_channel_names]
    target_names_upper = [ch.upper() for ch in target_channels]
    target_index_by_name = {name: idx for idx, name in enumerate(target_names_upper)}

    # Map each current channel to its position in target
    channels_mapped = 0
    for i, ch_current in enumerate(current_names_upper):
        if ch_current in target_index_by_name:
            j = target_index_by_name[ch_current]
            X_aligned[:, j, :] = X[:, i, :]
            channels_mapped += 1
        else:
            print(f"Warning: Channel '{current_channel_names[i]}' not found in target list; skipping.")

    print(f"Aligned {channels_mapped}/{len(current_channel_names)} channels to {n_target_chans}-channel format.")
    return X_aligned


def _extract_features(backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if hasattr(backbone, "forward_encoder"):
        import inspect

        sig = inspect.signature(backbone.forward_encoder)
        kwargs = {}
        if "mask_ratio" in sig.parameters:
            kwargs["mask_ratio"] = 0.0
        if "chan_idx" in sig.parameters:
            # Create channel indices as a long tensor on the same device as input
            kwargs["chan_idx"] = torch.arange(x.shape[1], dtype=torch.long, device=x.device)

        out = backbone.forward_encoder(x, **kwargs)
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
    
    # Dynamically slice positional/time embeddings if shape doesn't match target model parameters
    for name, param in backbone.named_parameters():
        if name in state:
            state_shape = state[name].shape
            param_shape = param.shape
            if state_shape != param_shape:
                if "time_embed" in name or "pos_embed" in name:
                    if len(state_shape) == len(param_shape) and state_shape[:-1] != param_shape[:-1]:
                        print(f"Truncating checkpoint parameter '{name}' from {state_shape} to {param_shape}")
                        slices = tuple(slice(0, min(s_dim, p_dim)) for s_dim, p_dim in zip(state_shape, param_shape))
                        state[name] = state[name][slices]
                        
    backbone.load_state_dict(state, strict=strict)
    return backbone


def save_lora_adapters(model: nn.Module, path: Path, *, format: str = "safetensors") -> None:
    """Save only LoRA adapter parameters from ``model`` to ``path``.

    Parameters
    ----------
    model: nn.Module
        Model containing LoRA adapters (e.g. after ``prepare_lora_backbone``).
    path: Path
        Destination file path. Extension recommended: ``.safetensors`` or ``.pt``.
    format: str
        Either ``safetensors`` or ``pt``. Defaults to ``safetensors``.
    """
    lora_state = {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora" in k.lower()}
    if not lora_state:
        raise RuntimeError("No LoRA parameters found in the provided model.")
    if format == "safetensors" or Path(path).suffix == ".safetensors":
        try:
            from safetensors.torch import save_file

            save_file(lora_state, str(path))
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError("Failed to save safetensors file") from exc
    else:
        torch.save(lora_state, str(path))


def load_backbone_with_lora(
    backbone_factory: Callable[[], nn.Module],
    backbone_checkpoint: Path,
    lora_path: Path | None = None,
    *,
    include_substrings: tuple[str, ...] = ("attn",),
    exclude_substrings: tuple[str, ...] = ("head", "classifier", "decoder", "patch_embed", "mask_token"),
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
    train_classifier: bool = False,
    strict_backbone_load: bool = False,
) -> nn.Module:
    """Create a backbone, load pretrained weights, inject LoRA modules and optionally load LoRA params.

    This helper lets you obtain a model configured with the pretrained encoder weights
    and the LoRA adapters applied, without having to re-train the adapters.

    Parameters
    ----------
    backbone_factory: Callable[[], nn.Module]
        A zero-argument callable that returns a fresh backbone instance (e.g. ``mae_vit_small_patch16``).
    backbone_checkpoint: Path
        Path to the pretrained backbone checkpoint (safetensors or torch).
    lora_path: Path | None
        Optional path to a saved LoRA parameters file. If provided, the LoRA parameters
        will be loaded into the injected adapters (file may be .safetensors or .pt).
    include_substrings, exclude_substrings, r, alpha, dropout, train_classifier:
        Forwarded to ``prepare_lora_backbone``.
    strict_backbone_load: bool
        Whether to load backbone weights with strict=True (defaults to False).

    Returns
    -------
    nn.Module
        Backbone instance with LoRA adapters injected and (optionally) LoRA params loaded.
    """
    backbone = backbone_factory()
    load_backbone_weights(backbone, backbone_checkpoint, strict=strict_backbone_load)
    prepare_lora_backbone(
        backbone,
        include_substrings=include_substrings,
        exclude_substrings=exclude_substrings,
        r=r,
        alpha=alpha,
        dropout=dropout,
        train_classifier=train_classifier,
    )

    if lora_path is not None:
        if lora_path.suffix == ".safetensors":
            from safetensors.torch import load_file

            lora_state = load_file(str(lora_path))
        else:
            lora_state = torch.load(str(lora_path), map_location="cpu")

        # Load LoRA params permissively into the backbone (only keys present will be set).
        try:
            backbone.load_state_dict(lora_state, strict=False)
        except Exception:  # pragma: no cover - defensive
            # Fall back to manual key-by-key copy for maximal compatibility
            bd = backbone.state_dict()
            for k, v in lora_state.items():
                if k in bd:
                    bd[k].copy_(v.to(bd[k].device))
            backbone.load_state_dict(bd)

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
