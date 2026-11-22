from __future__ import annotations

from pathlib import Path

from torch import nn

from .torch_models import freeze_non_lora_parameters, inject_lora_into_linears

ST_EEGFORMER_REPO = "braindecode/STEEGFormer-small"
ST_EEGFORMER_REVISION = "633c9c0d76c6e26459e1e0d1337c91858578f13d"
LABRAM_REPO = "braindecode/labram-pretrained"
LABRAM_REVISION = "0563b6c626e7b40d9a36653b763715db94d945d7"


def standard_channel_info(
    channel_names: list[str],
    *,
    sfreq: float,
) -> list[dict]:
    """Reconstruct channel positions omitted from portable preprocessing caches."""
    import mne

    info = mne.create_info(channel_names, sfreq=sfreq, ch_types="eeg")
    info.set_montage(
        mne.channels.make_standard_montage("standard_1005"),
        match_case=False,
        on_missing="raise",
    )
    return list(info["chs"])


def _pretrained_source(
    local_directory: Path | None,
    *,
    repo: str,
    revision: str,
) -> tuple[str, str | None]:
    if local_directory is None:
        return repo, revision
    source = local_directory.expanduser().resolve()
    if not source.is_dir() or not (source / "config.json").is_file():
        raise ValueError(
            "--checkpoint must point to a complete Hugging Face model directory "
            "containing config.json, not to a standalone weights file."
        )
    return str(source), None


def _prepare_lora(model: nn.Module) -> nn.Module:
    inject_lora_into_linears(
        model,
        include_substrings=("encoder", "blocks"),
        exclude_substrings=("final_layer", "head", "classifier"),
        r=8,
        alpha=16.0,
        dropout=0.05,
    )
    freeze_non_lora_parameters(
        model,
        train_classifier=True,
        train_tokenizer=False,
    )
    if not any("lora_" in name for name, _ in model.named_parameters()):
        raise RuntimeError("No LoRA adapters were attached to the pretrained model.")
    return model


def load_st_eegformer(
    *,
    n_chans: int,
    n_times: int,
    n_outputs: int,
    chs_info: list[dict],
    local_directory: Path | None = None,
) -> nn.Module:
    from braindecode.models import STEEGFormer

    source, revision = _pretrained_source(
        local_directory,
        repo=ST_EEGFORMER_REPO,
        revision=ST_EEGFORMER_REVISION,
    )
    kwargs = {
        "n_chans": n_chans,
        "n_times": n_times,
        "n_outputs": n_outputs,
        "chs_info": chs_info,
    }
    if revision is not None:
        kwargs["revision"] = revision
    model = STEEGFormer.from_pretrained(source, **kwargs)
    return _prepare_lora(model)


def load_labram(
    *,
    n_times: int,
    n_outputs: int,
    chs_info: list[dict],
    local_directory: Path | None = None,
) -> nn.Module:
    from braindecode.models import InterpolatedLaBraM

    source, revision = _pretrained_source(
        local_directory,
        repo=LABRAM_REPO,
        revision=LABRAM_REVISION,
    )
    model = InterpolatedLaBraM(
        n_times=n_times,
        n_outputs=n_outputs,
        chs_info=chs_info,
        patch_size=200,
    )
    if revision is None:
        weights_path = Path(source) / "model.safetensors"
    else:
        from huggingface_hub import hf_hub_download

        weights_path = Path(
            hf_hub_download(
                source,
                "model.safetensors",
                revision=revision,
            )
        )
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)

    from safetensors.torch import load_file

    state = load_file(str(weights_path))
    target_state = model.state_dict()
    adapted = {}
    skipped = set()
    for name, value in state.items():
        target = target_state.get(name)
        if target is None:
            skipped.add(name)
            continue
        if value.shape == target.shape:
            adapted[name] = value
        elif name == "temporal_embedding":
            if value.shape[0] != target.shape[0] or value.shape[2:] != target.shape[2:]:
                raise RuntimeError(
                    "LaBraM temporal embedding is incompatible with the target model."
                )
            adapted[name] = value[:, : target.shape[1]].clone()
        elif name.startswith("final_layer."):
            skipped.add(name)
        else:
            raise RuntimeError(
                f"Unexpected LaBraM checkpoint shape for {name}: "
                f"{tuple(value.shape)} vs {tuple(target.shape)}."
            )
    missing, unexpected = model.load_state_dict(adapted, strict=False)
    allowed_missing = {
        name
        for name in target_state
        if name.startswith("final_layer.")
        or name.startswith("interpolation_layer.")
    }
    unexplained_missing = set(missing) - allowed_missing
    if unexplained_missing or unexpected:
        raise RuntimeError(
            "LaBraM checkpoint did not match the official architecture: "
            f"missing={sorted(unexplained_missing)}, "
            f"unexpected={sorted(unexpected)}, skipped={sorted(skipped)}"
        )
    return _prepare_lora(model)


def trim_to_patch_size(X, patch_size: int):
    usable = X.shape[-1] // patch_size * patch_size
    if usable == 0:
        raise ValueError(
            f"Input has {X.shape[-1]} samples, fewer than one {patch_size}-sample patch."
        )
    return X[..., :usable]
