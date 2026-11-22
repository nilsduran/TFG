from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .config import ProjectPaths
from .preprocessing import (
    ERP_ANALYSIS_WINDOW,
    get_cache_analysis_window,
    get_preprocessing_preset,
)
from .registry import MODELS, get_model_spec


@dataclass(frozen=True)
class PreparationSpec:
    preset_id: str
    bandpass: tuple[float, float] | None
    resample_sfreq: float | None
    analysis_window: tuple[float, float]
    model_ids: tuple[str, ...]


def preparation_specs(model_ids: Sequence[str] | None = None) -> tuple[PreparationSpec, ...]:
    selected_models = list(model_ids) if model_ids is not None else list(MODELS)
    grouped: dict[
        tuple[tuple[float, float] | None, float | None, tuple[float, float]],
        tuple[str, list[str]],
    ] = {}
    for model_id in selected_models:
        model = get_model_spec(model_id)
        preset = get_preprocessing_preset(model.preprocessing)
        cache_window = get_cache_analysis_window(model.analysis_window)
        key = (preset.bandpass, preset.resample_sfreq, cache_window)
        _, grouped_models = grouped.setdefault(key, (preset.id, []))
        grouped_models.append(model.id)
    if model_ids is None:
        preset = get_preprocessing_preset("erp")
        key = (
            preset.bandpass,
            preset.resample_sfreq,
            ERP_ANALYSIS_WINDOW,
        )
        grouped[key] = (preset.id, ["temporal_onset"])

    return tuple(
        PreparationSpec(
            preset_id=preset_id,
            bandpass=bandpass,
            resample_sfreq=resample_sfreq,
            analysis_window=analysis_window,
            model_ids=tuple(grouped_models),
        )
        for (
            bandpass,
            resample_sfreq,
            analysis_window,
        ), (preset_id, grouped_models) in grouped.items()
    )


def prepare_all_data(
    paths: ProjectPaths,
    *,
    model_ids: Sequence[str] | None = None,
    force: bool = False,
    include_power: bool = True,
) -> tuple[PreparationSpec, ...]:
    from .data import prepare_epoch_cache

    specs = preparation_specs(model_ids)
    for index, spec in enumerate(specs, start=1):
        models = ", ".join(spec.model_ids)
        print(
            f"[{index}/{len(specs)}] {spec.preset_id} {spec.analysis_window} "
            f"for {models}",
            flush=True,
        )
        prepare_epoch_cache(
            paths,
            spec.preset_id,
            spec.analysis_window,
            refresh_cache=force,
        )
    if include_power:
        from .power import prepare_power_cache

        print("Preparing baseline-corrected theta/alpha power.", flush=True)
        prepare_power_cache(paths, force=force)
    print(
        f"Prepared {len(specs)} shared preprocessing variants"
        f"{' and power' if include_power else ''}."
    )
    return specs
