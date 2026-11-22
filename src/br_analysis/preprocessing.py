from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType

SHARED_ANALYSIS_WINDOW = (0.0, 1.5)
POWER_ANALYSIS_WINDOW = (-0.74, 1.5)
POWER_BASELINE = (-0.74, -0.24)
ERP_ANALYSIS_WINDOW = (-0.74, 1.5)
ERP_BASELINE = (-0.74, -0.24)


@dataclass(frozen=True)
class PreprocessingPreset:
    id: str
    bandpass: tuple[float, float] | None
    resample_sfreq: float | None
    picks: tuple[str, ...] | None = None

    def with_picks(self, picks: Sequence[str] | None) -> PreprocessingPreset:
        normalized = tuple(picks) if picks is not None else None
        return replace(self, picks=normalized)


PREPROCESSING_PRESETS: Mapping[str, PreprocessingPreset] = MappingProxyType(
    {
        "broadband": PreprocessingPreset(
            id="broadband",
            bandpass=(1.0, 40.0),
            resample_sfreq=128.0,
        ),
        "band_limited": PreprocessingPreset(
            id="band_limited",
            bandpass=(4.0, 13.0),
            resample_sfreq=128.0,
        ),
        "pretrained": PreprocessingPreset(
            id="pretrained",
            bandpass=(1.0, 40.0),
            resample_sfreq=128.0,
        ),
        "labram_pretrained": PreprocessingPreset(
            id="labram_pretrained",
            bandpass=(1.0, 40.0),
            resample_sfreq=200.0,
        ),
        "erp": PreprocessingPreset(
            id="erp",
            bandpass=(0.1, 30.0),
            resample_sfreq=250.0,
        ),
    }
)


def get_preprocessing_preset(
    preset_id: str,
    *,
    picks: Sequence[str] | None = None,
) -> PreprocessingPreset:
    try:
        preset = PREPROCESSING_PRESETS[preset_id]
    except KeyError as exc:
        available = ", ".join(PREPROCESSING_PRESETS)
        raise ValueError(f"Unknown preprocessing preset {preset_id!r}. Available presets: {available}") from exc
    return preset.with_picks(picks) if picks is not None else preset


def get_cache_analysis_window(
    analysis_window: tuple[float, float],
) -> tuple[float, float]:
    tmin, tmax = analysis_window
    shared_tmin, shared_tmax = SHARED_ANALYSIS_WINDOW
    if shared_tmin <= tmin and tmax <= shared_tmax:
        return SHARED_ANALYSIS_WINDOW
    return analysis_window
