from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SubjectExclusion:
    subject: str
    reason: str
    value: str | int | float | None = None

    def __str__(self) -> str:
        suffix = "" if self.value is None else f" ({self.value})"
        return f"{self.subject}: {self.reason}{suffix}"


@dataclass
class EpochDataset:
    X: np.ndarray
    y: np.ndarray
    subjects: np.ndarray
    trial_indices: np.ndarray
    channel_names: list[str]
    times: np.ndarray
    sfreq: float
    task_id: str
    class_ids: tuple[str, ...]
    class_labels: tuple[str, ...]
    exclusions: list[SubjectExclusion]
    info: Any | None = None
    preprocessing_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n_trials = len(self.y)
        if self.X.shape[0] != n_trials:
            raise ValueError("X and y must contain the same number of trials.")
        if len(self.subjects) != n_trials or len(self.trial_indices) != n_trials:
            raise ValueError("subjects and trial_indices must align with y.")
        if self.X.ndim != 3:
            raise ValueError("EpochDataset.X must have shape (trials, channels, times).")
        if self.X.shape[1] != len(self.channel_names):
            raise ValueError("channel_names must align with the channel axis.")
        if self.X.shape[2] != len(self.times):
            raise ValueError("times must align with the time axis.")

    @property
    def subject_ids(self) -> list[str]:
        return list(dict.fromkeys(self.subjects.tolist()))

    @property
    def skipped(self) -> list[str]:
        return [str(exclusion) for exclusion in self.exclusions]

    @property
    def class_counts(self) -> dict[str, int]:
        return {
            class_id: int(np.sum(self.y == class_index))
            for class_index, class_id in enumerate(self.class_ids)
        }
