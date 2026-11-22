from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType


@dataclass(frozen=True)
class ConditionSpec:
    id: str
    label: str
    event_codes: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Condition id cannot be empty.")
        if not self.event_codes:
            raise ValueError(f"Condition {self.id!r} must define at least one event code.")
        if len(set(self.event_codes)) != len(self.event_codes):
            raise ValueError(f"Condition {self.id!r} contains duplicate event codes.")


@dataclass(frozen=True)
class TaskSpec:
    id: str
    classes: tuple[ConditionSpec, ...]
    analysis_window: tuple[float, float] = (0.0, 1.5)
    min_trials_per_class: int = 20
    random_seed: int = 42

    def __post_init__(self) -> None:
        if len(self.classes) < 2:
            raise ValueError(f"Task {self.id!r} must contain at least two classes.")
        class_ids = [condition.id for condition in self.classes]
        if len(set(class_ids)) != len(class_ids):
            raise ValueError(f"Task {self.id!r} contains duplicate class ids.")
        all_codes = [code for condition in self.classes for code in condition.event_codes]
        if len(set(all_codes)) != len(all_codes):
            raise ValueError(f"Task {self.id!r} assigns an event code to multiple classes.")
        tmin, tmax = self.analysis_window
        if tmin >= tmax:
            raise ValueError("analysis_window must satisfy tmin < tmax.")
        if self.min_trials_per_class < 1:
            raise ValueError("min_trials_per_class must be positive.")

    @property
    def name(self) -> str:
        return self.id

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def class_ids(self) -> tuple[str, ...]:
        return tuple(condition.id for condition in self.classes)

    @property
    def class_labels(self) -> tuple[str, ...]:
        return tuple(condition.label for condition in self.classes)

    @property
    def event_codes(self) -> Mapping[str, tuple[int, ...]]:
        return MappingProxyType({condition.id: condition.event_codes for condition in self.classes})

    @property
    def event_to_class(self) -> Mapping[int, int]:
        return MappingProxyType(
            {
                code: class_index
                for class_index, condition in enumerate(self.classes)
                for code in condition.event_codes
            }
        )

    @property
    def cond_events(self) -> dict[str, list[str]]:
        return {
            condition.id: [f"event_{code}" for code in condition.event_codes]
            for condition in self.classes
        }

    @property
    def class_0(self) -> str:
        self.require_binary()
        return self.classes[0].id

    @property
    def class_1(self) -> str:
        self.require_binary()
        return self.classes[1].id

    @property
    def undersample_classes(self) -> bool:
        return False

    def require_binary(self) -> None:
        if self.n_classes != 2:
            raise ValueError(f"Task {self.id!r} has {self.n_classes} classes; a binary task is required.")

    def with_window(self, analysis_window: tuple[float, float]) -> TaskSpec:
        return replace(self, analysis_window=analysis_window)


CONDITIONS: Mapping[str, ConditionSpec] = MappingProxyType(
    {
        "c": ConditionSpec("c", "Congruent", (1, 2)),
        "ip": ConditionSpec("ip", "Incongruent Pure", (3, 4, 5, 6)),
        "im": ConditionSpec("im", "Incongruent Mixed", (7, 8)),
        "cr": ConditionSpec("cr", "Congruent Red", (1,)),
        "cg": ConditionSpec("cg", "Congruent Green", (2,)),
        "ipr": ConditionSpec("ipr", "Incongruent Pure Red", (3, 4)),
        "ipg": ConditionSpec("ipg", "Incongruent Pure Green", (5, 6)),
        "r": ConditionSpec("r", "Red Percept", (1, 3, 4)),
        "g": ConditionSpec("g", "Green Percept", (2, 5, 6)),
    }
)


TASKS: Mapping[str, TaskSpec] = MappingProxyType(
    {
        "c_vs_ip": TaskSpec("c_vs_ip", (CONDITIONS["c"], CONDITIONS["ip"])),
        "im_vs_ip": TaskSpec("im_vs_ip", (CONDITIONS["im"], CONDITIONS["ip"])),
        "c_vs_ip_vs_im": TaskSpec(
            "c_vs_ip_vs_im",
            (CONDITIONS["c"], CONDITIONS["ip"], CONDITIONS["im"]),
        ),
        "ipr_vs_ipg": TaskSpec("ipr_vs_ipg", (CONDITIONS["ipr"], CONDITIONS["ipg"])),
        "cr_vs_cg": TaskSpec("cr_vs_cg", (CONDITIONS["cr"], CONDITIONS["cg"])),
        "r_vs_g": TaskSpec("r_vs_g", (CONDITIONS["r"], CONDITIONS["g"])),
    }
)

TASK_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "c_vs_i": "c_vs_ip",
        "cip": "c_vs_ip",
        "imip": "im_vs_ip",
        "ipr_vs_ipg": "ipr_vs_ipg",
    }
)

EVIDENCE_GATE_EXEMPT_TASKS = frozenset({"im_vs_ip"})


def get_task(task_id: str, *, analysis_window: tuple[float, float] | None = None) -> TaskSpec:
    canonical_id = TASK_ALIASES.get(task_id, task_id)
    try:
        task = TASKS[canonical_id]
    except KeyError as exc:
        available = ", ".join(TASKS)
        raise ValueError(f"Unknown task {task_id!r}. Available tasks: {available}") from exc
    return task.with_window(analysis_window) if analysis_window is not None else task
