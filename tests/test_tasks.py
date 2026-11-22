from __future__ import annotations

import pytest

from br_analysis.tasks import TASKS, get_task


@pytest.mark.parametrize(
    ("task_id", "expected_codes"),
    [
        ("c_vs_ip", ((1, 2), (3, 4, 5, 6))),
        ("im_vs_ip", ((7, 8), (3, 4, 5, 6))),
        ("c_vs_ip_vs_im", ((1, 2), (3, 4, 5, 6), (7, 8))),
        ("ipr_vs_ipg", ((3, 4), (5, 6))),
        ("cr_vs_cg", ((1,), (2,))),
        ("r_vs_g", ((1, 3, 4), (2, 5, 6))),
    ],
)
def test_task_event_codes(task_id: str, expected_codes: tuple[tuple[int, ...], ...]) -> None:
    task = get_task(task_id)
    assert tuple(condition.event_codes for condition in task.classes) == expected_codes


def test_task_event_codes_do_not_overlap() -> None:
    for task in TASKS.values():
        codes = [code for condition in task.classes for code in condition.event_codes]
        assert len(codes) == len(set(codes))


def test_binary_compatibility_properties() -> None:
    task = get_task("c_vs_ip")
    assert task.class_0 == "c"
    assert task.class_1 == "ip"
    assert task.undersample_classes is False


def test_multiclass_task_rejects_binary_properties() -> None:
    with pytest.raises(ValueError, match="binary task"):
        _ = get_task("c_vs_ip_vs_im").class_0


def test_task_alias() -> None:
    assert get_task("c_vs_i") is get_task("c_vs_ip")
