from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "src" / "notebooks"
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
MODEL_REPORTS = {
    f"{index:02d}_{name}.ipynb"
    for index, name in (
        (4, "logistic_baselines"),
        (5, "riemann"),
        (6, "svc"),
        (7, "eegnet"),
        (8, "st_gnn"),
        (9, "st_eegformer"),
        (10, "labram"),
    )
}
FORBIDDEN_PATTERNS = (
    "google.colab",
    "drive.mount",
    "MyDrive",
    "shortcut-targets",
    "COLAB_GPU",
    "/content/drive",
    "/kaggle/",
)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_structure(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    assert all(
        not "".join(cell["source"]).endswith("\n\n")
        and (not cell["source"] or cell["source"][-1].strip())
        for cell in code_cells
    )

    for index, cell in enumerate(code_cells):
        tree = ast.parse("".join(cell["source"]))
        imports = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        if index > 0:
            assert not imports

    source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
    )
    for pattern in FORBIDDEN_PATTERNS:
        assert pattern not in source

    assert "Scientific question" in source
    assert "Interpretation" in source
    if path.name in MODEL_REPORTS:
        for section in (
            "Model rationale",
            "Selected task, cohort and configuration",
            "Reproducible CLI commands",
            "Per-subject and aggregate metrics",
            "Confusion matrix and prediction diagnostics",
            "Model-specific visualisation",
            "baseline comparison",
        ):
            assert section in source
