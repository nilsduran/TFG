from __future__ import annotations

from pathlib import Path

FORBIDDEN_PATTERNS = (
    "google.colab",
    "drive.mount",
    "MyDrive",
    "shortcut-targets",
    "COLAB_GPU",
    "/content/drive",
    "/kaggle/",
)


def test_package_has_no_platform_specific_paths() -> None:
    package_dir = Path(__file__).resolve().parents[1] / "src" / "br_analysis"
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in package_dir.glob("*.py")
        if path.name != "kaggle.py"
    )
    for pattern in FORBIDDEN_PATTERNS:
        assert pattern not in source
