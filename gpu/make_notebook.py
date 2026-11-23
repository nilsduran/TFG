"""Build ``kaggle_screen.ipynb``: a Kaggle notebook with ``deep_pooled.py`` embedded.

Embedding the script means the Kaggle dataset only has to hold the data files;
a new experiment is just a new notebook import.

    python gpu/make_notebook.py            # writes gpu/kaggle_screen.ipynb
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

INTRO = """# Binocular rivalry EEG: deep-learning screen (two GPUs in parallel)

1. **Settings** (right panel): Accelerator = **GPU T4 x2**, Internet = **On**.
2. **Add Input**: your private dataset with `c_vs_ip.npz` and `im_vs_ip.npz` (the script is embedded below).
3. **Save Version -> Save & Run All (Commit)**. It runs in the background (about 6-10 h); you can close the browser.
4. When it finishes, open the version, go to **Output** and download `results.zip`.

Quick test first? Set `QUICK = True` in the launch cell (about 10 minutes)."""

LAUNCH = r'''import os, subprocess, sys, time

QUICK = False          # True: 3 subjects, 2 epochs, a handful of configs (smoke test, ~10 min)
BUDGET_HOURS = 10.0    # per GPU; Kaggle stops a run at 12 h

extra = ["--max-subjects", "3", "--epochs", "2", "--keep2", "2", "--keep3", "2",
         "--only", "m_eeginception", "win_0.15-0.65", "aux3"] if QUICK else []
procs = {}
for gpu, task in enumerate(["c_vs_ip", "im_vs_ip"]):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu % max(1, NGPU)))
    log = open(f"/kaggle/working/screen_{task}.log", "w")
    cmd = [sys.executable, "-u", "/kaggle/working/deep_pooled.py", "--data-dir", DATA,
           "--out", "/kaggle/working/results", "--screen", "--tasks", task,
           "--budget-hours", str(BUDGET_HOURS), *extra]
    procs[task] = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)

seen = {task: 0 for task in procs}
def flush():
    for task in procs:
        lines = open(f"/kaggle/working/screen_{task}.log").read().splitlines()
        for line in lines[seen[task]:]:
            if "warning" not in line.lower() and not line.startswith(" "):
                print(line, flush=True)
        seen[task] = len(lines)

while any(p.poll() is None for p in procs.values()):
    time.sleep(60)
    flush()
flush()
print({task: p.returncode for task, p in procs.items()})'''

CELLS = [
    ("markdown", INTRO),
    ("code", '!pip install -q "braindecode==1.5.1"\n!nvidia-smi --query-gpu=name,memory.total --format=csv'),
    ("code", "%%writefile /kaggle/working/deep_pooled.py\n{script}"),
    ("code", "import glob, os, torch\n"
             "DATA = os.path.dirname(glob.glob('/kaggle/input/**/c_vs_ip.npz', recursive=True)[0])\n"
             "NGPU = torch.cuda.device_count()\n"
             "print('data folder:', DATA, os.listdir(DATA), '| GPUs:', NGPU)"),
    ("code", LAUNCH),
    ("code", "!cd /kaggle/working && cp screen_*.log results/ && zip -qr results.zip results && ls -la results.zip"),
]


def build(out_path: Path = HERE / "kaggle_screen.ipynb") -> Path:
    script = (HERE / "deep_pooled.py").read_text(encoding="utf-8")
    cells = []
    for index, (kind, source) in enumerate(CELLS):
        text = source.replace("{script}", script) if "{script}" in source else source
        cell = {"cell_type": kind, "id": f"cell{index}", "metadata": {}, "source": text.splitlines(keepends=True)}
        if kind == "code":
            cell |= {"execution_count": None, "outputs": []}
        cells.append(cell)
    notebook = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    print(build(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "kaggle_screen.ipynb"))
