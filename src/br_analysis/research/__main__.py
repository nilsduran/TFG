"""Command-line entry point: ``python -m br_analysis.research``.

Examples::

    python -m br_analysis.research run --tasks c_vs_ip im_vs_ip --candidates riemann xdawn_ts
    python -m br_analysis.research pooled --tasks c_vs_ip --candidates erp_lda
    python -m br_analysis.research summary --tasks c_vs_ip im_vs_ip
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .harness import export_for_gpu, import_gpu_results, run_many
from .pipelines import CANDIDATES
from .pooled import run_pooled
from .summary import format_summary, summarize_task


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m br_analysis.research")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "pooled"):
        p = sub.add_parser(name)
        p.add_argument("--tasks", nargs="+", default=["c_vs_ip", "im_vs_ip"])
        p.add_argument("--candidates", nargs="+", default=list(CANDIDATES))
        p.add_argument("--jobs", type=int, default=8 if name == "run" else 4)
        p.add_argument("--force", action="store_true")
        if name == "pooled":
            p.add_argument("--target-repeat", type=int, default=1)
    p = sub.add_parser("export", help="write stimulus-window data for the GPU script")
    p.add_argument("--tasks", nargs="+", default=["c_vs_ip", "im_vs_ip"])
    p = sub.add_parser("import-gpu", help="import results downloaded from the GPU run")
    p.add_argument("source", type=Path)
    p = sub.add_parser("summary")
    p.add_argument("--tasks", nargs="+", default=["c_vs_ip", "im_vs_ip"])
    p.add_argument("--reference", default="riemann")
    args = parser.parse_args(argv)

    if args.command == "run":
        run_many(args.tasks, args.candidates, n_jobs=args.jobs, force=args.force)
    elif args.command == "pooled":
        for task_id in args.tasks:
            for candidate_id in args.candidates:
                table = run_pooled(
                    task_id, candidate_id, target_repeat=args.target_repeat, n_jobs=args.jobs, force=args.force
                )
                print(
                    f"{task_id:>10} pooled_ea_{candidate_id:<12} dev={table.dev_bacc.mean():.3f} "
                    f"holdout={table.holdout_bacc.mean():.3f} ± {table.holdout_bacc.std():.3f}",
                    flush=True,
                )
    elif args.command == "export":
        print(f"GPU bundle ready in {export_for_gpu(args.tasks)}")
    elif args.command == "import-gpu":
        for path in import_gpu_results(args.source):
            print(f"imported {path}")
    else:
        for task_id in args.tasks:
            print(f"\n=== {task_id} ===")
            print(format_summary(summarize_task(task_id, reference=args.reference)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
