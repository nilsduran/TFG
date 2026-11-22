from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from tqdm.auto import tqdm

from .registry import MODELS, get_model_spec
from .tasks import EVIDENCE_GATE_EXEMPT_TASKS, TASKS


def _completed_run(
    artifacts_root: Path,
    task_id: str,
    model_id: str,
) -> bool:
    return _run_status(artifacts_root, task_id, model_id) == "X"


def _run_status(
    artifacts_root: Path,
    task_id: str,
    model_id: str,
) -> str:
    directory = artifacts_root / task_id / model_id
    metadata_path = directory / "metadata.json"
    required = (
        metadata_path,
        directory / "metrics.csv",
        directory / "predictions.npz",
    )
    if not all(path.is_file() for path in required):
        return "-"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "-"
    from .splits import EVALUATION_PROTOCOL

    if (
        metadata.get("task_id") != task_id
        or metadata.get("model_id") != model_id
    ):
        return "-"
    if (
        metadata.get("extra", {}).get("evaluation_protocol")
        == EVALUATION_PROTOCOL
    ):
        return "X"
    return "L"


def _run_coverage_table() -> str:
    from .config import ProjectPaths

    artifacts_root = ProjectPaths.from_env("help", create_dirs=False).artifacts_dir
    headers = ("Model", *TASKS)
    rows = [
        (
            model_id,
            *(
                _run_status(artifacts_root, task_id, model_id)
                for task_id in TASKS
            ),
        )
        for model_id in MODELS
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def format_row(row: tuple[str, ...]) -> str:
        return "  " + "  ".join(
            value.ljust(widths[index]) for index, value in enumerate(row)
        )

    separator = tuple("-" * width for width in widths)
    return "\n".join(
        [format_row(headers), format_row(separator)]
        + [format_row(row) for row in rows]
    )


def _help_epilog() -> str:
    return (
        "example runs:\n"
        "  br-analysis run --task c_vs_ip --model riemann\n"
        "  br-analysis run --task all --model riemann\n"
        "  br-analysis run --task c_vs_ip --model all\n"
        "  br-analysis temporal-onset --task all --jobs 4\n"
        "  br-analysis inter-subject --task all\n\n"
        "run status (X=current final holdout, L=legacy development CV):\n"
        f"{_run_coverage_table()}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="br-analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_help_epilog(),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("help", help="Show this help message.")
    subparsers.add_parser("list-tasks", help="List configured decoding tasks.")
    subparsers.add_parser("list-models", help="List configured models.")

    prepare_parser = subparsers.add_parser(
        "prepare-data",
        help="Cache every preprocessing variant shared by all tasks.",
    )
    prepare_parser.add_argument("--models", nargs="+")
    prepare_parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild preprocessing caches even when they already exist.",
    )
    prepare_parser.add_argument(
        "--skip-power",
        action="store_true",
        help="Prepare signal caches only, without theta/alpha power.",
    )

    onset_parser = subparsers.add_parser(
        "temporal-onset",
        help="Estimate and independently confirm pointwise ERP decoding onset.",
    )
    onset_parser.add_argument("--task", default="all", help="Task ID or 'all'.")
    onset_parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel subject workers (recommended: 4).",
    )
    onset_parser.add_argument(
        "--permutations",
        type=int,
        default=2000,
        help="Max-T, individual and confirmation permutations.",
    )
    onset_parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Hierarchical onset bootstrap samples.",
    )
    onset_parser.add_argument(
        "--bootstrap-trial-replicates",
        type=int,
        default=100,
        help="Cached trial-resampling curves per subject.",
    )
    onset_parser.add_argument(
        "--pseudo-iterations",
        type=int,
        default=100,
        help="Pseudo-trial sensitivity repetitions; use 0 to disable.",
    )
    onset_parser.add_argument("--force", action="store_true")

    inter_parser = subparsers.add_parser(
        "inter-subject",
        help="Evaluate pooled and aligned models on held-out subjects.",
    )
    inter_parser.add_argument("--task", default="all", help="Task ID or 'all'.")
    inter_parser.add_argument("--permutations", type=int, default=2000)
    inter_parser.add_argument("--force", action="store_true")

    validate_parser = subparsers.add_parser("validate-data", help="Validate local FIF files and event codes.")
    validate_parser.add_argument("--max-files", type=int, default=None)

    run_parser = subparsers.add_parser(
        "run",
        help="Run configured task/model combinations.",
    )
    run_parser.add_argument("--task", required=True, help="Task ID or 'all'.")
    run_parser.add_argument("--model", required=True, help="Model ID or 'all'.")
    run_parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel subject workers for classical models (recommended: 4).",
    )
    run_parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help=(
            "Optional local Hugging Face model directory, or MODEL=DIRECTORY; "
            "public checkpoints download automatically."
        ),
    )
    run_parser.add_argument("--explain", action="store_true")
    run_parser.add_argument(
        "--allow-unpromising",
        action="store_true",
        help="Allow deep-learning or SHAP runs even when classical baselines are at chance.",
    )
    run_parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute the run even when compatible artifacts already exist.",
    )

    report_parser = subparsers.add_parser("report", help="Generate the common HTML results report.")
    report_parser.add_argument("--tasks", nargs="+")
    report_parser.add_argument("--output", type=Path)

    kaggle_parser = subparsers.add_parser(
        "kaggle",
        help="Run GPU models remotely through Kaggle.",
    )
    kaggle_commands = kaggle_parser.add_subparsers(
        dest="kaggle_command",
        required=True,
    )
    source_parser = kaggle_commands.add_parser(
        "publish-source",
        help="Create or update the private source-code dataset.",
    )
    source_parser.add_argument("--ref", required=True, help="Kaggle owner/slug.")
    source_parser.add_argument("--update", action="store_true")

    data_parser = kaggle_commands.add_parser(
        "publish-data",
        help="Create or update the private preprocessed EEG dataset.",
    )
    data_parser.add_argument("--ref", required=True, help="Kaggle owner/slug.")
    data_parser.add_argument(
        "--models",
        nargs="+",
        help="Torch models to include (default: st_eegformer labram).",
    )
    data_parser.add_argument("--update", action="store_true")

    submit_parser = kaggle_commands.add_parser(
        "submit",
        help="Submit one task/model run to a Kaggle GPU.",
    )
    submit_parser.add_argument("--kernel", required=True, help="Kaggle owner/slug.")
    submit_parser.add_argument("--source-dataset", required=True)
    submit_parser.add_argument("--data-dataset", required=True)
    submit_parser.add_argument("--task", required=True)
    submit_parser.add_argument("--model", required=True)
    submit_parser.add_argument("--allow-unpromising", action="store_true")

    status_parser = kaggle_commands.add_parser(
        "status",
        help="Show the remote kernel status.",
    )
    status_parser.add_argument("--kernel", required=True)

    download_parser = kaggle_commands.add_parser(
        "download",
        help="Download and import remote artifacts.",
    )
    download_parser.add_argument("--kernel", required=True)
    download_parser.add_argument("--replace", action="store_true")
    return parser


def _list_tasks() -> int:
    for task in TASKS.values():
        classes = ", ".join(
            f"{condition.id}={list(condition.event_codes)}"
            for condition in task.classes
        )
        print(f"{task.id}: {classes}")
    return 0


def _list_models() -> int:
    for model in MODELS.values():
        print(
            f"{model.id}: family={model.family}, "
            f"preprocessing={model.preprocessing}, "
            f"representation={model.representation}"
        )
    return 0


def _selection(value: str, available: Sequence[str], kind: str) -> list[str]:
    if value == "all":
        return list(available)
    if value not in available:
        choices = ", ".join(available)
        raise ValueError(f"Unknown {kind} {value!r}. Available: {choices}, all")
    return [value]


def _checkpoint_paths(
    values: Sequence[str],
    model_ids: Sequence[str],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    unnamed: list[str] = []
    for value in values:
        if "=" in value:
            model_id, raw_path = value.split("=", 1)
            if model_id not in {"st_eegformer", "labram"}:
                raise ValueError(f"Checkpoint model must be st_eegformer or labram, got {model_id!r}.")
            paths[model_id] = Path(raw_path)
        else:
            unnamed.append(value)
    required_targets = [
        model_id
        for model_id in model_ids
        if model_id in {"st_eegformer", "labram"}
    ]
    if unnamed:
        if len(unnamed) > 1 or len(required_targets) != 1:
            raise ValueError(
                "Unnamed --checkpoint PATH requires exactly one pretrained target model. "
                "Use --checkpoint MODEL=PATH for batch runs."
            )
        paths[required_targets[0]] = Path(unnamed[0])
    return paths


def _run_combination(
    *,
    task_id: str,
    model_id: str,
    checkpoint: Path | None,
    explain: bool,
    force: bool,
    n_jobs: int,
) -> Path:
    model_spec = get_model_spec(model_id)
    if model_spec.family == "sklearn":
        from .benchmark import run_classical_benchmark

        return run_classical_benchmark(
            task_id=task_id,
            model_id=model_id,
            explain=explain,
            force=force,
            n_jobs=n_jobs,
        )

    from .deep_benchmark import run_torch_benchmark

    return run_torch_benchmark(
        task_id=task_id,
        model_id=model_id,
        checkpoint=checkpoint,
        explain=explain,
        force=force,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "help":
        parser.print_help()
        return 0
    if args.command == "list-tasks":
        return _list_tasks()
    if args.command == "list-models":
        return _list_models()
    if args.command == "prepare-data":
        from .config import ProjectPaths
        from .preparation import prepare_all_data

        paths = ProjectPaths.from_env("preparation", create_dirs=False)
        prepare_all_data(
            paths,
            model_ids=args.models,
            force=args.force,
            include_power=not args.skip_power,
        )
        return 0
    if args.command == "temporal-onset":
        from .config import ProjectPaths
        from .temporal_onset import (
            TemporalOnsetConfig,
            run_temporal_onset_analysis,
        )

        try:
            task_ids = _selection(args.task, tuple(TASKS), "task")
            if args.jobs < 1:
                raise ValueError("--jobs must be at least one.")
            if args.permutations < 1:
                raise ValueError("--permutations must be at least one.")
            if args.bootstrap_samples < 1:
                raise ValueError("--bootstrap-samples must be at least one.")
            if args.bootstrap_trial_replicates < 1:
                raise ValueError(
                    "--bootstrap-trial-replicates must be at least one."
                )
            if args.pseudo_iterations < 0:
                raise ValueError("--pseudo-iterations cannot be negative.")
        except ValueError as exc:
            parser.error(str(exc))
        output = run_temporal_onset_analysis(
            paths=ProjectPaths.from_env("temporal_onset"),
            task_ids=task_ids,
            config=TemporalOnsetConfig(
                permutations=args.permutations,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_trial_replicates=args.bootstrap_trial_replicates,
                pseudo_iterations=args.pseudo_iterations,
            ),
            n_jobs=args.jobs,
            force=args.force,
        )
        print(output)
        return 0
    if args.command == "inter-subject":
        from .config import ProjectPaths
        from .inter_subject import (
            InterSubjectConfig,
            run_inter_subject_analysis,
        )

        try:
            task_ids = _selection(args.task, tuple(TASKS), "task")
            if args.permutations < 1:
                raise ValueError("--permutations must be at least one.")
        except ValueError as exc:
            parser.error(str(exc))
        output = run_inter_subject_analysis(
            paths=ProjectPaths.from_env("inter_subject"),
            task_ids=task_ids,
            config=InterSubjectConfig(permutations=args.permutations),
            force=args.force,
        )
        print(output)
        return 0
    if args.command == "validate-data":
        from .validation import validate_data

        return validate_data(args.max_files)
    if args.command == "run":
        try:
            task_ids = _selection(args.task, tuple(TASKS), "task")
            model_ids = _selection(args.model, tuple(MODELS), "model")
            checkpoints = _checkpoint_paths(args.checkpoint, model_ids)
            if args.jobs < 1:
                raise ValueError("--jobs must be at least one.")
        except ValueError as exc:
            parser.error(str(exc))
        if args.jobs > 1 and any(
            MODELS[model_id].family == "torch"
            for model_id in model_ids
        ):
            print(
                "Note: --jobs applies to classical models only; "
                "deep models remain sequential."
            )

        combinations = [
            (task_id, model_id)
            for model_id in model_ids
            for task_id in task_ids
        ]
        is_batch = len(combinations) > 1
        artifacts_root = None
        if is_batch and not args.force:
            from .config import ProjectPaths

            artifacts_root = ProjectPaths.from_env(
                "batch",
                create_dirs=False,
            ).artifacts_dir
        completed = 0
        skipped = 0
        failed = 0
        batch_progress = tqdm(
            total=len(combinations),
            desc="Task/model combinations",
            unit="run",
            dynamic_ncols=True,
            disable=None if is_batch else True,
        )
        for index, (task_id, model_id) in enumerate(combinations, start=1):
            batch_progress.set_postfix_str(
                f"{task_id}/{model_id}",
                refresh=True,
            )
            print(f"[{index}/{len(combinations)}] {task_id} / {model_id}")
            if artifacts_root is not None and _completed_run(
                artifacts_root,
                task_id,
                model_id,
            ):
                print("  SKIP: completed artifacts already exist (use --force to retrain)")
                skipped += 1
                batch_progress.update()
                continue
            if (
                MODELS[model_id].family == "torch"
                or args.explain
            ) and not args.allow_unpromising and (
                task_id not in EVIDENCE_GATE_EXEMPT_TASKS
            ):
                from .config import ProjectPaths
                from .reporting import classical_task_evidence

                evidence_root = (
                    artifacts_root
                    or ProjectPaths.from_env(
                        "eligibility",
                        create_dirs=False,
                    ).artifacts_dir
                )
                evidence = classical_task_evidence(evidence_root, task_id)
                if evidence is False:
                    print(
                        "  SKIP: classical baselines are effectively at chance "
                        "(use --allow-unpromising to override)"
                    )
                    skipped += 1
                    batch_progress.update()
                    continue
            try:
                directory = _run_combination(
                    task_id=task_id,
                    model_id=model_id,
                    checkpoint=checkpoints.get(model_id),
                    explain=args.explain,
                    force=args.force,
                    n_jobs=args.jobs,
                )
            except Exception as exc:
                if not is_batch:
                    raise
                print(f"  FAIL: {type(exc).__name__}: {exc}")
                failed += 1
                batch_progress.update()
                continue
            print(f"  DONE: {directory}")
            completed += 1
            batch_progress.update()
        batch_progress.close()
        if is_batch:
            print(
                f"Batch complete: {completed} completed, "
                f"{skipped} skipped, {failed} failed."
            )
        return 1 if failed else 0
    if args.command == "report":
        from .config import ProjectPaths
        from .report import generate_html_report

        paths = ProjectPaths.from_env("report", create_dirs=False)
        report_path = generate_html_report(
            paths.artifacts_dir,
            output_path=args.output,
            task_ids=args.tasks,
        )
        print(report_path)
        return 0
    if args.command == "kaggle":
        from .config import ProjectPaths
        from .kaggle import (
            download_kernel_outputs,
            kernel_status,
            publish_data_dataset,
            publish_source_dataset,
            submit_kernel,
        )

        paths = ProjectPaths.from_env("kaggle", create_dirs=False)
        if args.kaggle_command == "publish-source":
            publish_source_dataset(paths, args.ref, update=args.update)
        elif args.kaggle_command == "publish-data":
            publish_data_dataset(
                paths,
                args.ref,
                model_ids=args.models,
                update=args.update,
            )
        elif args.kaggle_command == "submit":
            submit_kernel(
                paths,
                kernel_reference=args.kernel,
                source_dataset=args.source_dataset,
                data_dataset=args.data_dataset,
                task_id=args.task,
                model_id=args.model,
                allow_unpromising=args.allow_unpromising,
            )
        elif args.kaggle_command == "status":
            kernel_status(args.kernel)
        elif args.kaggle_command == "download":
            download_kernel_outputs(
                paths,
                args.kernel,
                replace=args.replace,
            )
        else:
            raise RuntimeError(
                f"Unhandled Kaggle command: {args.kaggle_command}"
            )
        return 0
    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
