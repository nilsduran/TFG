from __future__ import annotations

import base64
from collections.abc import Sequence
from datetime import UTC, datetime
from html import escape
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.integrate import trapezoid
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from .registry import MODELS
from .reporting import (
    classical_task_evidence,
    collect_metrics,
    model_summary,
    subject_metrics,
)
from .splits import EVALUATION_PROTOCOL
from .tasks import EVIDENCE_GATE_EXEMPT_TASKS, TASKS


def _table(frame: pd.DataFrame, *, index: bool = False) -> str:
    if frame.empty:
        return '<p class="empty">No completed runs yet.</p>'
    return frame.to_html(
        index=index,
        border=0,
        classes="data-table",
        float_format=lambda value: f"{value:.3f}",
        na_rep="-",
    )


def _load_predictions(directory: Path) -> dict[str, np.ndarray] | None:
    predictions_path = directory / "predictions.npz"
    if not predictions_path.is_file():
        return None
    try:
        with np.load(predictions_path, allow_pickle=True) as arrays:
            predictions = {
                "y_true": arrays["y_true"],
                "probabilities": arrays["probabilities"],
            }
            predictions["subjects"] = (
                arrays["subjects"].astype(str)
                if "subjects" in arrays
                else np.full(len(arrays["y_true"]), "all")
            )
    except (OSError, ValueError, KeyError):
        return None
    return predictions


def _confusion_table(directory: Path, class_ids: tuple[str, ...]) -> str:
    predictions = _load_predictions(directory)
    if predictions is None:
        return ""
    y_true = predictions["y_true"]
    y_pred = predictions["probabilities"].argmax(axis=1)
    matrices = []
    for subject in np.unique(predictions["subjects"]):
        mask = predictions["subjects"] == subject
        matrix = confusion_matrix(
            y_true[mask],
            y_pred[mask],
            labels=np.arange(len(class_ids)),
            normalize="true",
        )
        matrices.append(matrix)
    matrix = np.nanmean(matrices, axis=0)
    frame = pd.DataFrame(
        matrix,
        index=[f"True {class_id}" for class_id in class_ids],
        columns=[f"Pred. {class_id}" for class_id in class_ids],
    )
    return _table(frame, index=True)


def _mean_curves(
    directory: Path,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    predictions = _load_predictions(directory)
    if predictions is None:
        return None
    y_true = predictions["y_true"]
    probabilities = predictions["probabilities"]
    subjects = predictions["subjects"]
    roc_grid = np.linspace(0.0, 1.0, 101)
    recall_grid = np.linspace(0.0, 1.0, 101)
    subject_roc: list[np.ndarray] = []
    subject_precision: list[np.ndarray] = []
    for subject in np.unique(subjects):
        mask = subjects == subject
        class_roc: list[np.ndarray] = []
        class_precision: list[np.ndarray] = []
        for class_index in range(n_classes):
            binary = (y_true[mask] == class_index).astype(int)
            if len(np.unique(binary)) < 2:
                continue
            false_positive, true_positive, _ = roc_curve(
                binary,
                probabilities[mask, class_index],
            )
            precision, recall, _ = precision_recall_curve(
                binary,
                probabilities[mask, class_index],
            )
            class_roc.append(np.interp(roc_grid, false_positive, true_positive))
            class_precision.append(
                np.interp(recall_grid, recall[::-1], precision[::-1])
            )
        if class_roc:
            subject_roc.append(np.mean(class_roc, axis=0))
            subject_precision.append(np.mean(class_precision, axis=0))
    if not subject_roc:
        return None
    return (
        roc_grid,
        np.mean(subject_roc, axis=0),
        recall_grid,
        np.mean(subject_precision, axis=0),
    )


def _curve_figure(
    series: Sequence[tuple[str, Path, int]],
    *,
    title: str,
) -> str:
    figure = Figure(figsize=(10.5, 4.2), constrained_layout=True)
    axes = figure.subplots(1, 2)
    plotted = False
    for label, directory, n_classes in series:
        curves = _mean_curves(directory, n_classes)
        if curves is None:
            continue
        false_positive, true_positive, recall, precision = curves
        roc_auc = trapezoid(true_positive, false_positive)
        pr_auc = trapezoid(precision, recall)
        axes[0].plot(false_positive, true_positive, label=f"{label} ({roc_auc:.3f})")
        axes[1].plot(recall, precision, label=f"{label} ({pr_auc:.3f})")
        plotted = True
    if not plotted:
        return ""
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="0.55", linewidth=1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="Mean subject ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Mean subject precision-recall")
    for axis in axes:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle(title)
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=130)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return (
        f'<img class="metric-figure" alt="{escape(title)}" '
        f'src="data:image/png;base64,{encoded}">'
    )


def _png_image(path: Path, title: str) -> str:
    if not path.is_file():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        f'<img class="metric-figure" alt="{escape(title)}" '
        f'src="data:image/png;base64,{encoded}">'
    )


def _extension_section(
    artifacts_root: Path,
    *,
    analysis_id: str,
    title: str,
    description: str,
    task_ids: Sequence[str],
    columns: Sequence[str],
    image_name: str,
) -> str:
    root = artifacts_root / analysis_id
    summary_path = root / "summary.csv"
    if not summary_path.is_file():
        return (
            "<section>"
            f"<h2>{escape(title)}</h2>"
            f'<p class="empty">{escape(description)} Not run yet.</p>'
            "</section>"
        )
    summary = pd.read_csv(summary_path)
    summary = summary[summary["task_id"].isin(task_ids)]
    available_columns = [column for column in columns if column in summary]
    details = []
    for task_id in task_ids:
        image = _png_image(
            root / task_id / image_name,
            f"{title}: {task_id}",
        )
        if image:
            details.append(
                "<details>"
                f"<summary>{escape(task_id)}</summary>"
                f"{image}"
                "</details>"
            )
    return (
        "<section>"
        f"<h2>{escape(title)}</h2>"
        f'<p class="classes">{escape(description)}</p>'
        f"{_table(summary[available_columns])}"
        f"{''.join(details)}"
        "</section>"
    )


def _task_section(
    artifacts_root: Path,
    task_id: str,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    visible: bool,
) -> str:
    task = TASKS[task_id]
    aggregated = subject_metrics(metrics)
    task_metrics = (
        aggregated.loc[aggregated["task_id"] == task_id]
        if not aggregated.empty
        else aggregated
    )
    task_summary = summary.loc[summary["task_id"] == task_id] if not summary.empty else summary
    summary_columns = [
        "model_id",
        "balanced_accuracy_mean",
        "balanced_accuracy_ci_low",
        "balanced_accuracy_ci_high",
        "delta_chance",
        "f1_macro_mean",
        "roc_auc_mean",
        "n_subjects",
        "chance_level",
        "evaluation_protocol",
    ]
    class_description = " | ".join(
        f"{condition.label} [{', '.join(map(str, condition.event_codes))}]"
        for condition in task.classes
    )
    details: list[str] = []
    if not task_metrics.empty:
        for model_id in task_summary["model_id"]:
            model_metrics = task_metrics.loc[task_metrics["model_id"] == model_id]
            detail_columns = [
                column
                for column in (
                    "subject",
                    "accuracy",
                    "balanced_accuracy",
                    "f1_macro",
                    "roc_auc",
                    "n_samples",
                )
                if column in model_metrics
            ]
            confusion = _confusion_table(
                artifacts_root / task_id / model_id,
                task.class_ids,
            )
            confusion_content = confusion or '<p class="empty">Predictions unavailable.</p>'
            details.append(
                "<details>"
                f"<summary>{escape(model_id)}</summary>"
                "<h3>Metrics by subject</h3>"
                f"{_table(model_metrics[detail_columns])}"
                "<h3>Mean subject-normalized confusion matrix</h3>"
                f"{confusion_content}"
                "</details>"
            )
    curves = _curve_figure(
        [
            (
                model_id,
                artifacts_root / task_id / model_id,
                task.n_classes,
            )
            for model_id in task_summary["model_id"]
        ]
        if not task_summary.empty
        else [],
        title=f"{task_id}: saved prediction diagnostics",
    )
    evidence = classical_task_evidence(artifacts_root, task_id)
    if evidence is True:
        evidence_text = (
            "At least one classical baseline has a 95% confidence interval above "
            "chance and a balanced-accuracy margin of at least 0.02."
        )
    elif evidence is False:
        if task_id in EVIDENCE_GATE_EXEMPT_TASKS:
            evidence_text = (
                "Classical baselines are effectively at chance, but this task "
                "is retained as a prespecified exploratory deep-learning analysis."
            )
        else:
            evidence_text = (
                "Classical baselines are effectively at chance; deep-learning and "
                "SHAP runs are not recommended unless explicitly overridden."
            )
    else:
        evidence_text = "Not enough classical baselines are available to assess task viability."
    hidden = "" if visible else " hidden"
    return (
        f'<section id="task-{escape(task_id)}" class="task-section"{hidden}>'
        f"<h2>{escape(task_id)}</h2>"
        f'<p class="classes">{escape(class_description)}</p>'
        f'<p class="evidence">{escape(evidence_text)}</p>'
        "<h3>Model summary</h3>"
        f"{_table(task_summary[summary_columns] if not task_summary.empty else task_summary)}"
        "<h3>ROC and precision-recall diagnostics</h3>"
        f"{curves or '<p class=\"empty\">Predictions unavailable.</p>'}"
        f"{''.join(details)}"
        "</section>"
    )


def _model_curve_sections(
    artifacts_root: Path,
    task_ids: Sequence[str],
    completed: set[tuple[str, str]],
) -> str:
    sections = []
    for model_id in MODELS:
        available_tasks = [
            task_id
            for task_id in task_ids
            if (task_id, model_id) in completed
        ]
        if len(available_tasks) < 2:
            continue
        figure = _curve_figure(
            [
                (
                    task_id,
                    artifacts_root / task_id / model_id,
                    TASKS[task_id].n_classes,
                )
                for task_id in available_tasks
            ],
            title=f"{model_id}: comparison across tasks",
        )
        if figure:
            sections.append(
                "<details>"
                f"<summary>{escape(model_id)}</summary>"
                f"{figure}"
                "</details>"
            )
    if not sections:
        return '<p class="empty">Not enough completed runs for cross-task curves.</p>'
    return "".join(sections)


def generate_html_report(
    artifacts_root: Path,
    *,
    output_path: Path | None = None,
    task_ids: Sequence[str] | None = None,
) -> Path:
    selected_tasks = list(task_ids) if task_ids is not None else list(TASKS)
    unknown = sorted(set(selected_tasks) - set(TASKS))
    if unknown:
        raise ValueError(f"Unknown tasks: {', '.join(unknown)}")

    metrics = collect_metrics(artifacts_root, task_ids=selected_tasks)
    summary = model_summary(metrics)
    completed = (
        set(zip(summary["task_id"], summary["model_id"]))
        if not summary.empty
        else set()
    )
    protocols = (
        {
            (row.task_id, row.model_id): row.evaluation_protocol
            for row in summary.itertuples()
        }
        if not summary.empty
        else {}
    )
    status = pd.DataFrame(
        [
            {
                "model_id": model_id,
                **{
                    task_id: (
                        "X"
                        if protocols.get((task_id, model_id))
                        == EVALUATION_PROTOCOL
                        else "L"
                        if (task_id, model_id) in completed
                        else "-"
                    )
                    for task_id in selected_tasks
                },
            }
            for model_id in MODELS
        ]
    )
    first_visible = next(
        (task_id for task_id in selected_tasks if any(pair[0] == task_id for pair in completed)),
        selected_tasks[0],
    )
    options = "".join(
        f'<option value="{escape(task_id)}"{" selected" if task_id == first_visible else ""}>'
        f"{escape(task_id)}</option>"
        for task_id in selected_tasks
    )
    sections = "".join(
        _task_section(
            artifacts_root,
            task_id,
            metrics,
            summary,
            visible=task_id == first_visible,
        )
        for task_id in selected_tasks
    )
    model_curves = _model_curve_sections(
        artifacts_root,
        selected_tasks,
        completed,
    )
    temporal_onset = _extension_section(
        artifacts_root,
        analysis_id="temporal_onset",
        title="Temporal decoding onset",
        description=(
            "Pointwise shrinkage LDA on baseline-corrected ERPs. The first "
            "80% discovers an FWER-controlled interval and the chronological "
            "final 20% confirms that interval once."
        ),
        task_ids=selected_tasks,
        columns=(
            "task_id",
            "n_subjects",
            "discovery_onset_seconds",
            "bootstrap_ci_seconds",
            "bootstrap_detection_percent",
            "onset_stable",
            "confirmation_balanced_accuracy",
            "confirmation_p_holm",
            "confirmed",
            "subjects_with_onset_percent",
            "prestimulus_omnibus_p",
            "pseudo_trial_onset_seconds",
            "interpretation",
        ),
        image_name="onset.png",
    )
    inter_subject = _extension_section(
        artifacts_root,
        analysis_id="inter_subject",
        title="Inter-subject transfer",
        description=(
            "Held-out-subject pooled Log-Euclidean MDM compared with "
            "unsupervised Euclidean Alignment."
        ),
        task_ids=selected_tasks,
        columns=(
            "task_id",
            "n_subjects",
            "pooled_balanced_accuracy",
            "aligned_balanced_accuracy",
            "aligned_f1_macro",
            "alignment_gain",
            "alignment_paired_p",
        ),
        image_name="comparison.png",
    )
    generated_at = datetime.now(UTC).astimezone().strftime("%Y-%m-%d %H:%M %Z")
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Binocular rivalry results</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, Arial, sans-serif; color: #17202a; }}
    body {{ margin: 0; background: #f4f6f7; }}
    header {{ padding: 28px max(24px, 5vw); background: #19324a; color: white; }}
    header h1 {{ margin: 0 0 6px; font-size: 28px; letter-spacing: 0; }}
    header p {{ margin: 0; color: #d9e3ea; }}
    main {{ max-width: 1320px; margin: 0 auto; padding: 24px; }}
    .toolbar {{ display: flex; align-items: center; gap: 12px; margin: 22px 0; }}
    select {{ padding: 8px 32px 8px 10px; border: 1px solid #aeb8c2; background: white; }}
    section {{ margin: 28px 0; }}
    .task-section {{ padding-top: 4px; border-top: 3px solid #2b7a78; }}
    h2 {{ margin-bottom: 4px; }}
    h3 {{ margin-top: 24px; font-size: 16px; }}
    .classes, .empty {{ color: #5d6d7e; }}
    .evidence {{ padding: 10px 12px; border-left: 4px solid #2b7a78; background: #edf6f5; }}
    .table-wrap {{ overflow-x: auto; }}
    table.data-table {{ width: 100%; border-collapse: collapse; background: white; font-size: 14px; }}
    .data-table th, .data-table td {{ padding: 9px 10px; border-bottom: 1px solid #dfe5e8; text-align: right; }}
    .data-table th:first-child, .data-table td:first-child {{ text-align: left; }}
    .data-table th {{ background: #e8eef2; color: #243746; }}
    details {{ margin: 12px 0; border: 1px solid #cbd5dc; background: white; }}
    summary {{ padding: 12px; cursor: pointer; font-weight: 600; }}
    details > h3, details > table, details > p {{ margin-left: 16px; margin-right: 16px; }}
    .metric-figure {{ display: block; width: min(100%, 1100px); height: auto; margin: 14px auto; }}
    footer {{ padding: 20px 24px 36px; text-align: center; color: #6b7780; font-size: 13px; }}
  </style>
</head>
<body>
  <header>
    <h1>Binocular rivalry results</h1>
    <p>Completed model runs grouped by decoding task. Generated {escape(generated_at)}.</p>
  </header>
  <main>
    <section>
      <h2>Run coverage</h2>
      <p class="classes">X marks the current final-holdout protocol; L marks legacy development-CV artifacts that should be rerun.</p>
      <div class="table-wrap">{_table(status)}</div>
    </section>
    <section>
      <h2>Evaluation protocol</h2>
      <p class="classes">Hyperparameters and training duration are selected using the first 80% of each subject's chronological trials. The model is then fitted on that development period and evaluated once on the untouched final 20%. Metrics are aggregated once per subject before group summaries.</p>
    </section>
    <div class="toolbar">
      <label for="task-selector"><strong>Task</strong></label>
      <select id="task-selector">{options}</select>
    </div>
    {sections}
    {temporal_onset}
    {inter_subject}
    <section>
      <h2>Model-wise ROC and precision-recall</h2>
      <p class="classes">Each panel compares all completed tasks for one model using subject-weighted mean curves.</p>
      {model_curves}
    </section>
  </main>
  <footer>Source: local artifacts directory. The report does not retrain models.</footer>
  <script>
    const selector = document.getElementById("task-selector");
    selector.addEventListener("change", () => {{
      document.querySelectorAll(".task-section").forEach(section => section.hidden = true);
      document.getElementById(`task-${{selector.value}}`).hidden = false;
    }});
  </script>
</body>
</html>
"""
    destination = output_path or artifacts_root / "report" / "index.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")
    return destination
