from __future__ import annotations

import inspect
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

from ..dataset import EpochDataset
from ..explanations import (
    ExplanationResult,
    explain_tabular_model,
    stratified_background,
)
from ..metrics import classification_metrics
from ..models import ModelFactory
from ..splits import SplitPlan, development_splits

FeatureTransform = Callable[[np.ndarray, float], np.ndarray]


@dataclass
class ClassicalRunResult:
    metrics: pd.DataFrame
    y_true: np.ndarray
    probabilities: np.ndarray
    subjects: np.ndarray
    trial_indices: np.ndarray
    folds: np.ndarray
    explanation: ExplanationResult | None = None
    skipped_subjects: tuple[str, ...] = ()


@dataclass
class _SubjectResult:
    rows: list[dict[str, float | int | str]]
    y: list[np.ndarray]
    probabilities: list[np.ndarray]
    subjects: list[np.ndarray]
    trial_indices: list[np.ndarray]
    folds: list[np.ndarray]
    explanation_values: list[np.ndarray]
    explanation_samples: list[np.ndarray]


def _identity_transform(X: np.ndarray, sfreq: float) -> np.ndarray:
    del sfreq
    return X


def _run_subject(
    *,
    dataset: EpochDataset,
    subject: str,
    model_factory: ModelFactory,
    feature_transform: FeatureTransform,
    split_plan: SplitPlan,
    explain: bool,
    background_per_class: int,
    explanation_samples_per_class: int,
    subject_completed: Callable[[str, float], None],
) -> tuple[_SubjectResult, str | None]:
    n_classes = len(dataset.class_ids)
    subject_mask = dataset.subjects == subject
    X_subject = dataset.X[subject_mask]
    y_subject = dataset.y[subject_mask]
    trials_subject = dataset.trial_indices[subject_mask]
    result = _SubjectResult([], [], [], [], [], [], [], [])
    selection_splits, holdout_indices = development_splits(y_subject, split_plan)
    development_count = len(y_subject) - len(holdout_indices)
    development_indices = np.arange(development_count)
    expected_classes = np.arange(n_classes)
    if (
        not len(holdout_indices)
        or not np.array_equal(np.unique(y_subject[development_indices]), expected_classes)
        or not np.array_equal(np.unique(y_subject[holdout_indices]), expected_classes)
    ):
        subject_completed(subject, float("nan"))
        return result, f"{subject}: final chronological holdout does not contain every class"

    X_development = feature_transform(X_subject[development_indices], dataset.sfreq)
    X_holdout = feature_transform(X_subject[holdout_indices], dataset.sfreq)
    if "selection_splits" in inspect.signature(model_factory.make).parameters:
        model = model_factory.make(selection_splits=selection_splits)
    else:
        model = model_factory.make()
    model.fit(X_development, y_subject[development_indices])
    probabilities = model.predict_proba(X_holdout)
    model_classes = np.asarray(model.classes_, dtype=int)
    aligned_probabilities = np.zeros((len(holdout_indices), n_classes), dtype=float)
    aligned_probabilities[:, model_classes] = probabilities
    if explain:
        if X_development.ndim == 3 and hasattr(model, "steps"):
            explanation_model = model[-1]
            explanation_train = model[:-1].transform(X_development)
            explanation_test = model[:-1].transform(X_holdout)
        elif X_development.ndim == 2:
            explanation_model = model
            explanation_train = X_development
            explanation_test = X_holdout
        else:
            raise ValueError(
                "SHAP requires a 2D representation or a transformable sklearn pipeline."
            )
        background, _ = stratified_background(
            explanation_train,
            y_subject[development_indices],
            per_class=background_per_class,
            random_seed=split_plan.random_seed,
        )
        samples, _ = stratified_background(
            explanation_test,
            y_subject[holdout_indices],
            per_class=explanation_samples_per_class,
            random_seed=split_plan.random_seed + 1,
        )
        holdout_explanation = explain_tabular_model(
            explanation_model,
            background=background,
            samples=samples,
            class_ids=dataset.class_ids,
            algorithm="permutation",
            max_evaluations=2 * explanation_train.shape[1] + 1,
        )
        result.explanation_values.append(holdout_explanation.values)
        result.explanation_samples.append(holdout_explanation.samples)

    holdout_metrics = classification_metrics(
        y_subject[holdout_indices],
        aligned_probabilities,
        n_classes=n_classes,
    )
    result.rows.append(
        {
            "subject": subject,
            "evaluation": "final_chronological_holdout",
            **holdout_metrics.to_dict(),
        }
    )
    result.y.append(y_subject[holdout_indices])
    result.probabilities.append(aligned_probabilities)
    result.subjects.append(np.full(len(holdout_indices), subject, dtype=object))
    result.trial_indices.append(trials_subject[holdout_indices])
    result.folds.append(np.ones(len(holdout_indices), dtype=np.int16))
    subject_completed(subject, holdout_metrics.balanced_accuracy)
    return result, None


def run_subjectwise_classical(
    dataset: EpochDataset,
    model_factory: ModelFactory,
    *,
    feature_transform: FeatureTransform = _identity_transform,
    split_plan: SplitPlan = SplitPlan(),
    explain: bool = False,
    background_per_class: int = 10,
    explanation_samples_per_class: int = 10,
    progress_label: str | None = None,
    n_jobs: int = 1,
) -> ClassicalRunResult:
    if n_jobs < 1:
        raise ValueError("n_jobs must be at least one.")
    rows: list[dict[str, float | int | str]] = []
    y_blocks: list[np.ndarray] = []
    probability_blocks: list[np.ndarray] = []
    subject_blocks: list[np.ndarray] = []
    trial_blocks: list[np.ndarray] = []
    fold_blocks: list[np.ndarray] = []
    explanation_values: list[np.ndarray] = []
    explanation_samples: list[np.ndarray] = []

    subjects = dataset.subject_ids
    label = progress_label or f"{dataset.task_id}/{model_factory.name}"

    progress = tqdm(
        total=len(subjects),
        desc=label,
        unit="subject",
        dynamic_ncols=True,
        disable=None,
    )
    progress_lock = Lock()

    def subject_completed(
        subject: str,
        balanced_accuracy: float,
    ) -> None:
        with progress_lock:
            progress.set_postfix_str(
                f"subject={subject}, "
                f"balanced_accuracy={balanced_accuracy:.3f}",
                refresh=False,
            )
            progress.update()

    def run(work):
        subject = work
        return _run_subject(
            dataset=dataset,
            subject=subject,
            model_factory=model_factory,
            feature_transform=feature_transform,
            split_plan=split_plan,
            explain=explain,
            background_per_class=background_per_class,
            explanation_samples_per_class=explanation_samples_per_class,
            subject_completed=subject_completed,
        )

    try:
        if n_jobs == 1:
            subject_results = [run(subject) for subject in subjects]
        else:
            workers = min(n_jobs, len(subjects))
            with threadpool_limits(limits=1):
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    subject_results = list(executor.map(run, subjects))
    finally:
        progress.close()

    skipped_subjects: list[str] = []
    for subject_result, skip_reason in subject_results:
        if skip_reason is not None:
            skipped_subjects.append(skip_reason)
            continue
        rows.extend(subject_result.rows)
        y_blocks.extend(subject_result.y)
        probability_blocks.extend(subject_result.probabilities)
        subject_blocks.extend(subject_result.subjects)
        trial_blocks.extend(subject_result.trial_indices)
        fold_blocks.extend(subject_result.folds)
        explanation_values.extend(subject_result.explanation_values)
        explanation_samples.extend(subject_result.explanation_samples)

    if not y_blocks:
        raise RuntimeError("No valid subject-level folds were produced.")
    explanation = None
    if explanation_values:
        explanation = ExplanationResult(
            values=np.concatenate(explanation_values),
            base_values=np.array([]),
            samples=np.concatenate(explanation_samples),
            class_ids=dataset.class_ids,
            method="shap_permutation",
        )
    return ClassicalRunResult(
        metrics=pd.DataFrame(rows),
        y_true=np.concatenate(y_blocks),
        probabilities=np.concatenate(probability_blocks),
        subjects=np.concatenate(subject_blocks),
        trial_indices=np.concatenate(trial_blocks),
        folds=np.concatenate(fold_blocks),
        explanation=explanation,
        skipped_subjects=tuple(skipped_subjects),
    )
