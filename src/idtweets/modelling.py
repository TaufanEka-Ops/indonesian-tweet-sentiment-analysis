"""Modelling: estimators, honest baselines and stratified cross-validation.

The original coursework fit a single ``MultinomialNB`` on one 80/20 split and
reported accuracy. There was no baseline, no repeated resampling and no
macro-F1 -- all three matter here because the label distribution is imbalanced
(2245 / 1910 / 1170, i.e. 42% / 36% / 22%). Accuracy alone lets a model look
good by over-predicting the majority class.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .features import build_tfidf

RANDOM_STATE = 42

SCORING = {
    "accuracy": "accuracy",
    "macro_f1": "f1_macro",
    "weighted_f1": "f1_weighted",
}


def build_pipeline(estimator, *, vectorizer=None) -> Pipeline:
    """Wrap a vectoriser + estimator into a single sklearn Pipeline.

    Everything is inside the pipeline, so vectorisation is refit on every
    cross-validation fold -- there is no leakage of the held-out vocabulary.
    """
    vec = build_tfidf() if vectorizer is None else vectorizer
    return Pipeline([("tfidf", vec), ("clf", estimator)])


def default_estimators() -> dict[str, object]:
    """The model comparison set: baseline + three real classifiers."""
    return {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "naive_bayes": MultinomialNB(alpha=0.3),
        "logistic_regression": LogisticRegression(
            C=4.0, max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "linear_svc": LinearSVC(
            C=0.5, class_weight="balanced", random_state=RANDOM_STATE, dual="auto"
        ),
    }


@dataclass
class CVResult:
    name: str
    accuracy_mean: float
    accuracy_std: float
    macro_f1_mean: float
    macro_f1_std: float
    weighted_f1_mean: float
    weighted_f1_std: float
    fit_seconds: float
    folds: list[dict] = field(default_factory=list)

    def as_row(self) -> dict:
        return {
            "model": self.name,
            "accuracy_mean": round(self.accuracy_mean, 4),
            "accuracy_std": round(self.accuracy_std, 4),
            "macro_f1_mean": round(self.macro_f1_mean, 4),
            "macro_f1_std": round(self.macro_f1_std, 4),
            "weighted_f1_mean": round(self.weighted_f1_mean, 4),
            "weighted_f1_std": round(self.weighted_f1_std, 4),
            "fit_seconds": round(self.fit_seconds, 3),
        }


def run_repeated_cv(
    X: pd.Series,
    y: pd.Series,
    pipeline: Pipeline,
    *,
    name: str = "model",
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = RANDOM_STATE,
    n_jobs: int | None = None,
) -> CVResult:
    """Stratified repeated CV, reporting macro-F1 (the headline metric).

    Repeats matter at this dataset size: a single 5-fold split has a wide
    variance band and would hide the difference between close models.
    """
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    res = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=SCORING,
        n_jobs=n_jobs,
        return_train_score=False,
    )
    folds = [
        {
            "fold": i,
            "accuracy": float(res["test_accuracy"][i]),
            "macro_f1": float(res["test_macro_f1"][i]),
            "weighted_f1": float(res["test_weighted_f1"][i]),
        }
        for i in range(len(res["test_accuracy"]))
    ]
    return CVResult(
        name=name,
        accuracy_mean=float(np.mean(res["test_accuracy"])),
        accuracy_std=float(np.std(res["test_accuracy"])),
        macro_f1_mean=float(np.mean(res["test_macro_f1"])),
        macro_f1_std=float(np.std(res["test_macro_f1"])),
        weighted_f1_mean=float(np.mean(res["test_weighted_f1"])),
        weighted_f1_std=float(np.std(res["test_weighted_f1"])),
        fit_seconds=float(np.sum(res["fit_time"])),
        folds=folds,
    )


def compare_models(
    X: pd.Series,
    y: pd.Series,
    *,
    estimators: dict[str, object] | None = None,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[str, CVResult]]:
    """Run repeated stratified CV for every estimator and return a table."""
    estimators = default_estimators() if estimators is None else estimators
    results: dict[str, CVResult] = {}
    for name, est in estimators.items():
        results[name] = run_repeated_cv(
            X,
            y,
            build_pipeline(est),
            name=name,
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state,
        )
    table = pd.DataFrame([r.as_row() for r in results.values()]).sort_values(
        "macro_f1_mean", ascending=False
    )
    table = table.reset_index(drop=True)
    return table, results


def holdout_split(
    X: pd.Series,
    y: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
):
    """One stratified held-out split, used for confusion matrix + error analysis."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def oof_predictions(
    X: pd.Series,
    y: pd.Series,
    pipeline: Pipeline,
    *,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """Out-of-fold predictions (every row predicted by a model that never saw it)."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return cross_val_predict(pipeline, X, y, cv=cv, n_jobs=None)
