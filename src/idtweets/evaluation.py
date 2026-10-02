"""Evaluation helpers: metrics, confusion matrices, error analysis, coefficients."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)


def classification_metrics(y_true, y_pred, labels=None) -> dict:
    """Headline metrics, with macro-F1 as the primary number."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def report_frame(y_true, y_pred, label_names: dict | None = None) -> pd.DataFrame:
    """Per-class precision/recall/F1 as a tidy DataFrame."""
    labels = sorted(set(y_true) | set(y_pred))
    rep = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for label in labels:
        key = str(label)
        if key not in rep:
            continue
        name = label_names.get(label, key) if label_names else key
        rows.append(
            {
                "label": label,
                "label_name": name,
                "precision": round(rep[key]["precision"], 4),
                "recall": round(rep[key]["recall"], 4),
                "f1": round(rep[key]["f1-score"], 4),
                "support": int(rep[key]["support"]),
            }
        )
    return pd.DataFrame(rows)


def confusion_frame(y_true, y_pred, labels=None) -> pd.DataFrame:
    labels = sorted(set(y_true) | set(y_pred)) if labels is None else list(labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=[f"true_{l}" for l in labels],
                        columns=[f"pred_{l}" for l in labels])


def plot_confusion(y_true, y_pred, *, labels=None, label_names=None,
                   title="Confusion matrix", out_path: str | Path | None = None,
                   normalize: bool = False):
    """Render a confusion-matrix heat map (imports matplotlib lazily)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    labels = sorted(set(y_true) | set(y_pred)) if labels is None else list(labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fmt = "d"
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    tick = [label_names.get(l, str(l)) if label_names else str(l) for l in labels]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap="Blues", cbar=False,
        xticklabels=tick, yticklabels=tick, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
    return fig


def error_analysis(
    texts: pd.Series,
    y_true: pd.Series,
    y_pred,
    *,
    proba=None,
    label_names: dict | None = None,
    n_per_class: int = 8,
) -> pd.DataFrame:
    """Return real misclassified examples, grouped by their true label.

    ``confusion`` records the (true -> predicted) direction so a reader can see
    *which* boundary the model gets wrong, not merely that it is wrong.
    """
    df = pd.DataFrame(
        {
            "text": texts.astype("string").to_numpy(),
            "true": np.asarray(y_true),
            "pred": np.asarray(y_pred),
        }
    )
    if proba is not None:
        df["confidence"] = np.max(np.asarray(proba), axis=1)
    df["correct"] = df["true"] == df["pred"]
    err = df[~df["correct"]].copy()
    if label_names:
        err["true_name"] = err["true"].map(label_names)
        err["pred_name"] = err["pred"].map(label_names)
    err["confusion"] = err["true"].astype(str) + "->" + err["pred"].astype(str)
    if proba is not None:
        err = err.sort_values("confidence", ascending=False)
    sampled = (
        err.groupby("true", group_keys=False)
        .head(n_per_class)
        .reset_index(drop=True)
    )
    return sampled


def top_features_per_class(pipeline, *, top_n: int = 15) -> pd.DataFrame:
    """Top-weighted n-grams per class from a fitted linear model.

    Works for LogisticRegression (``coef_``) and LinearSVC (``coef_``).
    Returns a tidy frame ``class, rank, term, weight``.
    """
    vec = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        raise ValueError(
            f"{type(clf).__name__} has no coef_; use a linear model "
            "(LogisticRegression / LinearSVC) or inspect feature_log_prob_ "
            "for Naive Bayes."
        )
    names = np.asarray(vec.get_feature_names_out())
    coef = np.asarray(clf.coef_)
    classes = list(clf.classes_)
    rows = []
    if coef.shape[0] == 1:  # binary
        order = np.argsort(coef[0])
        picks = {
            classes[0]: order[:top_n],
            classes[1]: order[::-1][:top_n],
        }
    else:
        picks = {
            cls: np.argsort(coef[i])[::-1][:top_n] for i, cls in enumerate(classes)
        }
    for cls, idx in picks.items():
        for rank, j in enumerate(idx, start=1):
            rows.append(
                {"class": cls, "rank": rank, "term": names[j],
                 "weight": float(coef[classes.index(cls)][j])
                 if coef.shape[0] > 1 else float(coef[0][j])}
            )
    return pd.DataFrame(rows)


def top_features_naive_bayes(pipeline, *, top_n: int = 15) -> pd.DataFrame:
    """Top log-probability tokens per class for a fitted MultinomialNB."""
    vec = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    names = np.asarray(vec.get_feature_names_out())
    rows = []
    for i, cls in enumerate(clf.classes_):
        order = np.argsort(clf.feature_log_prob_[i])[::-1][:top_n]
        for rank, j in enumerate(order, start=1):
            rows.append(
                {
                    "class": cls,
                    "rank": rank,
                    "term": names[j],
                    "log_prob": float(clf.feature_log_prob_[i][j]),
                }
            )
    return pd.DataFrame(rows)
