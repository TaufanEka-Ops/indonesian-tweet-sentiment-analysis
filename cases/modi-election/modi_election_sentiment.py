"""Case study 2 -- Modi / Indian-election sentiment classification.

This is the supervised half of the repository. Unlike the original notebook it
reports macro-F1 (not just accuracy), compares against a majority-class
baseline, uses repeated stratified cross-validation, produces a confusion
matrix and a real error analysis.

Label distribution (measured): ``1`` positive 2245, ``0`` neutral 1910,
``-1`` negative 1170 -> imbalanced, so macro-F1 is the honest headline metric
and plain accuracy is reported only alongside it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from idtweets import modelling, pii
from idtweets.features import build_tfidf
from idtweets.io import CATEGORY_NAMES, load_modi
from idtweets.modelling import RANDOM_STATE, build_pipeline, default_estimators
from idtweets.preprocessing import Preprocessor, make_english_preprocessor

# Re-export for scripts/notebooks.
LABEL_NAMES = CATEGORY_NAMES


def prepare_dataset(df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Full data-prep path: encoding repair -> PII drop -> preprocessing.

    Returns the cleaned frame plus an audit dict of every row count removed and
    why. No row is ever dropped silently.
    """
    if df is None:
        df = load_modi()
    else:
        df = df.copy()
    audit: dict[str, object] = {"rows_loaded": int(len(df))}
    audit["dropped_missing_text"] = int(df.attrs.get("dropped_missing_text", 0))

    # 1. Encoding repair on the raw text (mojibake from CP1252 round-trip).
    from idtweets.preprocessing import fix_mojibake

    moji_mask = df["text"].map(lambda t: fix_mojibake(t) != t)
    audit["rows_with_mojibake"] = int(moji_mask.sum())
    df["text"] = df["text"].map(fix_mojibake)

    # 2. Privacy: drop rows carrying a personal phone number or e-mail.
    pii_mask = pii.flag_pii(df["text"])
    audit["pii_rows_dropped"] = int(pii_mask.sum())
    audit["pii_hits"] = pii.find_pii_hits(df["text"]).to_dict(orient="records")
    df = df[~pii_mask].copy()

    # 3. Text preprocessing (English preset -- the corpus is English).
    pre: Preprocessor = make_english_preprocessor()
    df["clean"] = pre.transform_series(df["text"])
    empty_mask = df["clean"].str.len() == 0
    audit["rows_empty_after_clean"] = int(empty_mask.sum())
    df = df[~empty_mask].copy()

    # 4. Exact duplicate cleaned texts.
    dupe_mask = df["clean"].duplicated(keep="first")
    audit["duplicate_rows_dropped"] = int(dupe_mask.sum())
    df = df[~dupe_mask].copy()

    df = df.reset_index(drop=True)
    audit["rows_final"] = int(len(df))
    audit["class_counts"] = {
        CATEGORY_NAMES[int(k)]: int(v)
        for k, v in df["category"].value_counts().sort_index().items()
    }
    return df, audit


def model_pipelines() -> dict[str, Pipeline]:
    """The comparison set, each wrapped in its own TF-IDF pipeline."""
    return {name: build_pipeline(est) for name, est in default_estimators().items()}


def run_comparison(
    df: pd.DataFrame,
    *,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, dict]:
    """Repeated stratified CV for every estimator."""
    return modelling.compare_models(
        df["clean"],
        df["category"],
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )


def best_linear_pipeline(df: pd.DataFrame, *, which: str = "logistic_regression"):
    return build_pipeline(default_estimators()[which], vectorizer=build_tfidf())


def label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["category"].value_counts().sort_index()
    out = pd.DataFrame(
        {
            "category": counts.index,
            "label_name": [CATEGORY_NAMES[int(c)] for c in counts.index],
            "count": counts.values,
        }
    )
    out["share"] = (out["count"] / out["count"].sum()).round(4)
    return out.reset_index(drop=True)


def class_weight_note(df: pd.DataFrame) -> str:
    vc = df["category"].value_counts(normalize=True)
    return (
        f"majority class {CATEGORY_NAMES[int(vc.idxmax())]} = {vc.max():.1%} "
        f"of rows -> a model that always predicts it scores {vc.max():.1%} accuracy "
        f"but only macro-F1 {(1 / 3) * (2 * vc.max() / (1 + vc.max())):.3f} "
        "(one class F1, two classes 0)."
    )


def macro_f1_gap(table: pd.DataFrame, baseline: str = "majority_baseline") -> float:
    """How much macro-F1 the best model beats the majority baseline by."""
    best = table["macro_f1_mean"].max()
    base = float(table.loc[table["model"] == baseline, "macro_f1_mean"].iloc[0])
    return float(best - base)


def bootstrap_ci(
    y_true, y_pred, metric, *, n_boot: int = 1000, alpha: float = 0.05, seed: int = RANDOM_STATE
) -> tuple[float, float]:
    """Percentile bootstrap CI for a metric (honesty about sample noise)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[i] = metric(y_true[idx], y_pred[idx])
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
