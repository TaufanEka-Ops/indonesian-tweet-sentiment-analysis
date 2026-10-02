#!/usr/bin/env python
"""06 -- Error analysis: real misclassified examples.

Reads the held-out predictions written by ``scripts/05_train_modi_sentiment.py``
and surfaces the mistakes, grouped by the true label and by the confusion
direction, plus the terms most associated with each error pattern.

Run:  python scripts/06_error_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from idtweets import evaluation  # noqa: E402
from idtweets.io import CATEGORY_NAMES, FIGURES_DIR, TABLES_DIR  # noqa: E402

sns.set_theme(style="whitegrid")


def main() -> None:
    preds_path = TABLES_DIR / "modi_test_predictions.csv"
    if not preds_path.exists():
        raise SystemExit(
            f"{preds_path} not found -- run scripts/05_train_modi_sentiment.py first."
        )
    df = pd.read_csv(preds_path)
    df["text"] = df["text"].fillna("")
    print(f"Loaded {len(df)} held-out predictions")

    n_correct = int((df["true"] == df["pred"]).sum())
    print(f"Correct: {n_correct}  Errors: {len(df) - n_correct}  "
          f"Accuracy: {n_correct / len(df):.4f}")

    # --- confusion directions -------------------------------------------- #
    err = df[df["true"] != df["pred"]].copy()
    err["direction"] = (
        err["true"].map(CATEGORY_NAMES) + " -> " + err["pred"].map(CATEGORY_NAMES)
    )
    directions = (
        err["direction"].value_counts().rename_axis("direction").reset_index(name="count")
    )
    directions["share_of_errors"] = (directions["count"] / len(err)).round(4)
    directions.to_csv(TABLES_DIR / "modi_error_directions.csv", index=False)
    print("\n--- Error directions ---")
    print(directions.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=directions, y="direction", x="count", ax=ax, color="#C44E52")
    ax.set_title("Where the classifier goes wrong")
    ax.set_xlabel("misclassified tweets")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "modi_error_directions.png", dpi=150)
    plt.close(fig)

    # --- concrete examples ------------------------------------------------ #
    examples = evaluation.error_analysis(
        err["text"].reset_index(drop=True),
        err["true"].reset_index(drop=True),
        err["pred"].reset_index(drop=True),
        label_names=CATEGORY_NAMES,
        n_per_class=10,
    )
    examples.to_csv(TABLES_DIR / "modi_error_examples.csv", index=False)
    print("\n--- Misclassified examples (up to 10 per true class) ---")
    for label in sorted(examples["true"].unique()):
        sub = examples[examples["true"] == label]
        print(f"\n[{CATEGORY_NAMES[label]}] {len(sub)} shown")
        for _, row in sub.head(5).iterrows():
            txt = row["text"]
            txt = txt[:110] + "..." if len(txt) > 110 else txt
            print(f"   true={CATEGORY_NAMES[row['true']]:8s} pred={CATEGORY_NAMES[row['pred']]:8s} | {txt}")

    # --- what the errors look like, lexically ----------------------------- #
    from sklearn.feature_extraction.text import CountVectorizer

    if len(err) > 5:
        import numpy as np

        vec = CountVectorizer(min_df=3)
        X = vec.fit_transform(err["text"])
        arr = np.asarray(X.sum(axis=0)).ravel()
        terms = (
            pd.DataFrame({"term": vec.get_feature_names_out(), "count": arr.astype(int)})
            .sort_values("count", ascending=False)
            .head(20)
            .reset_index(drop=True)
        )
        terms.to_csv(TABLES_DIR / "modi_error_terms.csv", index=False)
        print("\n--- Most frequent terms among errors ---")
        print(terms.to_string(index=False))

    print(f"\nWrote error-analysis outputs to {TABLES_DIR} and {FIGURES_DIR}")


if __name__ == "__main__":
    main()
