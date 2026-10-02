#!/usr/bin/env python
"""05 -- Supervised sentiment classification on the Modi / Indian-election set.

This is the evaluation the original coursework was missing:
  * majority-class baseline for honesty,
  * repeated stratified 5x3 cross-validation,
  * macro-F1 as the headline metric (labels are imbalanced),
  * a stratified held-out split with a confusion matrix,
  * bootstrap CI on the held-out macro-F1.

Run:  python scripts/05_train_modi_sentiment.py
"""

from __future__ import annotations

import json
import sys
import time
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from idtweets import evaluation, modelling  # noqa: E402
from idtweets.cases import load_case_module  # noqa: E402
from idtweets.io import FIGURES_DIR, TABLES_DIR  # noqa: E402
from idtweets.modelling import RANDOM_STATE, build_pipeline, default_estimators  # noqa: E402

sns.set_theme(style="whitegrid")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    case = load_case_module("modi-election", "modi_election_sentiment.py")

    # ---------------------------------------------------------------- data -- #
    df, audit = case.prepare_dataset()
    print("=" * 78)
    print("DATA PREPARATION AUDIT (nothing dropped silently)")
    print("=" * 78)
    for k, v in audit.items():
        if k == "pii_hits":
            print(f"{k:28s}: {len(v)} row(s) -> {v}")
        else:
            print(f"{k:28s}: {v}")

    dist = case.label_distribution(df)
    dist.to_csv(TABLES_DIR / "modi_label_distribution.csv", index=False)
    print("\n--- Label distribution ---")
    print(dist.to_string(index=False))
    print("\n" + case.class_weight_note(df))

    # ------------------------------------------------------------ baseline -- #
    print("\n" + "=" * 78)
    print("REPEATED STRATIFIED CROSS-VALIDATION (5 folds x 3 repeats = 15 fits)")
    print("=" * 78)
    t0 = time.perf_counter()
    table, results = case.run_comparison(df, n_splits=5, n_repeats=3)
    elapsed = time.perf_counter() - t0
    table.to_csv(TABLES_DIR / "modi_model_comparison.csv", index=False)
    print(table.to_string(index=False))
    print(f"\nTotal CV wall time: {elapsed:.1f}s")

    gap = case.macro_f1_gap(table)
    print(f"\nBest macro-F1 beats majority baseline by {gap:+.4f}")

    # per-fold spread for the two closest models
    for r in results.values():
        print(f"  {r.name:22s} macro-F1 {r.macro_f1_mean:.4f} +/- {r.macro_f1_std:.4f}")

    # model comparison figure
    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = table.sort_values("macro_f1_mean")
    ax.barh(
        order["model"], order["macro_f1_mean"],
        xerr=order["macro_f1_std"], color="#4C72B0", capsize=3,
    )
    ax.set_xlabel("macro-F1 (repeated stratified CV, mean +/- sd)")
    ax.set_title("Model comparison -- macro-F1")
    ax.set_xlim(0, 1)
    for y, (v, n) in enumerate(zip(order["macro_f1_mean"], order["model"])):
        ax.text(v + 0.01, y, f"{v:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "modi_model_comparison.png", dpi=150)
    plt.close(fig)

    # class distribution figure
    fig, ax = plt.subplots(figsize=(5.5, 4))
    sns.barplot(data=dist, x="label_name", y="count", ax=ax, color="#C44E52")
    ax.set_title("Class distribution (Modi dataset)")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "modi_class_distribution.png", dpi=150)
    plt.close(fig)

    # -------------------------------------------------------------- holdout - #
    print("\n" + "=" * 78)
    print("STRATIFIED HELD-OUT TEST SET (80/20) -- best model")
    print("=" * 78)
    best_name = str(table.iloc[0]["model"])
    print(f"Best CV model: {best_name}")

    Xtr, Xte, ytr, yte = modelling.holdout_split(df["clean"], df["category"], test_size=0.2)
    pipe = build_pipeline(default_estimators()[best_name])
    pipe.fit(Xtr, ytr)
    preds = pipe.predict(Xte)

    metrics = evaluation.classification_metrics(yte, preds)
    print("\nHeld-out metrics:")
    for k, v in metrics.items():
        print(f"  {k:14s}: {v:.4f}")

    lo, hi = case.bootstrap_ci(
        yte,
        preds,
        partial(f1_score, average="macro"),
        n_boot=1000,
    )
    print(f"  macro-F1 95% bootstrap CI: [{lo:.4f}, {hi:.4f}]")

    report = evaluation.report_frame(yte, preds, label_names=case.LABEL_NAMES)
    report["macro_avg"] = None
    report.to_csv(TABLES_DIR / "modi_classification_report.csv", index=False)
    print("\nPer-class:")
    print(report.to_string(index=False))

    pd.DataFrame([metrics | {"model": best_name, "macro_f1_ci_low": lo,
                             "macro_f1_ci_high": hi}]).to_csv(
        TABLES_DIR / "modi_holdout_metrics.csv", index=False
    )

    cm = evaluation.confusion_frame(yte, preds, labels=[-1, 0, 1])
    cm.to_csv(TABLES_DIR / "modi_confusion_matrix.csv")
    print("\nConfusion matrix (rows = actual, cols = predicted):")
    print(cm.to_string())

    evaluation.plot_confusion(
        yte, preds, labels=[-1, 0, 1], label_names=case.LABEL_NAMES,
        title=f"Confusion matrix -- {best_name}",
        out_path=FIGURES_DIR / "modi_confusion_matrix.png",
    )
    evaluation.plot_confusion(
        yte, preds, labels=[-1, 0, 1], label_names=case.LABEL_NAMES,
        title=f"Confusion matrix (row-normalised) -- {best_name}",
        out_path=FIGURES_DIR / "modi_confusion_matrix_normalized.png",
        normalize=True,
    )

    # per-class F1 figure
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=report, x="label_name", y="f1", ax=ax, color="#55A868")
    ax.set_ylim(0, 1)
    ax.set_title("Per-class F1 (held-out)")
    ax.set_xlabel("")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.3f", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "modi_per_class_f1.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------- interpretability #
    print("\n" + "=" * 78)
    print("TOP FEATURES PER CLASS (Logistic Regression)")
    print("=" * 78)
    lr_pipe = build_pipeline(default_estimators()["logistic_regression"])
    lr_pipe.fit(Xtr, ytr)
    top = evaluation.top_features_per_class(lr_pipe, top_n=15)
    top["label_name"] = top["class"].map(case.LABEL_NAMES)
    top.to_csv(TABLES_DIR / "modi_top_features.csv", index=False)
    for c in sorted(top["class"].unique()):
        terms = top[top["class"] == c]["term"].tolist()
        print(f"  {case.LABEL_NAMES[c]:9s}: {', '.join(terms[:12])}")

    # save held-out predictions for the error-analysis script
    pd.DataFrame(
        {
            "text": Xte.to_numpy(),
            "true": yte.to_numpy(),
            "pred": preds,
        }
    ).to_csv(TABLES_DIR / "modi_test_predictions.csv", index=False)

    (TABLES_DIR / "modi_data_audit.json").write_text(json.dumps(audit, indent=2))
    print(f"\nWrote tables to {TABLES_DIR}")
    print(f"Wrote figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
