#!/usr/bin/env python
"""03 -- Exploratory analysis of the Novel Baswedan corpus.

Produces tables + figures under results/. Descriptive only: this corpus has no
sentiment label, so no classifier is trained here.

Run:  python scripts/03_explore_novel_baswedan.py
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

from idtweets.cases import load_case_module  # noqa: E402
from idtweets.io import FIGURES_DIR, TABLES_DIR  # noqa: E402

sns.set_theme(style="whitegrid")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    case = load_case_module("novel-baswedan", "novel_baswedan_analysis.py")

    raw = case.load_raw()
    processed = case.preprocess(raw)
    deduped = case.deduplicate(processed)

    # --- hashtags ---------------------------------------------------------- #
    tags = case.hashtag_table(raw, top_n=15)
    tags.to_csv(TABLES_DIR / "novel_hashtags.csv", index=False)
    print("--- Top hashtags ---")
    print(tags.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=tags, y="hashtag", x="count", ax=ax, color="#4C72B0")
    ax.set_title("Top hashtags in the Novel Baswedan tweet corpus")
    ax.set_xlabel("tweets containing hashtag")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "novel_hashtags.png", dpi=150)
    plt.close(fig)

    # --- top terms --------------------------------------------------------- #
    terms = case.top_terms(deduped, top_n=25, min_df=3)
    terms.to_csv(TABLES_DIR / "novel_top_terms.csv", index=False)
    print("\n--- Top terms (de-duplicated, cleaned) ---")
    print(terms.head(25).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=terms, y="term", x="count", ax=ax, color="#55A868")
    ax.set_title("Most frequent terms (de-duplicated corpus)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "novel_top_terms.png", dpi=150)
    plt.close(fig)

    # --- name clusters ----------------------------------------------------- #
    entities = case.entity_table(deduped)
    entities.to_csv(TABLES_DIR / "novel_entities.csv", index=False)
    print("\n--- Name / entity clusters ---")
    print(entities.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=entities, y="entity", x="matching_tweets", ax=ax, color="#C44E52")
    ax.set_title("Entity mentions per cluster (de-duplicated corpus)")
    ax.set_xlabel("tweets")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "novel_entities.png", dpi=150)
    plt.close(fig)

    # --- length distribution ---------------------------------------------- #
    stats = case.length_stats(deduped)
    stats.to_csv(TABLES_DIR / "novel_length_stats.csv", index=False)
    print("\n--- Length statistics ---")
    print(stats.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(processed["clean_token_count"], bins=30, color="#8172B3")
    ax.set_title("Cleaned token count per tweet")
    ax.set_xlabel("tokens after preprocessing")
    ax.set_ylabel("tweets")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "novel_token_hist.png", dpi=150)
    plt.close(fig)

    # --- dedup effect ------------------------------------------------------ #
    dedup = pd.DataFrame(
        {
            "stage": ["raw", "raw exact-duplicate texts", "after de-duplication"],
            "rows": [
                len(raw),
                int(raw["text"].astype(str).duplicated().sum()),
                len(deduped),
            ],
        }
    )
    dedup.to_csv(TABLES_DIR / "novel_dedup_effect.csv", index=False)
    print("\n--- De-duplication effect ---")
    print(dedup.to_string(index=False))

    print(f"\nFigures written to {FIGURES_DIR}")
    print(f"Tables written to {TABLES_DIR}")


if __name__ == "__main__":
    main()
