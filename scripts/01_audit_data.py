#!/usr/bin/env python
"""01 -- Data audit for both corpora.

Answers, with measured numbers, the questions you must answer *before*
modelling: how big is each set, how imbalanced, how duplicated, how much
encoding damage, and does any row carry contact PII?

Run:  python scripts/01_audit_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from idtweets import pii  # noqa: E402
from idtweets.io import TABLES_DIR, load_modi, load_novel  # noqa: E402


def audit_modi() -> pd.DataFrame:
    df = load_modi()
    before = df.attrs.get("dropped_missing_text", 0)
    from idtweets.preprocessing import fix_mojibake

    moji = int(df["text"].map(lambda t: fix_mojibake(t) != t).sum())
    pii_hits = pii.find_pii_hits(df["text"])
    rows = [
        ("rows_loaded", len(df) + before),
        ("rows_dropped_missing_text", before),
        ("rows_usable", len(df)),
        ("n_classes", df["category"].nunique()),
        ("class__-1_negative", int((df["category"] == -1).sum())),
        ("class__0_neutral", int((df["category"] == 0).sum())),
        ("class__1_positive", int((df["category"] == 1).sum())),
        ("majority_class_share", round(float(df["category"].value_counts(normalize=True).max()), 4)),
        ("duplicate_texts", int(df["text"].duplicated().sum())),
        ("rows_with_mojibake", moji),
        ("rows_with_contact_pii", len(pii_hits)),
        ("mean_chars", round(float(df["text"].astype(str).str.len().mean()), 2)),
        ("max_chars", int(df["text"].astype(str).str.len().max())),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def audit_novel() -> pd.DataFrame:
    df = load_novel()
    texts = df["text"].astype(str)
    mentions = texts.str.count(r"@\w+")
    rows = [
        ("rows", len(df)),
        ("unique_ids", int(df["id_str"].nunique())),
        ("duplicate_ids", int(df["id_str"].duplicated().sum())),
        ("duplicate_texts", int(texts.duplicated().sum())),
        ("duplicate_text_share", round(float(texts.duplicated().mean()), 4)),
        ("retweets", int(texts.str.match(r"^\s*RT\b").sum())),
        ("retweet_share", round(float(texts.str.match(r"^\s*RT\b").mean()), 4)),
        ("rows_with_url", int(texts.str.contains("http", regex=False).sum())),
        ("rows_with_hashtag", int(texts.str.contains(r"#\w+", regex=True).sum())),
        ("rows_with_>=1_mention", int((mentions >= 1).sum())),
        ("rows_with_>=2_mentions", int((mentions >= 2).sum())),
        ("mentions_total", int(mentions.sum())),
        ("rows_with_contact_pii", int(pii.flag_pii(texts).sum())),
        ("mean_chars", round(float(texts.str.len().mean()), 2)),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    modi = audit_modi()
    novel = audit_novel()
    modi.to_csv(TABLES_DIR / "audit_modi.csv", index=False)
    novel.to_csv(TABLES_DIR / "audit_novel_baswedan.csv", index=False)

    print("=" * 72)
    print("MODI / Indian-election sentiment dataset  (English)")
    print("=" * 72)
    print(modi.to_string(index=False))
    print()
    print("=" * 72)
    print("NOVEL BASWEDAN tweet corpus  (Indonesian)")
    print("=" * 72)
    print(novel.to_string(index=False))
    print()
    print(f"Wrote {TABLES_DIR / 'audit_modi.csv'}")
    print(f"Wrote {TABLES_DIR / 'audit_novel_baswedan.csv'}")


if __name__ == "__main__":
    main()
