#!/usr/bin/env python
"""02 -- Preprocess the Novel Baswedan corpus and quantify the mention bug.

Produces:
  results/tables/novel_preprocessed.csv   cleaned corpus (published artifact)
  results/tables/novel_defect_impact.csv  the first-match-only bug, measured
  results/tables/novel_examples_bug.csv   concrete before/after rows

Run:  python scripts/02_preprocess_novel_baswedan.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from idtweets.cases import load_case_module  # noqa: E402
from idtweets.io import TABLES_DIR  # noqa: E402
from idtweets.preprocessing import (  # noqa: E402
    MENTION_RE,
    Preprocessor,
    remove_pattern_original,
    strip_mentions,
)


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    case = load_case_module("novel-baswedan", "novel_baswedan_analysis.py")

    raw = case.load_raw()
    print(f"Loaded {len(raw)} raw tweets")

    impact = case.defect_impact(raw)
    impact_df = pd.DataFrame(
        {"metric": list(impact.keys()), "value": list(impact.values())}
    )
    impact_df.to_csv(TABLES_DIR / "novel_defect_impact.csv", index=False)

    print("\n--- Defect #1: first-match-only mention stripping ---")
    print(impact_df.to_string(index=False))

    # Concrete before/after evidence: rows where the original left a mention.
    texts = raw["text"].astype(str)
    multi = texts[texts.str.count(r"@\w+") >= 2].head(5)
    rows = []
    for t in multi:
        rows.append(
            {
                "original_text": t,
                "original_logic_output": remove_pattern_original(t, r"@\w+"),
                "fixed_output": strip_mentions(t),
                "mentions_left_by_original": len(
                    MENTION_RE.findall(remove_pattern_original(t, r"@\w+"))
                ),
                "mentions_left_by_fixed": len(MENTION_RE.findall(strip_mentions(t))),
            }
        )
    examples = pd.DataFrame(rows)
    examples.to_csv(TABLES_DIR / "novel_examples_bug.csv", index=False)
    print("\n--- Concrete before/after (first 5 multi-mention rows) ---")
    with pd.option_context("display.max_colwidth", 70):
        print(examples[["original_logic_output", "fixed_output",
                        "mentions_left_by_original", "mentions_left_by_fixed"]].to_string(index=False))

    # Full preprocessing.
    processed = case.preprocess(raw)
    deduped = case.deduplicate(processed)
    out = deduped[["id_str", "text", "clean", "clean_token_count"]]
    out.to_csv(TABLES_DIR / "novel_preprocessed.csv", index=False)

    print("\n--- Preprocessing output ---")
    print(f"raw rows                  : {len(raw)}")
    print(f"rows after full pipeline  : {len(processed)}")
    print(f"rows after de-duplication : {len(deduped)}")
    print(f"mean tokens per tweet     : {processed['clean_token_count'].mean():.2f}")
    print(f"empty cleaned rows        : {int((processed['clean'].str.len() == 0).sum())}")

    # Re-running the same text must be stable.
    pre = Preprocessor()
    sample = out["text"].head(50)
    again = pre.transform_series(sample)
    stable = bool((again.values == pre.transform_series(sample).values).all())
    print(f"pipeline is deterministic : {stable}")

    print(f"\nWrote {TABLES_DIR / 'novel_preprocessed.csv'}")
    print(f"Wrote {TABLES_DIR / 'novel_defect_impact.csv'}")
    print(f"Wrote {TABLES_DIR / 'novel_examples_bug.csv'}")


if __name__ == "__main__":
    main()
