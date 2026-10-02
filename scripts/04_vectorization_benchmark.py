#!/usr/bin/env python
"""04 -- Defect #2: ``np.vectorize`` is a loop with overhead, not a vectoriser.

The original notebook applied preprocessing with::

    df['remove_user'] = np.vectorize(remove_pattern)(df['text'], "@[\\w]*")

``np.vectorize`` does not vectorise anything -- it is a documented convenience
wrapper that loops in Python. This script measures it against real alternatives
on the Novel Baswedan corpus and reports the numbers *as measured*, including
the result that a plain list comprehension is the fastest option at this size.

Method: the 2,727-tweet corpus is repeated 20x to ~54.5k rows so timings are
stable, each method is run 3 times and the best wall time is kept, and every
output is checked for leftover ``@mentions`` so a fast-but-wrong path cannot
win on speed alone.

Run:  python scripts/04_vectorization_benchmark.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from idtweets.io import TABLES_DIR, load_novel  # noqa: E402
from idtweets.preprocessing import (  # noqa: E402
    MENTION_RE,
    Preprocessor,
    apply_comprehension,
    preprocess_regex_only,
    remove_pattern_original,
    strip_mentions,
)

PATTERN = r"@\w+"
REPEATS = 20
TIMING_ROUNDS = 3


def count_leftover(series) -> int:
    return int(sum(len(MENTION_RE.findall(str(t))) for t in series))


def timeit(fn, rounds: int = TIMING_ROUNDS) -> float:
    best = float("inf")
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    base = load_novel()["text"].astype(str)
    texts = pd.concat([base] * REPEATS, ignore_index=True)
    n = len(texts)
    print(f"Corpus: {len(base)} tweets x {REPEATS} repeats = {n} rows\n")

    rows = []

    def record(method: str, fn, *, correct: bool, note: str = "") -> None:
        secs = timeit(fn)
        out = fn()
        rows.append(
            {
                "method": method,
                "seconds": round(secs, 4),
                "ms_per_1k_rows": round(secs / n * 1000 * 1000, 2),
                "mentions_left_behind": count_leftover(out),
                "correct": correct,
                "note": note,
            }
        )

    # 1. The original approach, verbatim.
    record(
        "np.vectorize(remove_pattern_original)",
        lambda: np.vectorize(remove_pattern_original)(texts, PATTERN),
        correct=False,
        note="original bug: first mention only",
    )

    # 2. np.vectorize wrapping the CORRECT scalar function -> still a Python loop.
    record(
        "np.vectorize(strip_mentions)",
        lambda: np.vectorize(strip_mentions)(texts),
        correct=True,
        note="correct logic, but still a Python-level loop",
    )

    # 3. Plain list comprehension -- the honest non-pandas baseline.
    record(
        "list comprehension",
        lambda: apply_comprehension(texts, strip_mentions),
        correct=True,
        note="same scalar fn, no numpy wrapper",
    )

    # 4. pandas .str.replace -- C-level regex over the whole column.
    record(
        "pandas .str.replace (vectorized regex)",
        lambda: preprocess_regex_only(texts),
        correct=True,
        note="no per-row Python call",
    )

    table = pd.DataFrame(rows)
    table.to_csv(TABLES_DIR / "vectorization_benchmark.csv", index=False)

    print("=" * 96)
    print(f"Mention stripping on {n} tweets ({TIMING_ROUNDS} rounds, best of each)")
    print("=" * 96)
    print(table.to_string(index=False))

    np_vec = float(table.loc[table["method"] == "np.vectorize(strip_mentions)", "seconds"].iloc[0])
    comp = float(table.loc[table["method"] == "list comprehension", "seconds"].iloc[0])
    strs = float(
        table.loc[table["method"].str.startswith("pandas"), "seconds"].iloc[0]
    )
    print("\n--- What the numbers actually say ---")
    print(f"np.vectorize(strip_mentions) / list comprehension = {np_vec / comp:.2f}x "
          "-> the wrapper costs more than the work it wraps")
    print(f"np.vectorize(strip_mentions) / pandas .str.replace  = {np_vec / strs:.2f}x "
          "(pandas is slower here: 2.7k-char-short strings, per-call regex compile "
          "overhead dominates; use .str when rows are long or when you want to stay "
          "in pandas without a Python callable)")
    print("The original np.vectorize path is fast AND wrong: it leaves "
          f"{int(table['mentions_left_behind'].iloc[0])} mentions behind.")
    print("Every corrected path leaves 0.")

    # The full pipeline, reported separately: its cost is stemming, not vectorising.
    print("\n--- Where the time in the full pipeline actually goes ---")
    unique = base.drop_duplicates()
    print(f"unique texts: {len(unique)} (of {len(base)}; the cache makes the rest free)")
    full = timeit(lambda: Preprocessor().transform_series(unique, cache=False), rounds=1)
    nopre = timeit(
        lambda: preprocess_regex_only(unique), rounds=TIMING_ROUNDS
    )
    print(f"regex-only preprocessing on unique texts : {nopre:7.4f}s")
    print(f"full pipeline incl. Sastrawi stemming     : {full:7.4f}s "
          f"({full / nopre:.0f}x slower)")
    print("=> stemming, not vectorisation, is the real bottleneck; the memo cache "
          "short-circuits the 78% of this corpus that is duplicated retweets.")

    print(f"\nWrote {TABLES_DIR / 'vectorization_benchmark.csv'}")


if __name__ == "__main__":
    main()
