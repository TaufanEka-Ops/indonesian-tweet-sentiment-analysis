"""Case study 1 -- Novel Baswedan tweet corpus (Indonesian).

Preprocessing + exploratory analysis. There is **no supervised label** in this
dataset (it is a single-issue tweet dump), so the honest thing to do is
descriptive analysis: cleaning, de-duplication, hashtag concentration and
name-cluster frequency. No classifier is trained here -- that is case study 2.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from idtweets import preprocessing as pp
from idtweets.io import load_novel
from idtweets.preprocessing import MENTION_RE, remove_pattern_original, strip_mentions

# Name / entity clusters relevant to the case. Surface patterns are matched
# case-insensitively against the *cleaned* text.
ENTITY_PATTERNS: dict[str, list[str]] = {
    "Novel Baswedan": [r"\bnovel\b", r"\bbaswedan\b"],
    "Jokowi": [r"\bjokowi\b", r"\bpresiden\b"],
    "KPK": [r"\bkpk\b"],
    "Police / Bareskrim": [r"\bpolisi\b", r"\bbareskrim\b", r"\bkapolri\b"],
    "Prosecutors / JPU": [r"\bjpu\b", r"\bjaksa\b", r"\bkejaksaan\b"],
    "Ombudsman": [r"\bombudsman\b"],
    "Prabowo": [r"\bprabowo\b"],
    "Fadli Zon": [r"\bfadli\b", r"\bzon\b"],
    "Rizieq": [r"\brizieq\b", r"\bhabib\b"],
    "Amien Rais": [r"\bamien\b", r"\brais\b"],
    "Acid attack (air keras)": [r"\bair\b", r"\bkeras\b"],
    "Corruption": [r"\bkorupsi\b", r"\brekayasa\b"],
    "Human rights": [r"\bham\b", r"\bkeadilan\b"],
}

MENTION_PATTERN = r"@\w+"


def load_raw() -> pd.DataFrame:
    return load_novel()


def defect_impact(df: pd.DataFrame) -> dict:
    """Quantify the first-match-only mention bug on the real corpus."""
    texts = df["text"].astype(str)
    original_leftover = texts.map(
        lambda t: len(MENTION_RE.findall(remove_pattern_original(t, MENTION_PATTERN)))
    )
    fixed_leftover = texts.map(lambda t: len(MENTION_RE.findall(strip_mentions(t))))
    mentions_per_row = texts.str.count(MENTION_PATTERN)
    return {
        "rows_total": int(len(df)),
        "rows_with_any_mention": int((mentions_per_row > 0).sum()),
        "rows_with_multiple_mentions": int((mentions_per_row >= 2).sum()),
        "mentions_total": int(mentions_per_row.sum()),
        "original_stripped_total": int((mentions_per_row - original_leftover).sum()),
        "original_left_behind_total": int(original_leftover.sum()),
        "fixed_left_behind_total": int(fixed_leftover.sum()),
        "rows_original_got_wrong": int((original_leftover > 0).sum()),
        "retweet_share": float(
            texts.str.match(r"^\s*RT\b").mean()
        ),
        "duplicate_texts": int(texts.duplicated().sum()),
        "duplicate_share": float(texts.duplicated().mean()),
    }


def preprocess(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Add a ``clean`` column using the full Indonesian pipeline."""
    pre = pp.Preprocessor(**kwargs)
    out = df.copy()
    out["clean"] = pre.transform_series(out["text"])
    out["clean_token_count"] = out["clean"].str.split().map(len)
    return out


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate tweet texts (retweets dominate this corpus)."""
    return df.drop_duplicates(subset=["clean"]).reset_index(drop=True)


def hashtag_table(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Frequency of hashtags in the raw text."""
    counts: dict[str, int] = {}
    for text in df["text"].astype(str):
        for tag in re.findall(r"#(\w+)", text):
            counts[tag] = counts.get(tag, 0) + 1
    s = pd.Series(counts).sort_values(ascending=False).head(top_n)
    return s.rename_axis("hashtag").reset_index(name="count")


def top_terms(df: pd.DataFrame, top_n: int = 25, min_df: int = 3) -> pd.DataFrame:
    """Most frequent terms in the cleaned text."""
    from sklearn.feature_extraction.text import CountVectorizer

    texts = df["clean"].fillna("")
    texts = texts[texts.str.len() > 0]
    if texts.empty:
        return pd.DataFrame(columns=["term", "count"])
    vec = CountVectorizer(min_df=min_df)
    X = vec.fit_transform(texts)
    counts = np.asarray(X.sum(axis=0)).ravel()
    names = vec.get_feature_names_out()
    out = (
        pd.DataFrame({"term": names, "count": counts.astype(int)})
        .sort_values("count", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return out


def entity_table(df: pd.DataFrame) -> pd.DataFrame:
    """How many tweets touch each name/entity cluster."""
    text = df["clean"].fillna("")
    rows = []
    for label, patterns in ENTITY_PATTERNS.items():
        pat = re.compile("|".join(patterns))
        hits = text.str.contains(pat, regex=True, na=False)
        rows.append(
            {
                "entity": label,
                "matching_tweets": int(hits.sum()),
                "share": round(float(hits.mean()), 4),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("matching_tweets", ascending=False)
        .reset_index(drop=True)
    )


def length_stats(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[["text", "clean", "clean_token_count"]]
        .assign(
            raw_chars=lambda d: d["text"].astype(str).str.len(),
            clean_chars=lambda d: d["clean"].astype(str).str.len(),
        )[["raw_chars", "clean_chars", "clean_token_count"]]
        .describe()
        .round(2)
        .reset_index(names="statistic")
    )
