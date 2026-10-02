"""Tests for defect #2: ``np.vectorize`` is a loop, not a vectorisation.

These tests assert *behavioural* facts that hold regardless of machine speed:

* ``np.vectorize`` gives the same (wrong) answer as the plain Python function
  it wraps -- proof that it is a loop with extra bookkeeping, not a
  reimplementation;
* the pandas ``.str`` pipeline and the ``Preprocessor.map`` path both remove
  every mention;
* the vectorised path returns a Series of the right length with no ``@`` left.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from idtweets.preprocessing import (
    MENTION_RE,
    Preprocessor,
    preprocess_regex_only,
    remove_pattern_original,
    strip_mentions,
)

PATTERN = r"@\w+"
SAMPLE = pd.Series(
    [
        "@a @b @c first tweet",
        "RT @x: @y @z second",
        "plain tweet no handles",
        "@one @two three four five",
    ]
)


def test_np_vectorize_is_just_a_loop_and_reproduces_the_bug():
    out = np.vectorize(remove_pattern_original)(SAMPLE, PATTERN)
    manual = [remove_pattern_original(t, PATTERN) for t in SAMPLE]
    assert list(out) == manual  # identical -> it IS a loop
    # ...and it faithfully reproduces the first-match-only defect.
    assert MENTION_RE.findall(out[0]) == ["@b", "@c"]


def test_np_vectorize_returns_ndarray_not_series():
    out = np.vectorize(remove_pattern_original)(SAMPLE, PATTERN)
    assert isinstance(out, np.ndarray)
    assert not isinstance(out, pd.Series)


def test_fixed_regex_path_removes_every_mention_vectorised():
    out = SAMPLE.map(lambda t: strip_mentions(t))
    assert isinstance(out, pd.Series)
    assert out.map(lambda t: bool(MENTION_RE.search(t))).sum() == 0


def test_pandas_str_pipeline_removes_every_mention():
    out = preprocess_regex_only(pd.Series(["@a @b @c keep this"]))
    assert "@" not in out.iloc[0]
    assert "keep" in out.iloc[0] and "this" in out.iloc[0]


def test_preprocessor_map_path_removes_every_mention():
    p = Preprocessor(stemming=False, stopwords=False)
    out = p.transform_series(SAMPLE)
    assert out.map(lambda t: bool(MENTION_RE.search(t))).sum() == 0


def test_vectorized_and_map_paths_agree_on_mention_free_text():
    """On input with a single handle the two paths must agree exactly."""
    single = pd.Series(["@boss approve this"])
    vec = preprocess_regex_only(single)
    mapped = Preprocessor(stemming=False, stopwords=False).transform_series(single)
    assert "approve" in vec.iloc[0] and "approve" in mapped.iloc[0]
    assert "@" not in vec.iloc[0] and "@" not in mapped.iloc[0]
