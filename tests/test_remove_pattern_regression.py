"""Regression tests for the ``remove_pattern`` first-match-only defect.

The original coursework helper was::

    def remove_pattern(tweet, pattern):
        r = re.findall(pattern, tweet)
        for i in r:
            tweet = re.sub(i, '', tweet)
            return tweet          # returns on the FIRST iteration

Because the ``return`` sits inside the ``for`` body, only the first match is
ever removed. A tweet with three ``@mentions`` loses one and keeps two.

These tests do three things:

1. ``test_original_logic_is_buggy`` *pins* the defect: it asserts the original
   implementation really does keep later mentions, so the suite fails loudly if
   anyone later "helps" by copying that snippet back in.
2. ``test_fixed_strips_every_mention`` asserts the corrected implementation
   removes all of them.
3. ``test_fixed_beats_original_on_real_corpus`` runs both implementations over
   the shipped Novel Baswedan tweets and asserts the fixed one leaves zero
   stray mentions while the original leaves many -- a measured impact number,
   not a synthetic toy.
"""

from __future__ import annotations

import re

import pytest

from idtweets.io import load_novel
from idtweets.preprocessing import MENTION_RE, remove_pattern_original, strip_mentions

MENTION_PATTERN = r"@\w+"


# --------------------------------------------------------------------------- #
# 1. Pin the original defect
# --------------------------------------------------------------------------- #
def test_original_logic_is_buggy():
    """The original helper removes only the FIRST mention (the defect)."""
    tweet = "@alpha @beta @gamma hello"
    out = remove_pattern_original(tweet, MENTION_PATTERN)
    remaining = MENTION_RE.findall(out)
    assert remaining == ["@beta", "@gamma"], (
        "Expected the original logic to keep the 2nd and 3rd mentions; "
        f"got {remaining!r}"
    )


@pytest.mark.parametrize(
    "tweet,n_kept",
    [
        ("@a @b hi", 1),
        ("@a @b @c hi", 2),
        ("RT @x: @y @z text", 2),
    ],
)
def test_original_keeps_all_but_first(tweet, n_kept):
    out = remove_pattern_original(tweet, MENTION_PATTERN)
    assert len(MENTION_RE.findall(out)) == n_kept


# --------------------------------------------------------------------------- #
# 2. The fix
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "tweet,payload",
    [
        ("@a @b hi", "hi"),
        ("@a @b @c hi", "hi"),
        ("RT @x: @y @z text", "text"),
        ("@one", ""),
        ("no mention here", "no mention here"),
        ("@user_1 @user_2 @user_3 @user_4 @user_5 done", "done"),
    ],
)
def test_fixed_strips_every_mention(tweet, payload):
    out = strip_mentions(tweet)
    assert MENTION_RE.search(out) is None, f"stray mention left in {out!r}"
    if payload:
        assert payload in out
    else:
        assert out.strip() == ""


def test_fixed_and_original_agree_when_only_one_mention():
    """Where the original happened to be correct, the fix must not regress."""
    tweet = "@only one mention"
    assert MENTION_RE.findall(remove_pattern_original(tweet, MENTION_PATTERN)) == []
    assert MENTION_RE.findall(strip_mentions(tweet)) == []


# --------------------------------------------------------------------------- #
# 3. Measured impact on the real corpus
# --------------------------------------------------------------------------- #
def test_fixed_beats_original_on_real_corpus():
    df = load_novel()
    texts = df["text"].astype(str)

    original_out = texts.map(lambda t: remove_pattern_original(t, MENTION_PATTERN))
    fixed_out = texts.map(strip_mentions)

    original_leftover = original_out.map(lambda t: len(MENTION_RE.findall(t)))
    fixed_leftover = fixed_out.map(lambda t: len(MENTION_RE.findall(t)))

    assert int(fixed_leftover.sum()) == 0
    assert int(original_leftover.sum()) > 0

    # Rows the original got WRONG (it left at least one mention behind).
    broken_rows = int((original_leftover > 0).sum())
    # Rows where only the fix is fully correct.
    assert broken_rows > 100, (
        "Expected the first-match bug to affect many real rows; "
        f"only {broken_rows} were affected."
    )

    # A tweet with a single mention must produce identical *content* in both.
    # (The fixed version substitutes a space, the original deleted the match,
    # so compare on whitespace-normalised tokens.)
    single = texts[texts.str.count(MENTION_PATTERN) == 1]
    if len(single):
        assert all(
            remove_pattern_original(t, MENTION_PATTERN).split()
            == strip_mentions(t).split()
            for t in single.head(200)
        )


def test_no_duplicate_implementation_slip_through():
    """``re.sub`` with a whole-string pattern is what makes the fix correct."""
    tweet = " ".join(f"@user{i}" for i in range(20))
    assert strip_mentions(tweet).strip() == ""
    assert len(MENTION_RE.findall(remove_pattern_original(tweet, MENTION_PATTERN))) == 19
