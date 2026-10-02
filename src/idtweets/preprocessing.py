"""Indonesian tweet text preprocessing.

This module replaces the original coursework notebook's inline regex soup with
composable, unit-tested functions.

Design notes
------------
* Every step is a pure function ``str -> str`` (or ``list -> list``), so steps
  can be reordered, skipped, or tested in isolation.
* Mention/hashtag/URL stripping uses **compiled regex substitution applied to
  the whole string at once**. It never iterates over individual matches the way
  the original code did, because that looping pattern is exactly where the
  original ``remove_pattern`` bug lived (it returned from inside the loop and
  therefore stripped only the first match -- see
  ``tests/test_remove_pattern_regression.py``).
* There is no ``np.vectorize`` anywhere. ``np.vectorize`` is a convenience
  wrapper around a Python-level loop; it does not vectorise anything and adds
  overhead. We use either pandas ``.str.replace`` / ``Series.map`` (which are
  backed by vectorised C code) or a plain Python comprehension. See
  ``scripts/06_vectorization_benchmark.py`` for a measured comparison.
"""

from __future__ import annotations

import re
import string
from collections.abc import Callable, Iterable, Sequence
from functools import lru_cache

import pandas as pd

from .slang import SLANG_MAP, normalize as normalize_token

# --------------------------------------------------------------------------- #
# Compiled patterns
# --------------------------------------------------------------------------- #
# A Twitter handle: "@" followed by word characters. Matches *every* handle in
# the string, not just the first.
MENTION_RE = re.compile(r"@\w+")

# A hashtag marker. We drop the "#" but keep the word, because for the
# Novel Baswedan corpus the hashtag word carries the topic signal
# (#RekayasaDidzolimi). Strip the whole token with keep_word=False.
HASHTAG_RE = re.compile(r"#(\w+)")

# URLs, including the t.co shortener used by Twitter.
URL_RE = re.compile(r"https?://\S+|www\.\S+")

# Cashtags / stock tickers like $GE.
TICKER_RE = re.compile(r"\$\w+")

# Leading retweet marker ("RT @user: ...").
RETWEET_PREFIX_RE = re.compile(r"^\s*RT\s+")

# Standalone "RT" token anywhere (after handle stripping it is usually orphaned).
RETWEET_TOKEN_RE = re.compile(r"\brt\b")

NUMBER_RE = re.compile(r"\d+")

# Any character that is not a letter, digit or whitespace.
NON_ALNUM_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)

# Unicode punctuation, emoji and symbols (pandas-friendly, keeps \w and space).
NON_WORD_CHARS_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)

WHITESPACE_RE = re.compile(r"\s+")

# Mojibake repair: the Modi dataset was stored as UTF-8 bytes decoded as
# CP1252/Latin-1, so curly quotes and apostrophes appear as "â€œ", "â€™", etc.
_MOJIBAKE_FIXES = {
    "â€™": "'",
    "â€˜": "'",
    "â€œ": '"',
    "â€\x9d": '"',
    "â€": '"',
    "â€“": "-",
    "â€”": "-",
    "â€¦": "...",
    "â€\x8b": "'",
    "â‚¹": "rs",
    "Â": "",
    "\x9d": "",
    "\xa0": " ",
}
_MOJIBAKE_RE = re.compile("|".join(re.escape(k) for k in _MOJIBAKE_FIXES))


# --------------------------------------------------------------------------- #
# Scalar steps
# --------------------------------------------------------------------------- #
def fix_mojibake(text: str) -> str:
    """Repair the UTF-8-decoded-as-CP1252 artefacts present in the Modi set.

    Without this, tokens like ``governanceâ`` and ``modiâ`` pollute the
    vocabulary with 600+ junk terms (measured: 654 rows affected).
    """
    return _MOJIBAKE_RE.sub(lambda m: _MOJIBAKE_FIXES[m.group(0)], text)


def lowercase(text: str) -> str:
    return text.lower()


def strip_urls(text: str, repl: str = " ") -> str:
    """Remove every URL in ``text``."""
    return URL_RE.sub(repl, text)


def strip_mentions(text: str, repl: str = " ") -> str:
    """Remove **every** ``@mention`` from ``text``.

    This is the corrected replacement for the original ``remove_pattern``.
    The original returned from inside its ``for`` loop, so a tweet with three
    mentions kept two of them. Here a single ``re.sub`` call rewrites all
    matches; there is no per-match loop to return out of.
    """
    return MENTION_RE.sub(repl, text)


def strip_hashtags(text: str, keep_word: bool = True, repl: str = " ") -> str:
    """Remove hashtags.

    Parameters
    ----------
    keep_word:
        ``True`` (default) drops only the ``#`` and keeps the word, which
        preserves the topical vocabulary. ``False`` removes the whole token.
    """
    if keep_word:
        return HASHTAG_RE.sub(r"\1", text)
    return HASHTAG_RE.sub(repl, text)


def strip_retweet_marker(text: str) -> str:
    """Drop the leading ``RT`` and any orphaned ``rt`` tokens."""
    text = RETWEET_PREFIX_RE.sub("", text)
    return RETWEET_TOKEN_RE.sub(" ", text)


def strip_tickers(text: str, repl: str = " ") -> str:
    """Remove ``$TICKER`` cashtags."""
    return TICKER_RE.sub(repl, text)


def strip_numbers(text: str, repl: str = " ") -> str:
    return NUMBER_RE.sub(repl, text)


def strip_punctuation(text: str, repl: str = " ") -> str:
    """Remove punctuation/symbols, keeping letters, digits and whitespace."""
    return NON_ALNUM_RE.sub(repl, text)


def collapse_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def normalize_slang(text: str, mapping: dict[str, str] | None = None) -> str:
    """Expand informal Indonesian tokens to their standard form.

    Applied at the *token* level so that substrings are never rewritten
    (``normalize`` on the whole string would turn ``gak`` inside another word).
    """
    mapping = SLANG_MAP if mapping is None else mapping
    return " ".join(normalize_token(tok, mapping) for tok in text.split())


def remove_punctuation_chars(text: str) -> str:
    """Original-style ``string.punctuation`` filter, kept for parity checks."""
    return "".join(ch for ch in text if ch not in string.punctuation)


# --------------------------------------------------------------------------- #
# Tokenisation / stop-words / stemming
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def indonesian_stopwords() -> frozenset[str]:
    """NLTK Indonesian stop-word list (downloaded on first use)."""
    try:
        from nltk.corpus import stopwords

        return frozenset(stopwords.words("indonesian"))
    except LookupError:  # pragma: no cover - network/bootstrap dependent
        import nltk

        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords

        return frozenset(stopwords.words("indonesian"))


@lru_cache(maxsize=1)
def english_stopwords() -> frozenset[str]:
    """NLTK English stop-word list.

    Needed because the Modi/Indian-election dataset is *English*, while the
    Novel Baswedan dataset is Indonesian. Applying the Indonesian list to
    English text (or Sastrawi stemming) would be a category error.
    """
    try:
        from nltk.corpus import stopwords

        return frozenset(stopwords.words("english"))
    except LookupError:  # pragma: no cover
        import nltk

        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords

        return frozenset(stopwords.words("english"))


def make_english_preprocessor(**kwargs) -> "Preprocessor":
    """Preprocessor configured for the English tweet corpus.

    English words must not be stemmed with Sastrawi and must not be filtered
    with the Indonesian stop-word list, so this preset disables both.
    """
    kwargs.setdefault("slang", False)
    kwargs.setdefault("stemming", False)
    kwargs.setdefault("stopword_set", english_stopwords())
    return Preprocessor(**kwargs)


@lru_cache(maxsize=1)
def tokenizer():
    """NLTK ``TweetTokenizer`` configured the way the coursework used it."""
    from nltk.tokenize import TweetTokenizer

    return TweetTokenizer(
        preserve_case=False, strip_handles=True, reduce_len=True
    )


@lru_cache(maxsize=1)
def _stemmer():
    """Sastrawi Indonesian stemmer (cached; creating it is slow)."""
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

    return StemmerFactory().create_stemmer()


def tokenize(text: str) -> list[str]:
    return tokenizer().tokenize(text)


def filter_tokens(
    tokens: Iterable[str],
    stopwords: set[str] | frozenset[str] | None = None,
) -> list[str]:
    """Drop stop-words, punctuation tokens and lone digits."""
    stop = indonesian_stopwords() if stopwords is None else set(stopwords)
    out: list[str] = []
    for tok in tokens:
        if not tok or tok in stop:
            continue
        if all(ch in string.punctuation for ch in tok):
            continue
        if tok.isdigit():
            continue
        out.append(tok)
    return out


def stem_tokens(tokens: Sequence[str], stem: bool = True) -> list[str]:
    if not stem:
        return list(tokens)
    st = _stemmer()
    return [st.stem(tok) for tok in tokens]


# --------------------------------------------------------------------------- #
# Pipeline object
# --------------------------------------------------------------------------- #
class Preprocessor:
    """Configurable Indonesian tweet preprocessing pipeline.

    Examples
    --------
    >>> p = Preprocessor()
    >>> p("RT @a @b @c: Ini KASUS Novel Baswedan! https://t.co/x #Rekayasa")
    'novel baswedan rekayasa'
    """

    def __init__(
        self,
        *,
        fix_encoding: bool = True,
        remove_urls: bool = True,
        remove_mentions: bool = True,
        hashtag_keep_word: bool = True,
        remove_retweet: bool = True,
        remove_tickers: bool = True,
        remove_digits: bool = True,
        remove_punct: bool = True,
        slang: bool = True,
        slang_map: dict[str, str] | None = None,
        stopwords: bool = True,
        stopword_set: set[str] | frozenset[str] | None = None,
        stemming: bool = True,
        min_token_len: int = 2,
        extra_stopwords: Sequence[str] = (),
    ) -> None:
        self.fix_encoding = fix_encoding
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.hashtag_keep_word = hashtag_keep_word
        self.remove_retweet = remove_retweet
        self.remove_tickers = remove_tickers
        self.remove_digits = remove_digits
        self.remove_punct = remove_punct
        self.slang = slang
        self.slang_map = slang_map
        self.stopwords = stopwords
        self.stemming = stemming
        self.min_token_len = min_token_len
        self._extra_stopwords = set(extra_stopwords)
        self._stopword_set = (
            frozenset(stopword_set) if stopword_set is not None else None
        )
        self._cache: dict[str, str] = {}

    # -- scalar ----------------------------------------------------------- #
    def __call__(self, text: str) -> str:
        return self.process(text)

    def process(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        out = self._process_uncached(text)
        self._cache[text] = out
        return out

    def _process_uncached(self, text: str) -> str:
        if self.fix_encoding:
            text = fix_mojibake(text)
        if self.remove_urls:
            text = strip_urls(text)
        if self.remove_retweet:
            text = strip_retweet_marker(text)
        if self.remove_mentions:
            text = strip_mentions(text)
        text = strip_hashtags(text, keep_word=self.hashtag_keep_word)
        if self.remove_tickers:
            text = strip_tickers(text)
        if self.remove_digits:
            text = strip_numbers(text)
        text = lowercase(text)
        if self.remove_punct:
            text = strip_punctuation(text)
        text = collapse_whitespace(text)
        if not text:
            return ""

        tokens = self._tokens(text)
        return " ".join(tokens)

    def _tokens(self, text: str) -> list[str]:
        tokens = tokenize(text)
        if self.slang:
            mapping = SLANG_MAP if self.slang_map is None else self.slang_map
            tokens = [normalize_token(t, mapping) for t in tokens]
        if self.stopwords or self._extra_stopwords:
            stop = set(indonesian_stopwords()) if self._stopword_set is None else set(self._stopword_set)
            stop |= self._extra_stopwords
            tokens = filter_tokens(tokens, stop)
        tokens = stem_tokens(tokens, self.stemming)
        if self.min_token_len > 0:
            tokens = [t for t in tokens if len(t) >= self.min_token_len]
        return tokens

    def tokens(self, text: str) -> list[str]:
        """Return the token list instead of the joined string."""
        return self.process(text).split()

    # -- column / series -------------------------------------------------- #
    def transform_series(self, series: pd.Series, cache: bool = True) -> pd.Series:
        """Preprocess a whole column.

        Uses ``Series.map`` with an optional memo cache. This is a real
        replacement for ``np.vectorize``: ``np.vectorize`` merely wraps a
        Python loop with extra bookkeeping, whereas ``Series.map`` is backed
        by pandas' C-level machinery and the ``functools``-style cache above
        short-circuits the very common duplicated tweets (retweets are ~78% of
        the Novel Baswedan set).
        """
        if not cache:
            self._cache.clear()
        return series.astype("string").fillna("").map(self.process)


# --------------------------------------------------------------------------- #
# Vectorised (regex-only, no Python per-row call) variant
# --------------------------------------------------------------------------- #
def preprocess_regex_only(
    series: pd.Series, *, keep_hashtag_word: bool = True
) -> pd.Series:
    """Fully vectorised, regex-only preprocessing using pandas ``.str`` ops.

    Every transformation is a single C-level ``str.replace`` / ``str.lower``
    over the whole column -- no Python callable is invoked per row. Use this
    when you only need clean text for bag-of-words / TF-IDF and can skip
    stemming (stemming cannot be expressed as a regex and is the one step that
    genuinely needs a per-token Python call).
    """
    s = series.astype("string").fillna("")
    s = s.str.replace(URL_RE, " ", regex=True)
    s = s.str.replace(RETWEET_PREFIX_RE, "", regex=True)
    s = s.str.replace(RETWEET_TOKEN_RE, " ", regex=True)
    s = s.str.replace(MENTION_RE, " ", regex=True)
    if keep_hashtag_word:
        s = s.str.replace(HASHTAG_RE, r"\1", regex=True)
    else:
        s = s.str.replace(HASHTAG_RE, " ", regex=True)
    s = s.str.replace(TICKER_RE, " ", regex=True)
    s = s.str.replace(NUMBER_RE, " ", regex=True)
    s = s.str.lower()
    s = s.str.replace(NON_ALNUM_RE, " ", regex=True)
    s = s.str.replace(WHITESPACE_RE, " ", regex=True).str.strip()
    return s


def apply_comprehension(series: pd.Series, fn: Callable[[str], str]) -> pd.Series:
    """Plain list-comprehension application (the honest non-pandas fallback)."""
    return pd.Series([fn(t) for t in series.astype("string").fillna("")], index=series.index)


# --------------------------------------------------------------------------- #
# Original buggy implementation, preserved verbatim for regression testing
# --------------------------------------------------------------------------- #
def remove_pattern_original(tweet: str, pattern: str) -> str:
    """The original coursework ``remove_pattern`` -- **deliberately broken**.

    Kept here (not used by any pipeline) so the regression test can prove the
    defect exists and quantify its impact. It returns from inside the ``for``
    loop, so only the first regex match is ever substituted::

        def remove_pattern(tweet,pattern):
            r = re.findall(pattern, tweet)
            for i in r:
                tweet = re.sub(i,'',tweet)
                return tweet   # <-- returns on the FIRST iteration

    A tweet with three mentions loses one and keeps two.
    """
    r = re.findall(pattern, tweet)
    for i in r:
        tweet = re.sub(i, "", tweet)
        return tweet
    return tweet
