"""Feature extraction: TF-IDF and count vectorisers wrapped in sklearn Pipelines."""

from __future__ import annotations

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def build_tfidf(
    *,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
    sublinear_tf: bool = True,
    max_features: int | None = None,
    use_idf: bool = True,
) -> TfidfVectorizer:
    """Word-level TF-IDF vectoriser with sensible defaults for short tweets.

    ``min_df=2`` drops the many one-off typos that dominate tweet vocabularies;
    ``sublinear_tf`` compresses the effect of repeated retweet vocabulary.
    """
    return TfidfVectorizer(
        lowercase=False,  # text is already normalised upstream
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        max_features=max_features,
        use_idf=use_idf,
        strip_accents=None,
        token_pattern=r"(?u)\b\w+\b",
    )


def build_count(**kwargs) -> CountVectorizer:
    """Count vectoriser (the bag-of-words baseline from the original notebook)."""
    defaults = dict(
        lowercase=False,
        ngram_range=(1, 1),
        min_df=2,
        token_pattern=r"(?u)\b\w+\b",
    )
    defaults.update(kwargs)
    return CountVectorizer(**defaults)


def build_char_tfidf(
    ngram_range: tuple[int, int] = (3, 5), min_df: int = 3, sublinear_tf: bool = True
) -> TfidfVectorizer:
    """Character n-gram TF-IDF -- robust to misspellings, useful as a

    secondary feature space for tweet text.
    """
    return TfidfVectorizer(
        lowercase=False,
        analyzer="char_wb",
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
    )
