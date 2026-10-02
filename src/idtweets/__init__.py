"""Indonesian tweet text-mining toolkit.

A small, dependency-light, reusable package that turns the original one-off
coursework notebook into importable, tested functions.

Public surface
--------------
preprocessing : Indonesian/tweet text normalisation (mentions, URLs,
                hashtags, slang, stopwords, stemming).
pii           : conservative detection of contact PII in tweet rows.
features      : TF-IDF / count vectoriser builders.
modelling     : estimators, stratified CV, majority-class baseline.
evaluation    : metrics, confusion matrices, error analysis.
io            : dataset loading with explicit provenance.
"""

from . import evaluation, features, io, modelling, pii, preprocessing  # noqa: F401

__all__ = [
    "preprocessing",
    "pii",
    "features",
    "modelling",
    "evaluation",
    "io",
]

__version__ = "0.1.0"
