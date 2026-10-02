# Methodology

## 1. Scope

Two case studies, deliberately kept apart because they support different claims:

| | `cases/novel-baswedan/` | `cases/modi-election/` |
|---|---|---|
| Data | 2,727 Indonesian tweets, Novel Baswedan case | 5,325 English tweets, Indian election |
| Labels | none | `-1 / 0 / 1` sentiment |
| Question | what does the corpus look like after honest cleaning? | can sentiment be predicted, and how well? |
| Output | descriptive tables + figures | cross-validated metrics, confusion matrix, error analysis |

No sentiment model is trained on the Indonesian corpus, because it has no
ground-truth labels. Producing a "sentiment score" from an unlabelled
single-issue tweet dump would be an unsupported claim, and the original
coursework never claimed one either.

## 2. Preprocessing

Implemented in `src/idtweets/preprocessing.py` as pure functions plus one
configurable `Preprocessor`. Order of operations:

1. **Mojibake repair** (`fix_mojibake`) — the Modi file was stored as UTF-8
   bytes decoded as CP1252, so `governanceâ` and `modiâ` appear. 595 rows are
   affected; without repair these become vocabulary items.
2. **URL removal**, **retweet-marker removal**, **mention removal**,
   **hashtag marker removal** (word kept), **cashtag removal**, **digit
   removal**, **lower-casing**, **punctuation removal**, whitespace collapse.
3. **Tokenisation** — NLTK `TweetTokenizer(preserve_case=False,
   strip_handles=True, reduce_len=True)`.
4. **Slang normalisation** — token-level lookup into a 90-entry map
   (`src/idtweets/slang.py`). Token-level, not substring, so `gakarta` is not
   rewritten. Applied **only to the Indonesian corpus**.
5. **Stop-word removal** — NLTK Indonesian list for case 1, NLTK English list
   for case 2.
6. **Stemming** — Sastrawi, Indonesian corpus only.

The two corpora get different settings on purpose. The Modi corpus is English;
running Sastrawi over English words or filtering them with an Indonesian
stop-word list would be a category error, so `make_english_preprocessor()`
disables both.

### 2.1 The three defects that were fixed

**(a) `remove_pattern` stripped only the first match.** The original returned
from inside its `for` loop:

```python
def remove_pattern(tweet, pattern):
    r = re.findall(pattern, tweet)
    for i in r:
        tweet = re.sub(i, '', tweet)
        return tweet        # returns on the FIRST iteration
```

On the real corpus this left **1,099 of 3,616 mentions in place, affecting 627
of 2,727 tweets**. The replacement is a single `re.sub` over the whole string
(`strip_mentions`), with no per-match loop. Pinned by
`tests/test_remove_pattern_regression.py`, which asserts *both* that the
original behaves this way and that the fix removes everything.

**(b) `np.vectorize` is not a vectorisation.** It is a documented convenience
wrapper around a Python loop. Measured in
`scripts/04_vectorization_benchmark.py` (2,727 tweets × 20 repeats, best of 3):
`np.vectorize(strip_mentions)` is **1.15× slower** than a plain list
comprehension of the same function, and pandas `.str.replace` is slower still
at this row length (~17 ms vs ~1 ms per 1,000 rows) because per-call regex
overhead dominates on short strings. The honest conclusion is not "pandas is
always faster" — it is that `np.vectorize` buys nothing, and the real cost in
this pipeline is Sastrawi stemming (~4,800× the regex-only path on unique
texts). The pipeline therefore offers three application paths and documents
when each applies.

**(c) No evaluation.** The original reported a single accuracy on a single
80/20 split and never compared against a baseline. Here:

* **majority-class baseline** (`DummyClassifier(strategy="most_frequent")`)
  so the reader can see what "doing nothing" scores;
* **repeated stratified 5×3 cross-validation** (15 fits), because a single
  5-fold split has a wide variance band at this corpus size;
* **macro-F1 as the headline metric** — the label distribution is
  22.1 / 35.5 / 42.4, so accuracy flatters a majority-class predictor;
* a **stratified held-out 80/20 split** for the confusion matrix and the
  error analysis, so those are computed on data the model has not seen;
* a **1,000-sample percentile bootstrap CI** on the held-out macro-F1, to make
  the sampling uncertainty explicit rather than implying a point estimate is
  exact.

## 3. Features and models

Everything lives inside an sklearn `Pipeline`, so TF-IDF is refit on every CV
fold and there is **no vocabulary leakage** from the test fold.

* TF-IDF, word 1–2 grams, `min_df=2`, `max_df=0.95`, `sublinear_tf=True`.
* Models: `DummyClassifier` (baseline), `MultinomialNB(alpha=0.3)`,
  `LogisticRegression(C=4, class_weight="balanced")`,
  `LinearSVC(C=0.5, class_weight="balanced")`.
* Class weights are balanced because the labels are imbalanced.

## 4. Error analysis

`scripts/06_error_analysis.py` reports, on the held-out fold:

* the **direction** of each error (`negative → neutral`, etc.) and its share of
  all errors, not just a global error count;
* up to 10 **real**, unedited misclassified tweets per true class;
* the most frequent terms among the misclassified rows.

Example text is quoted only as already-public tweet content. No handles,
e-mails or phone numbers are reproduced; rows with contact PII were removed
before modelling.

## 5. Reproducibility

* `make all` regenerates every table, figure and notebook output.
* Every number in the README comes from a script in `scripts/`; none is typed
  by hand.
* Random seeds are fixed (`RANDOM_STATE = 42`) in `src/idtweets/modelling.py`.

## 6. Known limitations

* **Corpus size.** 5,232 usable English tweets and 2,727 Indonesian tweets is
  small for text classification. Several models land within one standard
  deviation of each other, and the notebook says so rather than declaring a
  winner.
* **Label provenance.** The Modi labels were produced by the original dataset
  authors with an undisclosed method; the ceiling on achievable macro-F1 is
  therefore unknown, and no annotation-quality audit was possible.
* **Domain shift.** The Modi corpus is Indian political text from 2019; nothing
  here should be transferred to other domains or time periods without
  re-evaluation.
* **No Indonesian classifier.** The Indonesian corpus is unlabelled, so case
  study 1 makes no predictive claim.
