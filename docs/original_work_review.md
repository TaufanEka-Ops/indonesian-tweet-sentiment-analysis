# Original work — review and honest evaluation

This document records what the original coursework did, what was wrong with it,
and what was fixed here. It is written so a reviewer can see the delta rather
than take it on faith.

## What the original code was

Two notebooks, essentially the same linear cell sequence:

* `Project UTS/Project/Untitled.ipynb` — Indonesian tweet preprocessing on the
  Novel Baswedan corpus (2,727 tweets). Produced `Tweet hasil preprocessing.xls`.
* `Tugas 3/Tugas 3/Untitled.ipynb` — a near-identical copy of the same pipeline
  on a one-row sample (`text.xlsx`, 1 tweet). It is a duplicate of the Tugas 4
  copy and was **not** rebuilt as a separate project.
* `Modi_Election_Sentiment_Analysis.ipynb` — the supervised attempt: a
  `CountVectorizer` + `TfidfTransformer` + one `MultinomialNB`, a
  `train_test_split`, an accuracy print, a confusion matrix, and a 10-fold
  `cross_val_score` on accuracy.

Characteristic features: no functions beyond two regex helpers, no packaging, no
tests, hard-coded Colab paths (`/content/Twitter_Data.xlsx`), and no separation
between preprocessing, features and modelling.

## Defect 1 — `remove_pattern` removed only the first match

```python
def remove_pattern(tweet,pattern):
    r = re.findall(pattern, tweet)
    for i in r:
        tweet = re.sub(i,'',tweet)
        return tweet          # <-- inside the loop
```

The `return` is inside the `for` body, so the function exits after the first
iteration. Any tweet with *n* mentions keeps *n − 1* of them.

**Measured impact on the real corpus** (`scripts/02_preprocess_novel_baswedan.py`):

| metric | value |
|---|---|
| tweets | 2,727 |
| tweets containing a mention | 2,510 |
| tweets containing ≥2 mentions | 634 |
| total mentions | 3,616 |
| mentions the original removed | 2,517 |
| **mentions the original left behind** | **1,099 (30.4%)** |
| **tweets the original got wrong** | **627** |
| mentions the corrected function leaves | 0 |

This is a silent correctness bug: nothing crashes, the output column simply
still contains `@handles`, which then flow into the vocabulary of any
downstream model (or, as here, into a "cleaned" file that a human reads).

**Fix and proof:** `strip_mentions()` performs one `re.sub` over the whole
string. `tests/test_remove_pattern_regression.py` pins the original behaviour
(it asserts the old helper really does keep mentions 2..n), asserts the new
function removes all of them, and runs both over the shipped corpus asserting
zero leftovers for the fix and >100 affected rows for the original.

## Defect 2 — `np.vectorize` is not a vectorisation

```python
df['remove_user'] = np.vectorize(remove_pattern)(df['text'], "@[\\w]*")
```

NumPy's own documentation calls `np.vectorize` "a convenience function … [that]
does not actually provide any performance improvements". It is a Python-level
loop with extra bookkeeping.

**Measured** (`scripts/04_vectorization_benchmark.py`, 2,727 tweets × 20 = 54,540
rows, best of 3 runs):

| method | ms / 1k rows | mentions left behind | correct |
|---|---:|---:|---|
| `np.vectorize(remove_pattern_original)` | 1.99 | 21,980 | no |
| `np.vectorize(strip_mentions)` | 1.17 | 0 | yes |
| list comprehension | 1.01 | 0 | yes |
| `pandas .str.replace` (vectorised regex) | 17.29 | 0 | yes |

The honest reading, which the repo adopts: `np.vectorize` buys nothing (a plain
comprehension of the same function is ~1.15× faster), and pandas `.str` is
**slower** here because per-call regex overhead dominates on short strings. The
fix is therefore not "swap in pandas and claim a speedup" — it is to drop the
wrapper, keep the memo cache for the 78% of this corpus that is duplicated
retweets, and be explicit that the real bottleneck is Sastrawi stemming
(~4,800× the regex-only path on unique texts). Both application paths ship and
are tested.

## Defect 3 — no evaluation

The original reported:

* one accuracy number (`0.628`) from one unstratified `train_test_split`;
* a confusion matrix with no per-class metrics;
* a 10-fold `cross_val_score` **on accuracy**, over `x_train` only, using a
  `MultinomialNB` fit on all of `x_train` beforehand.

Nothing was compared against a baseline, and the reported metric (accuracy)
ignores that the labels are imbalanced (42% / 36% / 22%).

**What replaces it** (`scripts/05_train_modi_sentiment.py`):

* majority-class baseline;
* repeated stratified 5×3 cross-validation (15 fits per model);
* macro-F1 as the headline metric, with per-class precision/recall/F1;
* a stratified held-out 80/20 split for the confusion matrix and error analysis;
* a 1,000-sample bootstrap CI on the held-out macro-F1;
* a real error analysis with quoted misclassified tweets and error-direction
  breakdown.

## Other issues found and addressed

| issue | where | resolution |
|---|---|---|
| Hard-coded `/content/...` Colab path | Modi notebook | `idtweets.io` resolves paths from the repo root |
| CP1252 mojibake (`governanceâ`, `modiâ`) | Modi corpus, 595 rows | `fix_mojibake()` before tokenising |
| Rows containing phone numbers | Modi corpus, 5 rows | dropped by `idtweets.pii`, count logged |
| No train/test stratification | both | `stratified` split and `StratifiedKFold` used throughout |
| One `apply()` per pipeline stage | both notebooks | single `Preprocessor`/vectorised path, pure functions |
| Duplicate dataset copy (Tugas 3 vs Tugas 4) | repo | not rebuilt; noted here and in the README |
| No reusable code | both | `src/idtweets` package, 77 unit tests |
| No provenance / licence note | both | `docs/data_card.md` |

## What the original got right

To be fair: the label-choice of TF-IDF over raw counts, the use of
`TweetTokenizer` with `strip_handles`, and the recognition that Indonesian
slang needs normalisation were all sound instincts. The pipeline was also
deterministic and reproducible for the author — it just was not *evaluated*,
*packaged*, or *tested*.
