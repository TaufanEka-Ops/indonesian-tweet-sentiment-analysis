# Case study 2 — Modi / Indian-election sentiment classification

## Data

`data/raw/modi_twitter_data.xlsx` — the Kaggle **"Twitter_Data"** set
(English), columns `text` + `category` ∈ {−1, 0, 1}. See
[`docs/data_card.md`](../../docs/data_card.md) for provenance and licensing.

## Preparation pipeline

`prepare_dataset()` in `modi_election_sentiment.py` performs, in order:

| step | rows affected |
|---|---:|
| rows loaded | 5,324 |
| drop rows with no text | 1 |
| repair CP1252 mojibake (`governanceâ` → `governance`) | 595 |
| **drop rows containing contact PII (phone/e-mail)** | **5** |
| preprocess (English preset: lowercase, URLs/mentions/hashtags/punctuation/digits, English stop-words, **no Sastrawi stemming**) | 3 rows became empty → dropped |
| drop exact duplicate cleaned texts | 84 |
| **final usable rows** | **5,232** |

Class balance after cleaning: negative 1,158 (22.1%) · neutral 1,855 (35.5%) ·
positive 2,219 (42.4%).

The English preset matters: running the Indonesian stop-word list or Sastrawi
over English text would be a category error, so
`make_english_preprocessor()` disables both. This is a deliberate difference
from case study 1.

## Evaluation protocol

* **Features:** TF-IDF word 1–2 grams, `min_df=2`, `max_df=0.95`,
  `sublinear_tf=True`, inside the pipeline so it is refit per fold (no
  vocabulary leakage).
* **Models:** majority-class `DummyClassifier`, `MultinomialNB(alpha=0.3)`,
  `LogisticRegression(C=4, class_weight="balanced")`,
  `LinearSVC(C=0.5, class_weight="balanced")`.
* **Cross-validation:** repeated stratified 5 folds × 3 repeats = 15 fits per
  model.
* **Headline metric:** macro-F1 (labels are imbalanced; accuracy flatters a
  majority-class predictor).
* **Holdout:** stratified 80/20, best CV model, confusion matrix + error
  analysis on the held-out rows only.
* **Uncertainty:** 1,000-sample percentile bootstrap CI on the held-out
  macro-F1.

## Results

Repeated stratified 5×3 CV:

| model | accuracy | macro-F1 | weighted-F1 |
|---|---:|---:|---:|
| LinearSVC | 0.7343 ± 0.0124 | **0.7198 ± 0.0119** | 0.7330 |
| Logistic Regression | 0.7297 ± 0.0125 | 0.7158 ± 0.0126 | 0.7293 |
| Multinomial Naive Bayes | 0.6243 ± 0.0136 | 0.5773 ± 0.0149 | 0.6045 |
| Majority baseline | 0.4241 ± 0.0004 | 0.1985 ± 0.0001 | 0.2526 |

Held-out (LinearSVC, n = 1,047): accuracy 0.7163, macro-F1 0.7039
(95% CI [0.6760, 0.7321]), Cohen's κ 0.5587.

**Do not over-read the ranking.** LinearSVC and Logistic Regression differ by
0.004 macro-F1, which is *less than one fold standard deviation (0.012)*. The
honest statement is "LinearSVC and Logistic Regression are
indistinguishable at this corpus size"; the naive-bayes result is genuinely
worse.

## Error analysis

297 of 1,047 held-out tweets were misclassified. Direction breakdown:

| direction | count | share of errors |
|---|---:|---:|
| positive → neutral | 74 | 24.9% |
| neutral → positive | 63 | 21.2% |
| negative → positive | 50 | 16.8% |
| negative → neutral | 41 | 13.8% |
| positive → negative | 41 | 13.8% |
| neutral → negative | 28 | 9.4% |

Nearly half of all errors (46.1%) sit on the neutral boundary: short tweets with
no explicit sentiment cue ("passed expected modi government") are labelled
neutral or positive by the model and negative in the ground truth, or vice
versa. Outright polarity inversion (positive ↔ negative) is only 13.8% + 16.8%.

Real examples are in `results/tables/modi_error_examples.csv`; the terms most
frequent among errors are in `results/tables/modi_error_terms.csv` (dominated by
`modi` 298, `india` 43, `people` 37 — i.e. topic words, not sentiment words,
which is consistent with boundary noise rather than a broken feature space).

## Outputs

* `results/tables/modi_model_comparison.csv`
* `results/tables/modi_holdout_metrics.csv`
* `results/tables/modi_classification_report.csv`
* `results/tables/modi_confusion_matrix.csv`
* `results/tables/modi_error_directions.csv`, `modi_error_examples.csv`,
  `modi_error_terms.csv`
* `results/tables/modi_top_features.csv`
* `results/tables/modi_label_distribution.csv`, `modi_data_audit.json`
* `results/figures/modi_model_comparison.png`, `modi_confusion_matrix.png`,
  `modi_confusion_matrix_normalized.png`, `modi_class_distribution.png`,
  `modi_per_class_f1.png`, `modi_error_directions.png`

## Run it

```bash
python scripts/05_train_modi_sentiment.py   # ~3 min (LR is the slow one)
python scripts/06_error_analysis.py
```
