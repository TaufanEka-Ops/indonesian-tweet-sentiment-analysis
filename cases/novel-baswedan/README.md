# Case study 1 — Novel Baswedan tweet corpus (Indonesian)

## Data

`data/raw/novel_baswedan.xlsx` — 2,727 Indonesian tweets (`id_str`, `text`)
captured around the Novel Baswedan acid-attack case (2020). **No label column.**

| property | value |
|---|---:|
| rows | 2,727 |
| exact-duplicate texts | 2,136 (78.3%) |
| retweets (`RT ...`) | 2,366 (86.8%) |
| rows containing ≥1 `@mention` | 2,510 |
| rows containing ≥2 `@mentions` | 634 |
| total mentions | 3,616 |
| rows with contact PII | 0 |

## Why there is no classifier here

The dataset has no ground-truth sentiment label. Training a model on invented
labels, or reporting a "sentiment score" from an unlabelled corpus, would be a
claim the data cannot support. What the data *does* support is descriptive
analysis, which is what this case study does. The supervised work is in
[`../modi-election/`](../modi-election/).

## What this case study produces

`python scripts/02_preprocess_novel_baswedan.py`

* `results/tables/novel_preprocessed.csv` — the cleaned corpus (462 unique tweets)
* `results/tables/novel_defect_impact.csv` — the mention bug, measured
* `results/tables/novel_examples_bug.csv` — concrete before/after rows

`python scripts/03_explore_novel_baswedan.py`

* `results/tables/novel_hashtags.csv`, `novel_top_terms.csv`,
  `novel_entities.csv`, `novel_length_stats.csv`, `novel_dedup_effect.csv`
* `results/figures/novel_hashtags.png`, `novel_top_terms.png`,
  `novel_entities.png`, `novel_token_hist.png`

## Findings

* **The corpus is a retweet monoculture.** 78.3% of rows are exact duplicates,
  so the de-duplicated corpus is only 462 tweets. Any frequency statistic
  computed on the raw 2,727 rows measures retweet behaviour, not opinion.
* **One campaign hashtag dominates.** `#RekayasaDidzolimi` appears in 1,366 of
  2,727 tweets; the next tag has 46. This is the signature of a coordinated
  campaign, which is itself a finding — and another reason not to read the
  volume as public sentiment.
* **Named-entity concentration.** In the de-duplicated corpus, 90.7% of tweets
  mention Novel Baswedan, 15.4% Jokowi, 6.3% the KPK, 6.1% prosecutors. Several
  entities in our watch-list (Prabowo, Fadli Zon, Amien Rais) never appear.
* **The bug mattered here.** 627 tweets (23% of the corpus) went through the
  original pipeline with mentions still attached; 30.4% of all mentions survived
  cleaning. See `docs/original_work_review.md`.
* **Cleaning removes ~half the characters** (mean 194 raw → 96 cleaned) and
  leaves a mean of 14.6 tokens per de-duplicated tweet after stop-word removal
  and Sastrawi stemming.

## Module

`novel_baswedan_analysis.py` — loaded via `idtweets.cases.load_case_module`
(the directory name contains a hyphen, so it is not a dotted package).

| function | purpose |
|---|---|
| `load_raw()` | load the corpus via `idtweets.io` |
| `defect_impact(df)` | count mentions left behind by the original vs fixed logic |
| `preprocess(df)` | apply the full Indonesian `Preprocessor` |
| `deduplicate(df)` | drop exact duplicate cleaned tweets |
| `hashtag_table(df)` | hashtag frequency |
| `top_terms(df)` | most frequent cleaned terms |
| `entity_table(df)` | tweets per name/entity cluster |
| `length_stats(df)` | raw vs cleaned length summary |
