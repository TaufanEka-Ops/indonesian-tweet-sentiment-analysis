# Data card

Two third-party tweet datasets are used. **Neither was collected by this
project**; both were downloaded for university coursework (Text Mining,
IFM7346) and are reused here as-is. The MIT licence in this repository covers
the *code only*.

---

## 1. `data/raw/modi_twitter_data.xlsx` — Indian election / Modi sentiment

| | |
|---|---|
| Origin | Kaggle dataset **"Twitter_Data"** (Indian political / Narendra Modi tweets, 2019 period) |
| Rows | 5,325 (5,324 usable; 1 row has no text) |
| Columns | `text` (English tweet), `category` (int sentiment label) |
| Labels | `-1` negative (1,170) · `0` neutral (1,909) · `1` positive (2,245) |
| Language | English |
| Known defects | 595 rows carry CP1252/UTF-8 **mojibake** (`governanceâ`, `modiâ`); 49 exact duplicate texts |

**How to obtain it yourself:** search Kaggle for `Twitter_Data` (the file is
distributed as `Twitter_Data.xlsx` / `Twitter_Data.csv` with columns
`text,category`). The copy shipped here is the same file supplied with the
course, so the numbers in `results/` are reproducible from the file in `data/`.

**Licensing / reuse:** the dataset's own Kaggle page governs its reuse. It is
included here for reproducibility of a portfolio analysis; if you intend to
redistribute it further, check the original Kaggle licence first.

**Privacy:** 25 rows contain digit sequences that look like contact details; 5
of them survive a conservative keyword/format filter and are **dropped before
any modelling** (see `src/idtweets/pii.py` and
`results/tables/modi_data_audit.json`). No row is dropped silently.

---

## 2. `data/raw/novel_baswedan.xlsx` — Novel Baswedan acid-attack tweets

| | |
|---|---|
| Origin | Indonesian-language Twitter capture around the **Novel Baswedan** acid-attack case (2020), supplied with the coursework |
| Rows | 2,727 |
| Columns | `id_str` (tweet id), `text` (Indonesian tweet) |
| Language | Indonesian (heavy informal/slang spelling) |
| Labels | **none** — this is a single-issue dump, not a labelled sentiment set |
| Known characteristics | 86.8% retweets; 2,136 duplicate texts (78.3%); 634 tweets mention ≥2 accounts; top hashtag `#RekayasaDidzolimi` (1,366 tweets) |

**Why there is no classifier here.** With no ground-truth label, any "sentiment
score" would be invented. Case study 1 is therefore descriptive only
(preprocessing, hashtag and entity analysis). The supervised work lives in case
study 2.

**Privacy:** no e-mail or phone-shaped contact details were found in this corpus
(`rows_with_contact_pii = 0`, verified in `scripts/01_audit_data.py`). Tweet
text is public by construction, but the tweets are republished here only in
aggregate form in `results/`; the raw file is included so preprocessing is
reproducible. If you fork this repo for public distribution, consider keeping
`data/raw/novel_baswedan.xlsx` out of the published tree.

---

## 3. `data/reference/original_pipeline_output.xls`

The output produced by the **original** coursework notebook (the "already
cleaned" file). It is kept so the README can compare the old pipeline's output
with the corrected one. Columns: an index, `Tweet`. 458 rows.

---

## Reproducing

```bash
make setup     # or: pip install -r requirements.txt
make audit     # writes results/tables/audit_*.csv
```

Everything under `results/` is generated; nothing is hand-written.
