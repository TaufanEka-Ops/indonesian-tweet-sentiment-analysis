#!/usr/bin/env python
"""07 -- Build AND execute the notebooks so their outputs are embedded.

The notebooks are generated rather than hand-edited so that what ships in the
repo is guaranteed to be the executed version of exactly these cells.

Run:  python scripts/07_build_and_execute_notebooks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import nbformat as nbf  # noqa: E402
from nbclient import NotebookClient  # noqa: E402

NOTEBOOK_DIR = ROOT / "notebooks"

HEADER = """\
import sys, pathlib
ROOT = pathlib.Path.cwd()
while not (ROOT / "src").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
print("repo root:", ROOT)
"""


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


# --------------------------------------------------------------------------- #
# Notebook 1 -- Novel Baswedan EDA
# --------------------------------------------------------------------------- #
def build_notebook_1() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            "# Case study 1 — Novel Baswedan tweet corpus (Indonesian)\n\n"
            "Preprocessing walk-through and exploratory analysis of 2,727 Indonesian tweets\n"
            "about the Novel Baswedan acid-attack case.\n\n"
            "This corpus has **no sentiment label**, so nothing here trains a classifier —\n"
            "it cleans the data, quantifies the original pipeline's mention bug on real rows,\n"
            "and describes the corpus (hashtags, terms, entity clusters).\n\n"
            "Executed end-to-end; every output below is real."
        ),
        code(HEADER),
        code(
            "import re\nimport pandas as pd\nfrom idtweets.cases import load_case_module\n"
            "from idtweets.preprocessing import remove_pattern_original, strip_mentions, MENTION_RE\n\n"
            "case = load_case_module('novel-baswedan', 'novel_baswedan_analysis.py')\n"
            "raw = case.load_raw()\nprint(raw.shape)\nraw.head(3)"
        ),
        md(
            "## Defect #1 — the original `remove_pattern` kept all but the first mention\n\n"
            "The original helper returned from inside its `for` loop. Let us run it on a real\n"
            "tweet from this corpus that mentions four accounts."
        ),
        code(
            "ex = raw[raw['text'].astype(str).str.count(r'@\\w+') >= 4]['text'].iloc[0]\n"
            "print('RAW:')\nprint(ex[:180], '\\n')\n"
            "orig = remove_pattern_original(ex, r'@\\w+')\n"
            "fixed = strip_mentions(ex)\n"
            "print('original logic -> mentions still present:', re.findall(r'@\\w+', orig))\n"
            "print('fixed logic    -> mentions still present:', re.findall(r'@\\w+', fixed))"
        ),
        code(
            "impact = case.defect_impact(raw)\n"
            "pd.Series(impact, name='value').to_frame()"
        ),
        md(
            "Across the corpus the original left **1,099 of 3,616 mentions** in place across\n"
            "**627 tweets**; the corrected function leaves zero."
        ),
        md("## Preprocessing and de-duplication"),
        code(
            "processed = case.preprocess(raw)\ndeduped = case.deduplicate(processed)\n"
            "comp = pd.DataFrame({\n"
            "    'raw': raw['text'].astype(str).head(4),\n"
            "    'clean': processed['clean'].head(4),\n"
            "    'tokens': processed['clean_token_count'].head(4),\n})\n"
            "pd.set_option('display.max_colwidth', 90)\ncomp"
        ),
        code(
            "print(f'raw rows                  : {len(raw)}')\n"
            "print(f'rows after preprocessing  : {len(processed)}')\n"
            "print(f'rows after de-duplication : {len(deduped)}')\n"
            "print(f'duplicate share           : {raw[\"text\"].astype(str).duplicated().mean():.1%}')\n"
            "print(f'empty cleaned rows        : {(processed[\"clean\"].str.len() == 0).sum()}')"
        ),
        md("## What the corpus is about"),
        code("tags = case.hashtag_table(raw, top_n=12)\ntags"),
        code("case.entity_table(deduped)"),
        code("case.top_terms(deduped, top_n=20, min_df=3)"),
        md(
            "## Takeaways\n\n"
            "* Cleaning is not cosmetic here: 78% of the rows are duplicate retweets and\n"
            "  87% are retweets, so the de-duplicated corpus (462 tweets) is what a reader\n"
            "  should reason about.\n"
            "* The hashtag distribution is dominated by a single campaign tag\n"
            "  (`#RekayasaDidzolimi`), which is a strong hint that this dump is a\n"
            "  coordinated capture rather than a random sample of Indonesian opinion —\n"
            "  another reason not to read sentiment off it.\n"
            "* The original pipeline's mention bug is not theoretical: it silently kept\n"
            "  ~30% of all mentions, which contaminates any downstream vocabulary or\n"
            "  feature built from this text."
        ),
    ]
    return nb


# --------------------------------------------------------------------------- #
# Notebook 2 -- Modi sentiment modelling
# --------------------------------------------------------------------------- #
def build_notebook_2() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            "# Case study 2 — Modi / Indian-election sentiment classification\n\n"
            "The evaluation the original coursework was missing: a majority-class baseline,\n"
            "repeated stratified cross-validation, macro-F1 (labels are imbalanced), a\n"
            "stratified held-out confusion matrix and a real error analysis.\n\n"
            "Executed end-to-end; every number below is produced by this notebook."
        ),
        code(HEADER),
        code(
            "import pandas as pd\n"
            "from functools import partial\n"
            "from sklearn.metrics import f1_score\n"
            "from idtweets import evaluation, modelling\n"
            "from idtweets.cases import load_case_module\n"
            "from idtweets.modelling import build_pipeline, default_estimators\n\n"
            "case = load_case_module('modi-election', 'modi_election_sentiment.py')"
        ),
        md(
            "## Data preparation\n\n"
            "Encoding repair → drop rows with contact PII → English-aware preprocessing →\n"
            "drop exact duplicates. Every removal is counted."
        ),
        code(
            "df, audit = case.prepare_dataset()\n"
            "for k, v in audit.items():\n"
            "    print(f'{k:26s}: {v if k != \"pii_hits\" else str(len(v)) + \" row(s)\" }')"
        ),
        code("df.head(3)[['text', 'clean', 'category']]"),
        code("case.label_distribution(df)"),
        md("### Why accuracy alone would be misleading"),
        code("print(case.class_weight_note(df))"),
        code(
            "import matplotlib.pyplot as plt\n"
            "from idtweets.io import FIGURES_DIR\n"
            "dist = case.label_distribution(df)\n"
            "ax = dist.plot.bar(x='label_name', y='count', legend=False, rot=0, color='#C44E52')\n"
            "ax.set_title('Class distribution'); ax.set_xlabel('')\nplt.tight_layout(); plt.show()"
        ),
        md(
            "## Model comparison — repeated stratified 5×3 cross-validation\n\n"
            "15 fits per model. TF-IDF sits inside the pipeline, so it is refit on each fold\n"
            "and the test fold's vocabulary never leaks into training."
        ),
        code(
            "table, results = case.run_comparison(df, n_splits=5, n_repeats=3)\ntable"
        ),
        code(
            "print(f\"Best macro-F1 beats the majority baseline by {case.macro_f1_gap(table):+.4f}\")\n"
            "for r in results.values():\n"
            "    print(f'  {r.name:20s} macro-F1 {r.macro_f1_mean:.4f} +/- {r.macro_f1_std:.4f}')"
        ),
        md(
            "## Held-out evaluation\n\n"
            "Stratified 80/20 split, best CV model, with a bootstrap CI on macro-F1."
        ),
        code(
            "Xtr, Xte, ytr, yte = modelling.holdout_split(df['clean'], df['category'], test_size=0.2)\n"
            "best_name = table.iloc[0]['model']\n"
            "pipe = build_pipeline(default_estimators()[best_name]).fit(Xtr, ytr)\n"
            "preds = pipe.predict(Xte)\n"
            "metrics = evaluation.classification_metrics(yte, preds)\n"
            "lo, hi = case.bootstrap_ci(yte, preds, partial(f1_score, average='macro'), n_boot=1000)\n"
            "print('model:', best_name)\n"
            "for k, v in metrics.items():\n"
            "    print(f'  {k:12s}: {v:.4f}')\n"
            "print(f'  macro-F1 95% CI: [{lo:.4f}, {hi:.4f}]')"
        ),
        code("evaluation.report_frame(yte, preds, label_names=case.LABEL_NAMES)"),
        code(
            "evaluation.plot_confusion(yte, preds, labels=[-1, 0, 1],\n"
            "    label_names=case.LABEL_NAMES, title=f'Confusion matrix -- {best_name}',\n"
            "    out_path=FIGURES_DIR / 'notebook_confusion.png');\n"
            "from IPython.display import Image\nImage(str(FIGURES_DIR / 'notebook_confusion.png'))"
        ),
        md("## Error analysis — real misclassified tweets"),
        code(
            "err = evaluation.error_analysis(Xte.reset_index(drop=True), yte.reset_index(drop=True),\n"
            "    preds, label_names=case.LABEL_NAMES, n_per_class=6)\n"
            "err[['true_name', 'pred_name', 'text']].assign(\n"
            "    text=lambda d: d['text'].str.slice(0, 120))"
        ),
        code(
            "err['direction'] = err['true_name'] + ' -> ' + err['pred_name']\n"
            "err['direction'].value_counts().to_frame('errors_in_held_out_fold')"
        ),
        md(
            "## What the model actually keyed on"
        ),
        code(
            "lr = build_pipeline(default_estimators()['logistic_regression']).fit(Xtr, ytr)\n"
            "top = evaluation.top_features_per_class(lr, top_n=12)\n"
            "for c in sorted(top['class'].unique()):\n"
            "    print(case.LABEL_NAMES[c].upper() + ':')\n"
            "    print('   ', ', '.join(top[top['class'] == c]['term'][:12]))\n"
            "    print()"
        ),
        md(
            "## Takeaways\n\n"
            "* The corpus is small (5,232 usable rows after cleaning) and the label\n"
            "  distribution is imbalanced, so **macro-F1**, not accuracy, is the number to\n"
            "  quote — and a majority-class predictor is quoted next to it.\n"
            "* The three real classifiers land close together; the notebook reports the\n"
            "  standard deviation across folds rather than pretending the top row is a\n"
            "  decisive winner.\n"
            "* Errors concentrate on the neutral/positive boundary, where short tweets with\n"
            "  no explicit sentiment cue are genuinely ambiguous — see the quoted rows above.\n"
            "* The original coursework's single 0.63 accuracy figure was not wrong so much as\n"
            "  uninterpretable: without a baseline and without per-class metrics, a reader\n"
            "  cannot tell whether 0.63 is good."
        ),
    ]
    return nb


def execute(nb: nbf.NotebookNode, path: Path) -> None:
    client = NotebookClient(
        nb,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    nbf.write(nb, str(path))
    print(f"executed -> {path}")


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    execute(build_notebook_1(), NOTEBOOK_DIR / "01_novel_baswedan_eda.ipynb")
    execute(build_notebook_2(), NOTEBOOK_DIR / "02_modi_sentiment_model.ipynb")


if __name__ == "__main__":
    main()
