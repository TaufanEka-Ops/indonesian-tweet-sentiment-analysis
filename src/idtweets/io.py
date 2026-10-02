"""Dataset loading with explicit provenance.

Both datasets are third-party tweet collections used for coursework:

* ``modi_twitter_data.xlsx``  -- Kaggle "Twitter_Data" (Indian election /
  Narendra Modi sentiment), columns ``text``, ``category`` in {-1, 0, 1}.
* ``novel_baswedan.xlsx``     -- Indonesian tweets about the Novel Baswedan
  acid-attack case, columns ``id_str``, ``text``.

The repo ships the raw files under ``data/raw/`` so every script is runnable
out of the box. See ``docs/data_card.md`` for licensing and privacy notes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Repository root = parents of this file's package directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "raw"
REFERENCE_DIR = REPO_ROOT / "data" / "reference"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

MODI_PATH = DATA_DIR / "modi_twitter_data.xlsx"
NOVEL_PATH = DATA_DIR / "novel_baswedan.xlsx"
ORIGINAL_OUTPUT_PATH = REFERENCE_DIR / "original_pipeline_output.xls"

CATEGORY_NAMES = {-1: "negative", 0: "neutral", 1: "positive"}


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Expected dataset at {path}. See docs/data_card.md for provenance."
        )
    return path


def load_modi(path: str | Path | None = None) -> pd.DataFrame:
    """Load the Modi/Indian-election sentiment dataset.

    Columns: ``text`` (str), ``category`` (int in {-1, 0, 1}).
    Rows with a missing or empty text are dropped at load time and the count
    is available on ``df.attrs['dropped_missing_text']``.
    """
    df = pd.read_excel(_require(Path(path) if path else MODI_PATH))
    before = len(df)
    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype("string").str.strip()
    df = df[df["text"].str.len() > 0].copy()
    df["category"] = df["category"].astype(int)
    df = df.reset_index(drop=True)
    df.attrs["dropped_missing_text"] = before - len(df)
    return df


def load_novel(path: str | Path | None = None) -> pd.DataFrame:
    """Load the Novel Baswedan Indonesian tweet dataset (``id_str``, ``text``)."""
    df = pd.read_excel(_require(Path(path) if path else NOVEL_PATH))
    df["id_str"] = df["id_str"].astype("string")
    df["text"] = df["text"].astype("string")
    return df.reset_index(drop=True)


def load_original_output(path: str | Path | None = None) -> pd.DataFrame:
    """Load the output produced by the original coursework notebook."""
    df = pd.read_excel(_require(Path(path) if path else ORIGINAL_OUTPUT_PATH))
    return df


def category_label(value: int) -> str:
    return CATEGORY_NAMES.get(int(value), str(value))
