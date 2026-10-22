"""Loading the three fake-news datasets used in the paper.

Labels are mapped to 1 = fake, 0 = real. Supported layouts:

* **Two files** (``--fake`` / ``--real``), e.g. Kaggle *Fake.csv* and *True.csv* — every row of
  the first file is fake, every row of the second is real.
* **One labelled file** (``--data``) with a label column; values such as ``FAKE``/``REAL``,
  ``fake``/``real`` or ``1``/``0`` are recognised (use ``--fake-value`` if your file differs).

Title and body are joined when both exist. Rows with empty text and duplicates are dropped.
"""
from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

TEXT_CANDIDATES = ("text", "content", "article", "news")
FAKE_WORDS = {"fake", "false", "1"}
REAL_WORDS = {"real", "true", "0"}


def read_table(path: str) -> pd.DataFrame:
    if path.endswith((".json", ".jsonl")):
        return pd.read_json(path, lines=path.endswith(".jsonl"))
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t" if path.endswith(".tsv") else ",")


def _text(df: pd.DataFrame, text_cols: Optional[Sequence[str]]) -> pd.Series:
    cols = list(text_cols) if text_cols else [c for c in ("title", *TEXT_CANDIDATES) if c in df]
    if not cols:
        raise KeyError(f"no text column found; columns are {list(df.columns)} (use --text-cols)")
    return df[cols].fillna("").astype(str).agg(" ".join, axis=1).str.strip()


def to_label(value, fake_value=None) -> int:
    if fake_value is not None:
        return int(str(value).strip().lower() == str(fake_value).strip().lower())
    v = str(value).strip().lower()
    if v in FAKE_WORDS:
        return 1
    if v in REAL_WORDS:
        return 0
    raise ValueError(f"cannot interpret label {value!r}; pass --fake-value")


def load_two_files(fake_path: str, real_path: str, text_cols=None) -> pd.DataFrame:
    fake = pd.DataFrame({"text": _text(read_table(fake_path), text_cols), "label": 1})
    real = pd.DataFrame({"text": _text(read_table(real_path), text_cols), "label": 0})
    return _clean(pd.concat([fake, real], ignore_index=True))


def load_labelled(path: str, label_col: str = "label", text_cols=None, fake_value=None) -> pd.DataFrame:
    df = read_table(path)
    if label_col not in df:
        raise KeyError(f"label column {label_col!r} not in {list(df.columns)}")
    out = pd.DataFrame({"text": _text(df, text_cols), "label": [to_label(v, fake_value) for v in df[label_col]]})
    return _clean(out)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["text"].str.len() > 0].drop_duplicates(subset=["text"])
    return df.sample(frac=1.0, random_state=42).reset_index(drop=True)
