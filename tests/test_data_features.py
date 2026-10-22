import math
from pathlib import Path

import numpy as np
import pytest

from hqnn.data import load_labelled, load_two_files, to_label
from hqnn.features import build_features, fit_transform

S = Path(__file__).resolve().parents[1] / "data" / "sample"


def test_two_file_and_labelled_layouts_agree():
    a = load_two_files(str(S / "fake.csv"), str(S / "real.csv"))
    b = load_labelled(str(S / "labelled.csv"))
    assert len(a) == len(b) == 60
    assert a["label"].sum() == b["label"].sum() == 30
    assert a["text"].str.contains("shocking", case=False).any()  # title and body are joined


def test_label_mapping():
    assert to_label("FAKE") == 1 and to_label("real") == 0 and to_label(1) == 1
    assert to_label("satire", fake_value="satire") == 1
    with pytest.raises(ValueError):
        to_label("maybe")


def test_features_are_six_angles_in_range():
    df = load_labelled(str(S / "labelled.csv"))
    pipe = build_features(n_qubits=6)
    x_tr, (x_te,) = fit_transform(pipe, df["text"][:40], [df["text"][40:]])
    assert x_tr.shape == (40, 6) and x_te.shape == (20, 6) and x_tr.dtype == np.float32
    assert x_tr.min() >= 0 and x_tr.max() <= math.pi + 1e-6
    assert x_te.min() >= 0 and x_te.max() <= math.pi + 1e-6  # clipped for unseen data
