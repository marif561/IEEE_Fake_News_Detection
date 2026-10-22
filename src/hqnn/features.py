"""Classical feature pipeline: TF-IDF -> PCA (one component per qubit) -> angle scaling.

PCA output is unbounded, so the components are min-max scaled to ``[0, π]`` before they
are used as rotation angles (the paper does not state the scaling; this is the common
choice for angle embedding). Everything is fitted on the training fold only.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler


def build_features(n_qubits: int = 6, max_features: int = 5000, seed: int = 42):
    """TF-IDF + PCA. ``TruncatedSVD`` is PCA for sparse TF-IDF matrices (no densifying)."""
    return make_pipeline(
        TfidfVectorizer(max_features=max_features, stop_words="english", sublinear_tf=True),
        TruncatedSVD(n_components=n_qubits, random_state=seed),
        MinMaxScaler(feature_range=(0.0, math.pi), clip=True),
    )


def fit_transform(pipe, train_texts, other_texts=()):
    x_train = pipe.fit_transform(train_texts).astype(np.float32)
    return x_train, [pipe.transform(t).astype(np.float32) for t in other_texts]
