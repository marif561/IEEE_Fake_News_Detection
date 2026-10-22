"""Train and evaluate the HQNN with 5-fold cross-validation (paper Sect. III).

Per fold: fit TF-IDF -> PCA(6) -> [0, π] scaling on the training part, train the HQNN for
20 epochs (batch 32, Adam lr 1e-3, cross-entropy), then report accuracy, precision, recall
and F1 on the held-out part (fake = positive class). Results are averaged over folds
(mean ± std), as in Table I.

``--baselines`` additionally fits logistic regression on (a) the same 6 PCA features and
(b) the full TF-IDF matrix, as a classical reference point. These are not part of the paper.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

from .data import load_labelled, load_two_files
from .features import build_features, fit_transform
from .quantum import HQNN

METRICS = ("accuracy", "precision", "recall", "f1")


def scores(y, pred):
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", pos_label=1, zero_division=0)
    return {"accuracy": accuracy_score(y, pred), "precision": p, "recall": r, "f1": f}


@torch.no_grad()
def predict(model, x, bs=256):
    model.eval()
    return torch.cat([model(x[i:i + bs]).argmax(-1) for i in range(0, len(x), bs)]).numpy()


def train_fold(x_tr, y_tr, x_te, y_te, args, log):
    torch.manual_seed(args.seed)
    model = HQNN(args.qubits, args.layers, args.backend)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    xt, yt, xe = torch.from_numpy(x_tr), torch.from_numpy(y_tr).long(), torch.from_numpy(x_te)
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(xt))
        total = 0.0
        for i in range(0, len(xt), args.batch_size):
            idx = perm[i:i + args.batch_size]
            opt.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        acc = accuracy_score(y_te, predict(model, xe))
        log.append({"epoch": epoch, "loss": total / len(xt), "test_accuracy": acc})
        print(f"    epoch {epoch:>2}  loss {total / len(xt):.4f}  test acc {acc:.4f}")
    return scores(y_te, predict(model, xe))


def summarise(folds):
    return {m: {"mean": float(np.mean([f[m] for f in folds])), "std": float(np.std([f[m] for f in folds]))}
            for m in METRICS}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="one labelled file (CSV / JSON / parquet)")
    src.add_argument("--fake", help="file with fake articles only (use with --real)")
    ap.add_argument("--real", help="file with real articles only")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--fake-value", default=None, help="label value that means fake, if not fake/real/1/0")
    ap.add_argument("--text-cols", nargs="+", default=None)
    ap.add_argument("--name", default="dataset", help="name used for the output folder")
    ap.add_argument("--qubits", type=int, default=6)
    ap.add_argument("--layers", type=int, default=2, help="entangling layers (not stated in the paper)")
    ap.add_argument("--backend", choices=["torch", "pennylane"], default="torch")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-features", type=int, default=5000)
    ap.add_argument("--max-samples", type=int, default=None, help="random subsample for quick runs")
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args(argv)

    if args.fake and not args.real:
        ap.error("--fake needs --real")
    df = (load_two_files(args.fake, args.real, args.text_cols) if args.fake
          else load_labelled(args.data, args.label_col, args.text_cols, args.fake_value))
    if args.max_samples and len(df) > args.max_samples:
        df = df.groupby("label", group_keys=False).apply(
            lambda g: g.sample(min(len(g), args.max_samples // 2), random_state=args.seed)).reset_index(drop=True)
    texts, y = df["text"].to_numpy(), df["label"].to_numpy()
    print(f"{args.name}: {len(df):,} articles ({int(y.sum()):,} fake / {int((1 - y).sum()):,} real) · "
          f"{args.qubits} qubits · {args.layers} layers · backend={args.backend}")

    folds, curves, base = [], [], {"lr_pca": [], "lr_tfidf": []}
    t0 = time.time()
    for k, (tr, te) in enumerate(StratifiedKFold(args.folds, shuffle=True, random_state=args.seed).split(texts, y), 1):
        print(f"  fold {k}/{args.folds}")
        pipe = build_features(args.qubits, args.max_features, args.seed)
        x_tr, (x_te,) = fit_transform(pipe, texts[tr], [texts[te]])
        log = []
        folds.append(train_fold(x_tr, y[tr], x_te, y[te], args, log))
        curves.append(log)
        if args.baselines:
            base["lr_pca"].append(scores(y[te], LogisticRegression(max_iter=1000).fit(x_tr, y[tr]).predict(x_te)))
            vec = TfidfVectorizer(max_features=args.max_features, stop_words="english", sublinear_tf=True)
            lr = LogisticRegression(max_iter=1000).fit(vec.fit_transform(texts[tr]), y[tr])
            base["lr_tfidf"].append(scores(y[te], lr.predict(vec.transform(texts[te]))))

    res = {"dataset": args.name, "n": len(df), "config": vars(args), "hqnn": summarise(folds),
           "folds": folds, "curves": curves, "seconds": round(time.time() - t0, 1)}
    if args.baselines:
        res["baselines"] = {k: summarise(v) for k, v in base.items()}
    out = Path(args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print("\n" + "  ".join(f"{m} {100 * res['hqnn'][m]['mean']:.2f} ± {100 * res['hqnn'][m]['std']:.2f}" for m in METRICS))
    for name, s in res.get("baselines", {}).items():
        print(f"[{name}] " + "  ".join(f"{m} {100 * s[m]['mean']:.2f}" for m in METRICS))


if __name__ == "__main__":
    main()
