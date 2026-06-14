"""Fit a logistic-regression combiner over all 7 signal scores.

With only 3 signals, grid-searching weights was tractable. With 7 signals
and a threshold, the grid has 11^7 ~= 20M points — too many. Instead we
fit a logistic regression by gradient descent on the labeled set and
report train/test accuracy + the learned per-signal weights.

This is still pure numpy — no sklearn, no torch. The only "ML" is the
gradient-descent loop, which is six lines.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from detector import SIGNAL_NAMES, all_scores

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def score_dir(d: Path) -> list[tuple[np.ndarray, str]]:
    rows: list[tuple[np.ndarray, str]] = []
    for p in sorted(d.iterdir()):
        if p.suffix.lower() not in EXTS:
            continue
        try:
            img = Image.open(p)
            s = all_scores(img)
            rows.append((np.array([s[n] for n in SIGNAL_NAMES], dtype=np.float64), p.name))
        except Exception as e:
            print(f"  skip {p.name}: {e}", file=sys.stderr)
    return rows


def stratified_split(rows: list, test_frac: float, rng: np.random.Generator) -> tuple[list, list]:
    idx = np.arange(len(rows))
    rng.shuffle(idx)
    n_test = max(1, int(round(len(rows) * test_frac))) if len(rows) >= 2 else 0
    test_set = set(idx[:n_test].tolist())
    train = [rows[i] for i in range(len(rows)) if i not in test_set]
    test = [rows[i] for i in range(len(rows)) if i in test_set]
    return train, test


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def fit_logistic(
    X: np.ndarray, y: np.ndarray, lr: float = 0.3, epochs: int = 4000, l2: float = 0.05
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Standardize features, then minimize binary cross-entropy + L2.

    Returns (weights, bias, mu, sigma). Predictions on a new point x:
        p = sigmoid(((x - mu) / sigma) @ weights + bias)
    """
    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-8
    Xs = (X - mu) / sigma
    n, k = Xs.shape

    w = np.zeros(k)
    b = 0.0
    for _ in range(epochs):
        p = sigmoid(Xs @ w + b)
        err = p - y
        w -= lr * (Xs.T @ err / n + l2 * w)
        b -= lr * err.mean()
    return w, b, mu, sigma


def predict(X: np.ndarray, w: np.ndarray, b: float, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return sigmoid(((X - mu) / sigma) @ w + b)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--test-frac", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--lr", type=float, default=0.3)
    p.add_argument("--l2", type=float, default=0.05)
    args = p.parse_args()

    root = Path(__file__).parent / "samples"
    real_dir, fake_dir = root / "real", root / "fake"
    if not real_dir.is_dir() or not fake_dir.is_dir():
        print(f"missing {real_dir} or {fake_dir}", file=sys.stderr)
        return 1

    print(f"scoring {real_dir}/ ...")
    real = score_dir(real_dir)
    print(f"scoring {fake_dir}/ ...")
    fake = score_dir(fake_dir)
    if not real or not fake:
        print("need at least 1 image in each folder", file=sys.stderr)
        return 1

    n_real, n_fake = len(real), len(fake)
    print(f"\n{n_real} real, {n_fake} fake\n")

    print(f"{'label':6s} " + " ".join(f"{n:>9s}" for n in SIGNAL_NAMES) + "  name")
    for x, name in real:
        print(f"{'real':6s} " + " ".join(f"{v:9.3f}" for v in x) + f"  {name}")
    for x, name in fake:
        print(f"{'fake':6s} " + " ".join(f"{v:9.3f}" for v in x) + f"  {name}")

    rng = np.random.default_rng(args.seed)
    real_train, real_test = stratified_split(real, args.test_frac, rng)
    fake_train, fake_test = stratified_split(fake, args.test_frac, rng)
    print(f"\nsplit: train = {len(real_train)} real + {len(fake_train)} fake, "
          f"test = {len(real_test)} real + {len(fake_test)} fake "
          f"(test_frac={args.test_frac}, seed={args.seed})")

    if not real_train or not fake_train:
        print("\nERROR: train split empty for one class — add more samples.", file=sys.stderr)
        return 1

    X_train = np.stack([r[0] for r in real_train + fake_train])
    y_train = np.concatenate([np.zeros(len(real_train)), np.ones(len(fake_train))])
    w, b, mu, sigma = fit_logistic(X_train, y_train, lr=args.lr, epochs=args.epochs, l2=args.l2)

    p_train = predict(X_train, w, b, mu, sigma)
    train_acc = float(((p_train > 0.5) == y_train).mean())

    print(f"\nlearned logistic-regression coefficients (on standardized scores):")
    for n, weight in sorted(zip(SIGNAL_NAMES, w), key=lambda kv: -abs(kv[1])):
        sign = "+" if weight >= 0 else "-"
        print(f"  {n:11s} {sign}{abs(weight):.3f}")
    print(f"  bias        {b:+.3f}")
    print(f"  train acc   {train_acc:.3f}")

    if real_test and fake_test:
        X_test = np.stack([r[0] for r in real_test + fake_test])
        y_test = np.concatenate([np.zeros(len(real_test)), np.ones(len(fake_test))])
        p_test = predict(X_test, w, b, mu, sigma)
        test_acc = float(((p_test > 0.5) == y_test).mean())
        tn = int(((p_test <= 0.5) & (y_test == 0)).sum())
        tp = int(((p_test > 0.5) & (y_test == 1)).sum())
        print(f"  test  acc   {test_acc:.3f}  "
              f"({tn}/{len(real_test)} real correct, {tp}/{len(fake_test)} fake correct)")

        print("\ntest predictions:")
        names = [r[1] for r in real_test] + [r[1] for r in fake_test]
        truths = ["real"] * len(real_test) + ["fake"] * len(fake_test)
        for nm, truth, pr in zip(names, truths, p_test):
            verdict = "fake" if pr > 0.5 else "real"
            mark = "" if verdict == truth else "  X"
            print(f"  {truth:6s} {verdict:6s} p_fake={pr:.3f}  {nm}{mark}")
    else:
        print("  test acc    (skipped — not enough samples for both classes in test split)")

    total = n_real + n_fake
    if total < 30:
        print(f"\n  WARNING: only {total} samples — logistic regression overfits with this little data.")
        print("  Aim for 20-30+ per class for trustworthy weights.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
