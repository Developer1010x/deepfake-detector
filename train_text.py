"""Train and persist the learned AI-text detector.

Fits a calibrated gradient-boosted classifier on the 8-d forensic bank over the
self-supervised pseudo corpus (no labelled AI text) and saves it to
``models/text.joblib``. The CLI and web app load this model for a well-calibrated
P(AI); without it they fall back to the training-free classical combiner, which
ranks well (high AUC) but is poorly calibrated at the 0.5 threshold.

    python3 train_text.py --per-source 6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

import text_detect as td
import text_selfsup

MODEL_PATH = Path("models/text.joblib")


def build_model(seed: int):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    base = make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=seed))
    return CalibratedClassifierCV(base, method="sigmoid", cv=3)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-source", type=int, default=6)
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    samples = text_selfsup.generate(per_source=args.per_source, seed=args.seed,
                                    window=args.window)
    y = np.array([s.label for s in samples])
    X = np.asarray([[td.all_scores(s.text)[n] for n in td.SIGNAL_NAMES]
                    for s in samples], dtype=np.float64)
    print(f"training on {len(samples)} samples ({(y==0).sum()} human / {(y==1).sum()} ai)")

    clf = build_model(args.seed)
    clf.fit(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"clf": clf, "signal_names": td.SIGNAL_NAMES,
                 "n_train": len(samples), "seed": args.seed}, MODEL_PATH)
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y, clf.predict_proba(X)[:, 1]))
    print(f"saved {MODEL_PATH} (in-sample AUC={auc:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
