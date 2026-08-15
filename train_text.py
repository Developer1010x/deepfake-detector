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
import sys
import json
from pathlib import Path

import joblib
import numpy as np

import text_detect as td
import text_selfsup
from pipeline import CLASSICAL_DEPS, env_versions

MODEL_PATH = Path(__file__).resolve().parent / "models" / "text.joblib"


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
    # Line-buffer stdout: these jobs run for minutes and their progress is
    # useless if it sits in a 4 KB block buffer until the process exits.
    sys.stdout.reconfigure(line_buffering=True)

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

    # Model card: what this artifact was fitted on and with which libraries.
    # pipeline.load_text_model re-checks the `env` block and reports mismatches
    # instead of silently downgrading to the classical combiner.
    card = {
        "corpus": f"text_selfsup pseudo corpus (per_source={args.per_source}, "
                  f"window={args.window})",
        "n_train": len(samples),
        "n_human": int((y == 0).sum()),
        "n_ai": int((y == 1).sum()),
        "signal_names": list(td.SIGNAL_NAMES),
        "in_sample_auc": round(auc, 4),
        "seed": args.seed,
        "env": env_versions(CLASSICAL_DEPS),
    }
    MODEL_PATH.with_suffix(".meta.json").write_text(json.dumps(card, indent=2))
    print(f"saved {MODEL_PATH} (in-sample AUC={auc:.3f})")
    print(f"saved model card -> {MODEL_PATH.with_suffix('.meta.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
