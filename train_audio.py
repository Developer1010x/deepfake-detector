"""Train and persist the learned audio-deepfake detector.

Fits a calibrated gradient-boosted classifier on the 8-d forensic bank over the
self-supervised pseudo corpus (no labelled synthetic speech) and saves it to
``models/audio.joblib``. The CLI and web app load this model for the headline
P(synthetic); without it they fall back to the training-free classical combiner.

    python3 train_audio.py --sources 60 --per-source 5
"""

from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

import joblib
import numpy as np

import audio_detect as ad
import audio_selfsup as asup
from pipeline import CLASSICAL_DEPS, env_versions

MODEL_PATH = Path(__file__).resolve().parent / "models" / "audio.joblib"


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
    ap.add_argument("--sources", type=int, default=60)
    ap.add_argument("--per-source", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    # Line-buffer stdout: these jobs run for minutes and their progress is
    # useless if it sits in a 4 KB block buffer until the process exits.
    sys.stdout.reconfigure(line_buffering=True)

    corpus = asup.synth_corpus(args.sources, seed=args.seed)
    samples = asup.generate(corpus, per_source=args.per_source, seed=args.seed)
    y = np.array([s.label for s in samples])
    X = np.asarray([[ad.all_scores(s.waveform, s.sr)[n] for n in ad.SIGNAL_NAMES]
                    for s in samples], dtype=np.float64)
    print(f"training on {len(samples)} samples ({(y==0).sum()} real / {(y==1).sum()} fake)")

    clf = build_model(args.seed)
    clf.fit(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"clf": clf, "signal_names": ad.SIGNAL_NAMES,
                 "n_train": len(samples), "seed": args.seed}, MODEL_PATH)
    from sklearn.metrics import roc_auc_score
    train_auc = float(roc_auc_score(y, clf.predict_proba(X)[:, 1]))

    # Model card. The corpus line is the important one: this artifact has never
    # heard a human voice - both classes are procedurally synthesised - so its
    # numbers are in-distribution separability, not a deployment figure.
    card = {
        "corpus": f"audio_selfsup.synth_corpus ({args.sources} procedural "
                  f"utterances, per_source={args.per_source}) - NO recorded "
                  f"speech, both classes are synthesised",
        "n_train": len(samples),
        "n_real": int((y == 0).sum()),
        "n_fake": int((y == 1).sum()),
        "signal_names": list(ad.SIGNAL_NAMES),
        "in_sample_auc": round(train_auc, 4),
        "seed": args.seed,
        "env": env_versions(CLASSICAL_DEPS),
    }
    MODEL_PATH.with_suffix(".meta.json").write_text(json.dumps(card, indent=2))
    print(f"saved {MODEL_PATH} (in-sample AUC={train_auc:.3f})")
    print(f"saved model card -> {MODEL_PATH.with_suffix('.meta.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
