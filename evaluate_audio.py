"""Reproducible evaluation for the label-free audio-deepfake detector.

The audio counterpart of :mod:`evaluate`. Same leakage-free protocol:

  * **Leave-source-out GroupKFold** - every view of an utterance falls in a
    single fold, so no source leaks between train and test.
  * **Ablation** - the training-free fixed-weight combiner (:func:`audio_detect.detect`)
    vs. a learned classifier on the 8-d forensic bank.
  * **Threshold-free + thresholded metrics** reused from :mod:`evaluate`.
  * **Per-artifact recall** by which synthesis transform produced each pseudo-fake.

Outputs land in ``paper/``:
    paper/results_audio.json
    paper/results_audio.md
    paper/figures/audio_roc.png, audio_scores.png, audio_per_artifact.png

Example
-------
    python3 evaluate_audio.py --sources 40 --per-source 4 --folds 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import audio_detect as ad
import audio_selfsup as asup
from evaluate import (
    FIG_DIR,
    PAPER_DIR,
    compute_metrics,
    plot_per_artifact,
    plot_roc,
    plot_scores,
)


def handcrafted_matrix(samples: list[asup.PseudoAudioSample]) -> np.ndarray:
    """8-d forensic feature vector per sample (audio_detect signal bank)."""
    rows = []
    for s in samples:
        sc = ad.all_scores(s.waveform, s.sr)
        rows.append([sc[name] for name in ad.SIGNAL_NAMES])
    return np.asarray(rows, dtype=np.float64)


def classical_scores(samples: list[asup.PseudoAudioSample]) -> np.ndarray:
    return np.asarray([ad.detect(s.waveform, s.sr)["combined"] for s in samples],
                      dtype=np.float64)


def grouped_oof(X, y, groups, folds: int, seed: int) -> tuple[np.ndarray, list[float]]:
    """Out-of-fold calibrated P(fake) + per-fold AUC for the learned model."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    oof = np.full(len(y), np.nan)
    fold_aucs: list[float] = []
    gkf = GroupKFold(n_splits=folds)
    for tr, te in gkf.split(X, y, groups):
        base = make_pipeline(
            StandardScaler(),
            GradientBoostingClassifier(random_state=seed),
        )
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
        if len(np.unique(y[te])) == 2:
            fold_aucs.append(float(roc_auc_score(y[te], oof[te])))
    return oof, fold_aucs


def write_markdown(results: dict, path: Path) -> None:
    c = results["corpus"]
    lines = ["# Audio-deepfake evaluation results", "",
             f"_Protocol: {results['protocol']}_  ",
             f"_Corpus: {c['n_sources']} synthetic utterances -> {c['n_samples']} "
             f"pseudo-samples ({c['n_real']} real / {c['n_fake']} fake)_", "",
             "## Ablation (leave-source-out, out-of-fold)", "",
             "| Model | AUC | AP | Acc | Precision | Recall | F1 | Brier |",
             "|---|---|---|---|---|---|---|---|"]
    for name, m in results["models"].items():
        fold = results["fold_auc"].get(name)
        auc = f"{m['auc']:.3f}" + (f" ±{np.std(fold):.3f}" if fold else "")
        lines.append(f"| {name} | {auc} | {m['ap']:.3f} | {m['accuracy']:.3f} | "
                     f"{m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | "
                     f"{m['brier']:.3f} |")
    lines += ["", "## Per-artifact recall (learned)", "", "| Artifact | Recall | n |",
              "|---|---|---|"]
    for t, (r, n) in sorted(results["per_artifact"].items(), key=lambda kv: kv[1][0]):
        lines.append(f"| {t} | {r:.3f} | {n} |")
    lines += ["", "## Figures", "",
              "![audio ROC](figures/audio_roc.png)",
              "![audio score distribution](figures/audio_scores.png)",
              "![audio per-artifact recall](figures/audio_per_artifact.png)"]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", type=int, default=40, help="synthetic utterances")
    ap.add_argument("--per-source", type=int, default=4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    # Line-buffer stdout: these jobs run for minutes and their progress is
    # useless if it sits in a 4 KB block buffer until the process exits.
    sys.stdout.reconfigure(line_buffering=True)

    corpus = asup.synth_corpus(args.sources, seed=args.seed)
    samples = asup.generate(corpus, per_source=args.per_source, seed=args.seed)
    y = np.array([s.label for s in samples])
    groups = np.array([s.source for s in samples])
    transforms = [s.transforms for s in samples]
    n_groups = len(np.unique(groups))
    folds = max(2, min(args.folds, n_groups))
    print(f"corpus {len(corpus)} utterances -> {len(samples)} samples "
          f"({(y==0).sum()} real / {(y==1).sum()} fake); GroupKFold folds={folds}")

    t0 = time.time()
    X = handcrafted_matrix(samples)
    print(f"features in {time.time()-t0:.1f}s")

    results = {
        "protocol": f"leave-source-out {folds}-fold GroupKFold (gboost, seed={args.seed})",
        "corpus": {"n_sources": len(corpus), "n_samples": len(samples),
                   "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum())},
        "models": {}, "fold_auc": {},
    }
    preds: dict[str, np.ndarray] = {}

    p_classical = classical_scores(samples)
    results["models"]["classical (fixed weights)"] = compute_metrics(y, p_classical)
    preds["classical"] = p_classical

    oof, fold_aucs = grouped_oof(X, y, groups, folds, args.seed)
    results["models"]["learned (handcrafted)"] = compute_metrics(y, oof)
    results["fold_auc"]["learned (handcrafted)"] = fold_aucs
    preds["learned"] = oof
    for name, m in results["models"].items():
        print(f"  {name:26s} AUC={m['auc']:.3f} AP={m['ap']:.3f} F1={m['f1']:.3f}")

    # per-artifact recall (learned model)
    yhat = (oof >= 0.5).astype(int)
    hit, tot = defaultdict(int), defaultdict(int)
    for i, lab in enumerate(y):
        if lab == 1:
            for t in transforms[i]:
                tot[t] += 1
                hit[t] += int(yhat[i] == 1)
    results["per_artifact"] = {t: (hit[t] / tot[t], tot[t]) for t in tot}

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_roc(y, preds, FIG_DIR / "audio_roc.png")
    plot_scores(y, oof, FIG_DIR / "audio_scores.png")
    plot_per_artifact({t: r for t, (r, _) in results["per_artifact"].items()},
                      FIG_DIR / "audio_per_artifact.png")

    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    (PAPER_DIR / "results_audio.json").write_text(json.dumps(results, indent=2))
    write_markdown(results, PAPER_DIR / "results_audio.md")
    print(f"\nwrote {PAPER_DIR/'results_audio.json'}, {PAPER_DIR/'results_audio.md'}, "
          f"+ 3 figures to {FIG_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
