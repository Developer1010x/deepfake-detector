"""Reproducible evaluation suite for the AI-generated-text detector.

Mirrors ``evaluate.py`` (image track) for the TEXT modality, under the same
leakage-free protocol:

  * **Leave-source-out GroupKFold CV** - folds are split by *source* (the
    sentence-window id from ``text_selfsup.build_sources``), so no view of a
    source ever appears in both train and test. This is the only honest
    protocol for the self-supervised setting.
  * **Ablation study** - the classical fixed-weight combiner (training-free
    baseline, ``text_detect.detect``) vs. a learned classifier on the same
    hand-crafted feature bank, evaluated on identical folds.
  * **Threshold-free + thresholded metrics** - ROC-AUC and average precision
    (PR-AUC) plus accuracy / precision / recall / F1 / Brier at t=0.5 (same
    definitions as ``evaluate.compute_metrics``).
  * **Per-artifact recall** - detection rate broken down by which LLM-style
    injection transform produced each pseudo-AI sample.

Outputs land in ``paper/``:
    paper/results_text.json     machine-readable metrics
    paper/results_text.md       markdown tables (same style as results.md)

Examples
--------
    python3 evaluate_text.py
    python3 evaluate_text.py --folds 5 --per-source 4 --seed 0
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import text_detect
import text_selfsup

PAPER_DIR = Path("paper")


# --------------------------------------------------------------------------- #
# metrics  (identical definitions to evaluate.compute_metrics)
# --------------------------------------------------------------------------- #


def compute_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    yhat = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "ap": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "accuracy": float(accuracy_score(y, yhat)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "brier": float(brier_score_loss(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


# --------------------------------------------------------------------------- #
# feature extraction
# --------------------------------------------------------------------------- #


def handcrafted_matrix(texts: list[str]) -> np.ndarray:
    """(N, 8) interpretable forensic feature matrix from text_detect signals."""
    if not texts:
        return np.empty((0, len(text_detect.SIGNAL_NAMES)))
    rows = []
    for t in texts:
        s = text_detect.all_scores(t)
        rows.append([s[n] for n in text_detect.SIGNAL_NAMES])
    return np.asarray(rows, dtype=np.float64)


def classical_scores(texts: list[str]) -> np.ndarray:
    """Training-free baseline: the fixed-weight combiner from text_detect."""
    return np.asarray(
        [text_detect.detect(t)["combined"] for t in texts], dtype=np.float64
    )


# --------------------------------------------------------------------------- #
# cross-validated out-of-fold predictions
# --------------------------------------------------------------------------- #


def _build_classifier(seed: int):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    clf = GradientBoostingClassifier(random_state=seed)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    return pipe


def grouped_oof(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int, seed: int
) -> tuple[np.ndarray, list[float]]:
    """Out-of-fold P(ai) and per-fold AUC for the learned model (GroupKFold).

    The pipeline (StandardScaler + GradientBoostingClassifier) is wrapped in
    ``CalibratedClassifierCV`` (sigmoid) when each train class has enough members
    for inner CV folds, matching fusion.py's calibration policy.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    oof = np.full(len(y), np.nan)
    fold_aucs: list[float] = []
    gkf = GroupKFold(n_splits=folds)
    for tr, te in gkf.split(X, y, groups):
        pipe = _build_classifier(seed)
        ytr = y[tr]
        min_class = int(min(np.bincount(ytr, minlength=2)))
        if min_class >= 4:
            cv = min(3, min_class)
            est = CalibratedClassifierCV(pipe, method="sigmoid", cv=cv)
        else:
            est = pipe
        est.fit(X[tr], ytr)
        oof[te] = est.predict_proba(X[te])[:, 1]
        if len(np.unique(y[te])) == 2:
            fold_aucs.append(float(roc_auc_score(y[te], oof[te])))
    return oof, fold_aucs


# --------------------------------------------------------------------------- #
# markdown report  (same style as evaluate.write_markdown)
# --------------------------------------------------------------------------- #


def _read_text_dir(d: Path) -> list[str]:
    return [p.read_text(errors="ignore") for p in sorted(d.glob("*.txt"))]


def evaluate_external(real_dir: Path, fake_dir: Path, X, y, seed: int) -> dict | None:
    """Train on the full pseudo corpus, test on a genuinely-labelled set
    (real_dir = human/label 0, fake_dir = AI/label 1). Reports both the learned
    model and the training-free classical combiner on the same external texts."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import roc_auc_score

    real = _read_text_dir(real_dir)
    fake = _read_text_dir(fake_dir)
    if not real or not fake:
        print(f"external: missing texts in {real_dir} / {fake_dir}, skipping")
        return None
    test_texts = real + fake
    test_y = np.array([0] * len(real) + [1] * len(fake))
    Xt = handcrafted_matrix(test_texts)

    est = CalibratedClassifierCV(_build_classifier(seed), method="sigmoid", cv=3)
    est.fit(X, y)
    p_learned = est.predict_proba(Xt)[:, 1]
    p_classical = classical_scores(test_texts)

    learned = compute_metrics(test_y, p_learned)
    classical = compute_metrics(test_y, p_classical)
    print(f"external HC3: n={len(test_y)} ({len(real)} human / {len(fake)} ai)  "
          f"learned AUC={learned['auc']:.3f}  classical AUC={classical['auc']:.3f}")
    return {"source": f"{real_dir}/{fake_dir}", "n_human": len(real), "n_ai": len(fake),
            "learned": learned, "classical": classical}


def write_markdown(results: dict, path: Path) -> None:
    lines = ["# AI-text detection — evaluation results", "",
             f"_Protocol: {results['protocol']}_  ",
             f"_Corpus: {results['corpus']['n_paragraphs']} paragraphs "
             f"-> {results['corpus']['n_sources']} sources "
             f"-> {results['corpus']['n_samples']} pseudo-samples "
             f"({results['corpus']['n_human']} human / {results['corpus']['n_ai']} ai)_",
             "", "## Ablation (cross-validated out-of-fold)", "",
             "| Model | AUC | AP | Acc | Precision | Recall | F1 | Brier |",
             "|---|---|---|---|---|---|---|---|"]
    for name, m in results["models"].items():
        fold = results["fold_auc"].get(name)
        auc_cell = f"{m['auc']:.3f}"
        if fold:
            auc_cell += f" ±{np.std(fold):.3f}"
        lines.append(
            f"| {name} | {auc_cell} | {m['ap']:.3f} | {m['accuracy']:.3f} | "
            f"{m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | "
            f"{m['brier']:.3f} |"
        )
    if results.get("per_artifact"):
        lines += ["", "## Per-artifact recall (learned model)", "",
                  "| Artifact | Recall | n |", "|---|---|---|"]
        for t, (r, n) in sorted(results["per_artifact"].items(), key=lambda kv: kv[1][0]):
            lines.append(f"| {t} | {r:.3f} | {n} |")
    ext = results.get("external")
    if ext:
        lines += ["", "## External validation — real human vs ChatGPT (HC3)", "",
                  f"_n = {ext['n_human']} human / {ext['n_ai']} ai; learned model "
                  f"trained on the pseudo corpus only_", "",
                  "| Model | AUC | AP | Acc | F1 | Brier |", "|---|---|---|---|---|---|"]
        for nm in ("learned", "classical"):
            m = ext[nm]
            lines.append(f"| {nm} | {m['auc']:.3f} | {m['ap']:.3f} | {m['accuracy']:.3f} "
                         f"| {m['f1']:.3f} | {m['brier']:.3f} |")
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--per-source", type=int, default=4)
    ap.add_argument("--window", type=int, default=2, help="sentences per source")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--real", type=Path, default=None, help="labelled human dir (external)")
    ap.add_argument("--fake", type=Path, default=None, help="labelled AI dir (external)")
    args = ap.parse_args()

    # ---- corpus -> sources -> pseudo dataset ----
    samples = text_selfsup.generate(
        per_source=args.per_source, seed=args.seed, window=args.window
    )
    texts = [s.text for s in samples]
    y = np.array([s.label for s in samples])
    groups = np.array([s.source for s in samples])
    transforms = [s.transforms for s in samples]
    n_groups = len(np.unique(groups))
    folds = max(2, min(args.folds, n_groups))           # guard n_groups < folds
    n_para = len(text_selfsup.HUMAN_CORPUS)
    print(f"corpus {n_para} paragraphs -> {n_groups} sources -> {len(samples)} samples "
          f"({(y==0).sum()} human / {(y==1).sum()} ai); GroupKFold folds={folds}")

    # ---- features (extract once) ----
    t0 = time.time()
    X = handcrafted_matrix(texts)
    print(f"features in {time.time()-t0:.2f}s ({X.shape[1]} signals)")

    results = {
        "protocol": f"leave-source-out {folds}-fold GroupKFold "
                    f"(classifier=GradientBoosting+sigmoid-calib, seed={args.seed})",
        "corpus": {"n_paragraphs": n_para, "n_sources": n_groups,
                   "n_samples": len(samples), "n_human": int((y == 0).sum()),
                   "n_ai": int((y == 1).sum())},
        "models": {}, "fold_auc": {},
    }

    # ---- (a) training-free classical baseline ----
    p_classical = classical_scores(texts)
    results["models"]["classical (fixed weights)"] = compute_metrics(y, p_classical)
    print(f"  classical (fixed weights)  AUC="
          f"{results['models']['classical (fixed weights)']['auc']:.3f} "
          f"AP={results['models']['classical (fixed weights)']['ap']:.3f} "
          f"F1={results['models']['classical (fixed weights)']['f1']:.3f}")

    # ---- (b) learned model on hand-crafted bank (out-of-fold) ----
    oof, fold_aucs = grouped_oof(X, y, groups, folds, args.seed)
    results["models"]["learned (handcrafted)"] = compute_metrics(y, oof)
    results["fold_auc"]["learned (handcrafted)"] = fold_aucs
    print(f"  learned (handcrafted)      AUC="
          f"{results['models']['learned (handcrafted)']['auc']:.3f} "
          f"AP={results['models']['learned (handcrafted)']['ap']:.3f} "
          f"F1={results['models']['learned (handcrafted)']['f1']:.3f}")

    # ---- per-artifact recall (learned model, t=0.5) ----
    yhat = (oof >= 0.5).astype(int)
    hit, tot = defaultdict(int), defaultdict(int)
    for i, lab in enumerate(y):
        if lab == 1:
            for t in transforms[i]:
                tot[t] += 1
                hit[t] += int(yhat[i] == 1)
    per_artifact = {t: (hit[t] / tot[t], tot[t]) for t in tot}
    results["per_artifact"] = per_artifact

    # ---- external validation on a genuinely-labelled set (optional) ----
    if args.real and args.fake:
        ext = evaluate_external(args.real, args.fake, X, y, args.seed)
        if ext:
            results["external"] = ext

    # ---- write reports ----
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    (PAPER_DIR / "results_text.json").write_text(json.dumps(results, indent=2))
    write_markdown(results, PAPER_DIR / "results_text.md")
    print(f"\nwrote {PAPER_DIR/'results_text.json'} and {PAPER_DIR/'results_text.md'}")

    print("\nper-artifact recall (learned):")
    for t, (r, n) in sorted(per_artifact.items(), key=lambda kv: kv[1][0]):
        print(f"  {t:24s} {r:.3f}  (n={n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
