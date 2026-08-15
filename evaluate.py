"""Rigorous, reproducible evaluation suite for the hybrid detector.

Produces the numbers and figures a paper needs, under a leakage-free protocol:

  * **Grouped K-fold cross-validation** - folds are split by *source scene*
    (``GroupKFold`` on the patch/file id), so no view of a source ever appears
    in both train and test. This is the only honest protocol for the
    self-supervised setting.
  * **Ablation study** - the classical fixed-weight combiner (training-free
    baseline) vs. learned ``handcrafted`` / ``deep`` / ``fusion`` models, all on
    identical folds.
  * **Threshold-free + thresholded metrics** - ROC-AUC and average precision
    (PR-AUC) plus accuracy / precision / recall / F1 / Brier at t=0.5.
  * **Per-artifact recall** - detection rate broken down by which generation
    transform produced each pseudo-fake.
  * **External validation (optional)** - ``--real DIR --fake DIR`` evaluates a
    model trained on the full pseudo corpus against a genuinely labelled test
    set, the bridge from self-supervised training to real-world deepfakes.

Outputs land in ``paper/``:
    paper/results.json          machine-readable metrics
    paper/results.md            markdown tables (paste into the paper)
    paper/figures/roc.png       ROC curves (all models)
    paper/figures/pr.png        precision-recall curves
    paper/figures/confusion.png fusion confusion matrix
    paper/figures/scores.png    fusion score distribution
    paper/figures/calibration.png reliability diagram
    paper/figures/per_artifact.png per-transform recall

Examples
--------
    python3 fetch_corpus.py --n 160                   # a real photographic corpus
    python3 evaluate.py --folds 5 --per-image 2       # what produced the figures
    python3 evaluate.py --real data/real --fake data/fake   # + external test

``--patch`` defaults to *off* for corpora of 50+ images. Patching a 160-photo
corpus into 128px tiles yields 15,360 pseudo-samples and hours of feature
extraction; it exists for corpora too small to supply enough distinct sources.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

import deep_features
import selfsup
from detector import DEFAULT_WEIGHTS, SIGNAL_NAMES, detect
from fusion import (
    MODE_DEEP,
    MODE_FUSION,
    MODE_HANDCRAFTED,
    FusionConfig,
    FusionModel,
    deep_matrix,
    handcrafted_matrix,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
PAPER_DIR = HERE / "paper"

#: A corpus at least this large already has plenty of independent sources.
#: Patching it as well multiplies the sample count by ~48 (a 160-photo corpus
#: becomes 15,360 pseudo-samples) and turns a 15-minute evaluation into an
#: hours-long one for no statistical gain. Patching exists for *small* corpora.
PATCH_FREE_ABOVE = 50

#: Figures produced by this module, in report order.
IMAGE_FIGURES = ("roc.png", "pr.png", "confusion.png", "scores.png",
                 "calibration.png", "per_artifact.png")


def default_corpus() -> list[Path]:
    """Prefer the fetched photographic corpus, fall back to the demo samples.

    ``fetch_corpus.py`` writes real photographs to ``corpus/``. Training on the
    four images in ``samples/`` instead is what produced the original shipped
    model, and it is not good enough - so if a real corpus is present, use it,
    and say which one was chosen either way.
    """
    corpus = HERE / "corpus"
    if corpus.is_dir() and any(corpus.glob("*.jpg")):
        return [corpus]
    return [HERE / "samples"]


FIG_DIR = PAPER_DIR / "figures"


# --------------------------------------------------------------------------- #
# metrics
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


def classical_scores(imgs: list[Image.Image]) -> np.ndarray:
    """Training-free baseline: the fixed-weight linear combiner from detector.py."""
    out = []
    for im in imgs:
        r = detect(im, weights=DEFAULT_WEIGHTS)
        out.append(r["combined"])
    return np.asarray(out, dtype=np.float64)


# --------------------------------------------------------------------------- #
# cross-validated out-of-fold predictions
# --------------------------------------------------------------------------- #


def grouped_oof(
    mode: str, cfg_base: FusionConfig, X_hand, X_deep, y, groups, folds: int
) -> tuple[np.ndarray, list[float]]:
    """Out-of-fold P(fake) and per-fold AUC for one feature mode (GroupKFold)."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    oof = np.full(len(y), np.nan)
    fold_aucs: list[float] = []
    gkf = GroupKFold(n_splits=folds)
    for tr, te in gkf.split(np.zeros(len(y)), y, groups):
        cfg = FusionConfig(**{**cfg_base.__dict__, "mode": mode})
        model = FusionModel(cfg)
        xh_tr = X_hand[tr] if model.uses_handcrafted else None
        xd_tr = X_deep[tr] if model.uses_deep else None
        model.fit(xh_tr, xd_tr, y[tr])
        xh_te = X_hand[te] if model.uses_handcrafted else None
        xd_te = X_deep[te] if model.uses_deep else None
        oof[te] = model.predict_proba(xh_te, xd_te)
        if len(np.unique(y[te])) == 2:
            fold_aucs.append(float(roc_auc_score(y[te], oof[te])))
    return oof, fold_aucs


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #


def plot_roc(y, preds: dict[str, np.ndarray], path: Path) -> None:
    from sklearn.metrics import roc_auc_score, roc_curve

    plt.figure(figsize=(5, 5))
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, p):.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="chance")
    plt.xlabel("false positive rate"); plt.ylabel("true positive rate")
    plt.title("ROC - leave-source-out CV"); plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_pr(y, preds: dict[str, np.ndarray], path: Path) -> None:
    from sklearn.metrics import average_precision_score, precision_recall_curve

    plt.figure(figsize=(5, 5))
    for name, p in preds.items():
        prec, rec, _ = precision_recall_curve(y, p)
        plt.plot(rec, prec, label=f"{name} (AP={average_precision_score(y, p):.3f})")
    plt.xlabel("recall"); plt.ylabel("precision")
    plt.title("Precision-Recall - leave-source-out CV")
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_confusion(cm: dict, path: Path) -> None:
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    plt.figure(figsize=(4, 4))
    plt.imshow(mat, cmap="Blues")
    for (i, j), v in np.ndenumerate(mat):
        plt.text(j, i, str(v), ha="center", va="center",
                 color="white" if v > mat.max() / 2 else "black", fontsize=14)
    plt.xticks([0, 1], ["pred real", "pred fake"])
    plt.yticks([0, 1], ["real", "fake"])
    plt.title("Fusion confusion matrix (t=0.5)")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_scores(y, p, path: Path) -> None:
    plt.figure(figsize=(5, 4))
    plt.hist(p[y == 0], bins=25, alpha=0.6, label="real", density=True)
    plt.hist(p[y == 1], bins=25, alpha=0.6, label="fake", density=True)
    plt.axvline(0.5, color="k", ls="--", alpha=0.5)
    plt.xlabel("P(synthetic)"); plt.ylabel("density")
    plt.title("Fusion score distribution"); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_calibration(y, p, path: Path) -> None:
    from sklearn.calibration import calibration_curve

    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect")
    plt.plot(mean_pred, frac_pos, "o-", label="fusion")
    plt.xlabel("mean predicted P(synthetic)"); plt.ylabel("observed fraction synthetic")
    plt.title("Reliability diagram"); plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_per_artifact(recall_by_t: dict[str, float], path: Path) -> None:
    items = sorted(recall_by_t.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    plt.figure(figsize=(6, 3.5))
    plt.barh(names, vals, color="steelblue")
    plt.xlabel("recall (fusion, t=0.5)"); plt.xlim(0, 1)
    plt.title("Per-artifact detection rate")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


# --------------------------------------------------------------------------- #
# markdown report
# --------------------------------------------------------------------------- #


def write_markdown(results: dict, path: Path) -> None:
    lines = ["# Evaluation results", "",
             f"_Protocol: {results['protocol']}_  ",
             f"_Corpus: {results['corpus']['n_images']} images "
             f"-> {results['corpus']['n_sources']} sources "
             f"-> {results['corpus']['n_samples']} pseudo-samples "
             f"({results['corpus']['n_real']} real / {results['corpus']['n_fake']} fake)_",
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
        lines += ["", "## Per-artifact recall (fusion)", "", "| Artifact | Recall | n |",
                  "|---|---|---|"]
        for t, (r, n) in sorted(results["per_artifact"].items(), key=lambda kv: kv[1][0]):
            lines.append(f"| {t} | {r:.3f} | {n} |")
    if results.get("external"):
        lines += ["", "## External validation (labelled test set)", "",
                  "| Metric | Value |", "|---|---|"]
        for k, v in results["external"]["metrics"].items():
            if k != "confusion":
                lines.append(f"| {k} | {v:.3f} |")
    lines += ["", "## Figures", ""]
    # Only the figures *this* script wrote - evaluate_audio.py shares FIG_DIR, and
    # globbing it would embed the audio ROC in the image report.
    for name in IMAGE_FIGURES:
        if (FIG_DIR / name).exists():
            lines.append(f"![{Path(name).stem}](figures/{name})")
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, nargs="+", default=None,
                    help="unlabeled corpus roots (default: corpus/ if present, "
                         "else samples/)")
    ap.add_argument("--patch", type=int, default=None,
                    help="tile each corpus image into PxP patches, multiplying the "
                         "number of independent sources; 0 disables. Default: off "
                         f"for corpora of >= {PATCH_FREE_ABOVE} images, 128 below.")
    ap.add_argument("--per-image", type=int, default=4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--classifier", default="gboost",
                    choices=["logreg", "mlp", "gboost", "rf", "svm"])
    ap.add_argument("--no-spectral", action="store_true")
    ap.add_argument("--deep-pca-dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--real", type=Path, default=None, help="labelled real dir (external test)")
    ap.add_argument("--fake", type=Path, default=None, help="labelled fake dir (external test)")
    args = ap.parse_args()
    # Line-buffer stdout: this job runs for minutes and its progress is useless
    # if it sits in a 4 KB block buffer until the process exits.
    sys.stdout.reconfigure(line_buffering=True)
    if args.corpus is None:
        args.corpus = default_corpus()
    print(f"corpus roots: {', '.join(str(c) for c in args.corpus)}")

    have_deep = deep_features.available()
    modes = [MODE_HANDCRAFTED]
    if have_deep:
        modes += [MODE_DEEP, MODE_FUSION]
    else:
        print("WARNING: torch unavailable -> evaluating handcrafted mode only.", file=sys.stderr)

    cfg_base = FusionConfig(classifier=args.classifier, spectral=not args.no_spectral,
                            deep_pca_dim=args.deep_pca_dim, seed=args.seed)

    # ---- corpus -> pseudo dataset ----
    paths = selfsup.gather_corpus(args.corpus)
    if not paths:
        print(f"ERROR: no images under {args.corpus}", file=sys.stderr)
        return 1
    if args.patch is None:
        args.patch = 0 if len(paths) >= PATCH_FREE_ABOVE else 128
        print(f"patch: {args.patch or 'off'} (auto, from {len(paths)} corpus images)")
    sources = selfsup.load_sources(paths, patch=args.patch or None)
    samples = selfsup.generate(sources, per_image=args.per_image, seed=args.seed)
    imgs = [s.image for s in samples]
    y = np.array([s.label for s in samples])
    groups = np.array([s.source for s in samples])
    transforms = [s.transforms for s in samples]
    n_groups = len(np.unique(groups))
    folds = max(2, min(args.folds, n_groups))
    print(f"corpus {len(paths)} imgs -> {len(sources)} sources -> {len(samples)} samples "
          f"({(y==0).sum()} real / {(y==1).sum()} fake); GroupKFold folds={folds}")

    # ---- features (extract once) ----
    t0 = time.time()
    X_hand = handcrafted_matrix(imgs)
    X_deep = deep_matrix(imgs, spectral=cfg_base.spectral) if have_deep else None
    print(f"features in {time.time()-t0:.1f}s")

    # ---- cross-validated predictions per model ----
    results = {
        "protocol": f"leave-source-out {folds}-fold GroupKFold "
                    f"(classifier={args.classifier}, seed={args.seed})",
        "corpus": {"n_images": len(paths), "n_sources": len(sources),
                   "n_samples": len(samples), "n_real": int((y == 0).sum()),
                   "n_fake": int((y == 1).sum())},
        "models": {}, "fold_auc": {},
    }
    preds_for_plot: dict[str, np.ndarray] = {}

    # training-free classical baseline
    p_classical = classical_scores(imgs)
    results["models"]["classical (fixed weights)"] = compute_metrics(y, p_classical)
    preds_for_plot["classical"] = p_classical

    fusion_oof = None
    for mode in modes:
        oof, fold_aucs = grouped_oof(mode, cfg_base, X_hand, X_deep, y, groups, folds)
        results["models"][mode] = compute_metrics(y, oof)
        results["fold_auc"][mode] = fold_aucs
        preds_for_plot[mode] = oof
        print(f"  {mode:11s} AUC={results['models'][mode]['auc']:.3f} "
              f"AP={results['models'][mode]['ap']:.3f} "
              f"F1={results['models'][mode]['f1']:.3f}")
        if mode == MODE_FUSION:
            fusion_oof = oof
    if fusion_oof is None:  # handcrafted-only environment
        fusion_oof = preds_for_plot[MODE_HANDCRAFTED]
        ref_mode = MODE_HANDCRAFTED
    else:
        ref_mode = MODE_FUSION

    # ---- per-artifact recall (reference model) ----
    yhat = (fusion_oof >= 0.5).astype(int)
    hit, tot = defaultdict(int), defaultdict(int)
    for i, lab in enumerate(y):
        if lab == 1:
            for t in transforms[i]:
                tot[t] += 1
                hit[t] += int(yhat[i] == 1)
    per_artifact = {t: (hit[t] / tot[t], tot[t]) for t in tot}
    results["per_artifact"] = per_artifact

    # ---- external labelled validation (optional) ----
    if args.real and args.fake:
        ext = evaluate_external(args, cfg_base, have_deep, X_hand, X_deep, y)
        if ext:
            results["external"] = ext

    # ---- figures ----
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_roc(y, preds_for_plot, FIG_DIR / "roc.png")
    plot_pr(y, preds_for_plot, FIG_DIR / "pr.png")
    plot_confusion(results["models"][ref_mode]["confusion"], FIG_DIR / "confusion.png")
    plot_scores(y, fusion_oof, FIG_DIR / "scores.png")
    plot_calibration(y, fusion_oof, FIG_DIR / "calibration.png")
    plot_per_artifact({t: r for t, (r, _) in per_artifact.items()},
                      FIG_DIR / "per_artifact.png")

    # ---- write reports ----
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    (PAPER_DIR / "results.json").write_text(json.dumps(results, indent=2))
    write_markdown(results, PAPER_DIR / "results.md")
    n_figs = sum((FIG_DIR / f).exists() for f in IMAGE_FIGURES)
    print(f"\nwrote {PAPER_DIR/'results.json'}, {PAPER_DIR/'results.md'}, "
          f"and {n_figs} figures to {FIG_DIR}/")
    return 0


def evaluate_external(args, cfg_base, have_deep, X_hand, X_deep, y) -> dict | None:
    """Train the reference model on the full pseudo corpus, test on labelled data."""
    real_paths = selfsup.gather_corpus([args.real])
    fake_paths = selfsup.gather_corpus([args.fake])
    if not real_paths or not fake_paths:
        print("external: missing real/fake images, skipping", file=sys.stderr)
        return None
    test_imgs, test_y = [], []
    for p in real_paths:
        try:
            test_imgs.append(Image.open(p).convert("RGB")); test_y.append(0)
        except Exception:
            pass
    for p in fake_paths:
        try:
            test_imgs.append(Image.open(p).convert("RGB")); test_y.append(1)
        except Exception:
            pass
    test_y = np.array(test_y)
    mode = MODE_FUSION if have_deep else MODE_HANDCRAFTED
    cfg = FusionConfig(**{**cfg_base.__dict__, "mode": mode})
    model = FusionModel(cfg)
    model.fit(X_hand if model.uses_handcrafted else None,
              X_deep if model.uses_deep else None, y)
    p = model.predict_proba_images(test_imgs)
    m = compute_metrics(test_y, p)
    print(f"external: n={len(test_y)} AUC={m['auc']:.3f} acc={m['accuracy']:.3f}")
    return {"n_real": int((test_y == 0).sum()), "n_fake": int((test_y == 1).sum()),
            "metrics": m}


if __name__ == "__main__":
    sys.exit(main())
