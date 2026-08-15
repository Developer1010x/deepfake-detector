"""Self-supervised training entry point.

Pipeline (no hand-labelled deepfakes anywhere):

    corpus images
        -> selfsup.generate()        # construct balanced pseudo-real/fake views
        -> feature extraction         # 13-d hand-crafted + 1024-d deep
        -> grouped train/val split    # split by *source image* (no leakage)
        -> FusionModel.fit()          # calibrated classifier on pseudo-labels
        -> held-out pseudo metrics    # sanity AUC/accuracy
        -> save models/fusion.joblib

Splitting by source image is important: the pseudo-real and pseudo-fake views of
one photo must not straddle the train/val boundary, or the validation score is
optimistic. We therefore group on ``PseudoSample.source``.

Examples
--------
    python3 fetch_corpus.py --n 160                   # get a real photo corpus
    python3 train_selfsup.py --patch 0 --per-image 3   # what ships in models/
    python3 train_selfsup.py --corpus /data/images --per-image 4 --classifier mlp
    python3 train_selfsup.py --mode handcrafted       # no torch needed
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

import deep_features
import selfsup
from fusion import FEATURE_MODES, MODE_FUSION, FusionConfig, FusionModel
from pipeline import env_versions

HERE = Path(__file__).resolve().parent


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


def grouped_split(groups: np.ndarray, val_frac: float, seed: int) -> np.ndarray:
    """Boolean 'is validation' mask, split by unique group id (source image)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * val_frac))) if len(uniq) >= 2 else 0
    val_groups = set(uniq[:n_val].tolist())
    return np.array([g in val_groups for g in groups])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, nargs="+", default=None,
                    help="image dirs/files used as the unlabeled corpus "
                         "(default: corpus/ if fetch_corpus.py has run, else samples/)")
    ap.add_argument("--per-image", type=int, default=4,
                    help="pseudo-real (and pseudo-fake) views per source")
    ap.add_argument("--patch", type=int, default=128,
                    help="tile each corpus image into PxP content patches "
                         "(distinct sources); 0 disables patching (default: 128)")
    ap.add_argument("--mode", choices=FEATURE_MODES, default=MODE_FUSION,
                    help="feature families to use (default: fusion)")
    ap.add_argument("--classifier", default="gboost",
                    choices=["logreg", "mlp", "gboost", "rf", "svm"])
    ap.add_argument("--no-spectral", action="store_true",
                    help="disable the spectral deep stream (spatial only)")
    ap.add_argument("--deep-pca-dim", type=int, default=64)
    ap.add_argument("--no-calibrate", action="store_true")
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=None, help="cap corpus size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=HERE / "models" / "fusion.joblib")
    args = ap.parse_args()
    # Line-buffer stdout: this job runs for minutes and its progress is useless
    # if it sits in a 4 KB block buffer until the process exits.
    sys.stdout.reconfigure(line_buffering=True)
    if args.corpus is None:
        args.corpus = default_corpus()

    cfg = FusionConfig(
        classifier=args.classifier,
        mode=args.mode,
        spectral=not args.no_spectral,
        deep_pca_dim=args.deep_pca_dim,
        calibrate=not args.no_calibrate,
        seed=args.seed,
    )

    if cfg.mode in ("deep", "fusion") and not deep_features.available():
        print("ERROR: --mode requires torch/torchvision, which are unavailable.\n"
              "       Install them or use --mode handcrafted.", file=sys.stderr)
        return 1

    # 1. Corpus -> pseudo-labelled dataset.
    paths = selfsup.gather_corpus(args.corpus, limit=args.limit)
    if not paths:
        print(f"ERROR: no images found under {args.corpus}", file=sys.stderr)
        return 1
    sources = selfsup.load_sources(paths, patch=args.patch or None)
    print(f"corpus: {len(paths)} images -> {len(sources)} sources "
          f"(patch={args.patch or 'off'}) from {', '.join(str(c) for c in args.corpus)}")
    samples = selfsup.generate(sources, per_image=args.per_image, seed=args.seed)
    y = np.array([s.label for s in samples])
    groups = np.array([s.source for s in samples])
    print(f"generated {len(samples)} pseudo-samples "
          f"({(y == 0).sum()} real / {(y == 1).sum()} fake) "
          f"from {len(sources)} sources, {args.per_image} views each")

    # 2. Feature extraction.
    t0 = time.time()
    imgs = [s.image for s in samples]
    X_hand = None
    X_deep = None
    if cfg.mode in ("handcrafted", "fusion"):
        from fusion import handcrafted_matrix

        X_hand = handcrafted_matrix(imgs)
    if cfg.mode in ("deep", "fusion"):
        from fusion import deep_matrix

        print("extracting deep features (first run downloads backbone weights)...")
        X_deep = deep_matrix(imgs, spectral=cfg.spectral)
    print(f"features extracted in {time.time() - t0:.1f}s "
          f"(hand={None if X_hand is None else X_hand.shape}, "
          f"deep={None if X_deep is None else X_deep.shape})")

    # 3. Grouped split (no source image straddles the boundary).
    val_mask = grouped_split(groups, args.val_frac, args.seed)
    tr = ~val_mask

    def _sel(M, m):
        return None if M is None else M[m]

    model = FusionModel(cfg)
    model.fit(_sel(X_hand, tr), _sel(X_deep, tr), y[tr])
    print(f"\ntrained {cfg.classifier} ({cfg.mode}) on {tr.sum()} samples; "
          f"calibration: {model.train_meta_['calibrated']}")

    # 4. Held-out pseudo-validation (sanity only - not the paper's real metric).
    if val_mask.sum() >= 2 and len(np.unique(y[val_mask])) == 2:
        from sklearn.metrics import accuracy_score, roc_auc_score

        p_val = model.predict_proba(_sel(X_hand, val_mask), _sel(X_deep, val_mask))
        auc = roc_auc_score(y[val_mask], p_val)
        acc = accuracy_score(y[val_mask], (p_val > 0.5).astype(int))
        print(f"held-out pseudo-val: n={int(val_mask.sum())}  "
              f"AUC={auc:.3f}  acc={acc:.3f}")
        model.train_meta_.update(val_auc=round(float(auc), 4),
                                 val_acc=round(float(acc), 4),
                                 val_n=int(val_mask.sum()))
    else:
        print("held-out pseudo-val: skipped (too few validation groups)")

    # 5. Model card, then persist.
    #
    # Without this block the artifact is anonymous: a downloader cannot tell
    # whether the .joblib they got was fitted on a real photographic corpus or
    # on four demo images, nor which sklearn/torch pickled it. Both questions
    # have burned this project before, so they are answered on disk.
    model.train_meta_["corpus"] = ", ".join(_describe_corpus(c) for c in args.corpus)
    model.train_meta_["corpus_images"] = len(paths)
    model.train_meta_["corpus_sources"] = len(sources)
    model.train_meta_["patch"] = args.patch or None
    model.train_meta_["per_image"] = args.per_image
    model.train_meta_["env"] = env_versions()
    model.save(args.out)
    print(f"\nsaved model -> {args.out}")
    print(f"saved model card -> {args.out.with_suffix('.meta.json')}")
    return 0


def _describe_corpus(root: Path) -> str:
    """Human-readable provenance for one corpus root, using its manifest if any."""
    manifest = root / "manifest.json"
    if manifest.is_dir() or not manifest.exists():
        return root.name
    try:
        import json

        m = json.loads(manifest.read_text())
        return f"{root.name} ({m.get('n', '?')} images from {m.get('source', 'unknown')})"
    except Exception:
        return root.name


if __name__ == "__main__":
    sys.exit(main())
