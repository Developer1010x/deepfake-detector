"""Self-supervised pseudo-label generation for label-free deepfake detection.

The detector is trained *without any hand-labelled deepfakes*. Instead, we take
a corpus of ordinary images (which we treat as the "real" manifold) and
synthesise two views of each:

  * a **pseudo-real** view  - benign photometric/geometric augmentation only
    (the kind of edit a real photo legitimately undergoes: mild crop, flip,
    brightness/contrast jitter, light JPEG re-save).
  * a **pseudo-fake** view  - the same image with one or more *generation-style
    artifacts* injected. These transforms emulate, in a controlled and
    fully self-supervised way, the traces that real generative pipelines leave
    behind.

This is the core of the label-free paradigm: the supervision signal is
*constructed*, not annotated. It is directly inspired by self-blended images
(Shiohara & Yamasaki, CVPR 2022) and frequency-artifact studies of GAN/diffusion
up-samplers (Wang et al. 2020; Frank et al. 2020; Corvi et al. 2023). A
classifier that learns to separate these two views generalises to genuine
synthetic media because it keys on the *forensic artifact*, not on any specific
generator.

Pseudo-fake transforms (each parameter is randomised per sample):
  upsample_fingerprint  - downscale then upscale, implanting the periodic
                          spectral peaks characteristic of transpose-conv /
                          nearest-neighbour up-samplers.
  self_blend            - paste a warped, colour-shifted copy of the image back
                          over itself through a soft elliptical mask, creating
                          the tell-tale blending boundary of face-swaps/inpaints.
  double_jpeg           - re-encode twice at mismatched qualities, the
                          "generated-then-saved" double-compression trace.
  spectral_inject       - add a faint periodic sinusoid in the frequency domain
                          (a synthetic generator fingerprint).
  diffusion_smooth      - bilateral/Gaussian over-smoothing that flattens the
                          heavy-tailed high-frequency statistics of real sensors.

All transforms are pure numpy / Pillow / OpenCV and deterministic given a seed.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# --------------------------------------------------------------------------- #
# corpus gathering
# --------------------------------------------------------------------------- #


def gather_corpus(roots: list[Path], limit: int | None = None) -> list[Path]:
    """Recursively collect image paths under the given roots (any labels are
    ignored - every image is treated as an unlabeled corpus member)."""
    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file() and root.suffix.lower() in IMAGE_EXTS:
            paths.append(root)
            continue
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() in IMAGE_EXTS:
                paths.append(p)
    # Deduplicate while preserving order.
    seen, uniq = set(), []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq[:limit] if limit else uniq


# --------------------------------------------------------------------------- #
# low-level pseudo-fake transforms (numpy uint8 HWC RGB in/out)
# --------------------------------------------------------------------------- #


def _as_uint8(a: np.ndarray) -> np.ndarray:
    return np.clip(a, 0, 255).astype(np.uint8)


def upsample_fingerprint(a: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = a.shape[:2]
    factor = rng.choice([2, 3, 4])
    down = cv2.resize(a, (max(1, w // factor), max(1, h // factor)),
                      interpolation=cv2.INTER_AREA)
    up_interp = rng.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
    return cv2.resize(down, (w, h), interpolation=int(up_interp))


def self_blend(a: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Warp + colour-shift a copy and blend it back through a soft mask."""
    h, w = a.shape[:2]
    src = a.astype(np.float32)

    # Mild affine warp of the donor copy.
    angle = rng.uniform(-6, 6)
    tx, ty = rng.uniform(-0.04, 0.04) * w, rng.uniform(-0.04, 0.04) * h
    scale = rng.uniform(0.96, 1.04)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[:, 2] += (tx, ty)
    donor = cv2.warpAffine(src, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Photometric shift so a blending boundary is forensically visible.
    donor *= rng.uniform(0.9, 1.1, size=3).astype(np.float32)
    donor += rng.uniform(-12, 12, size=3).astype(np.float32)

    # Soft elliptical mask centred on a random point.
    cy, cx = rng.uniform(0.3, 0.7) * h, rng.uniform(0.3, 0.7) * w
    ry, rx = rng.uniform(0.2, 0.45) * h, rng.uniform(0.2, 0.45) * w
    yy, xx = np.ogrid[:h, :w]
    mask = (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1.0
    mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0),
                            sigmaX=rng.uniform(6, 18))[:, :, None]
    blended = src * (1 - mask) + donor * mask
    return _as_uint8(blended)


def double_jpeg(a: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    img = Image.fromarray(a)
    for q in (int(rng.integers(60, 96)), int(rng.integers(60, 96))):
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=q)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return np.asarray(img)


def spectral_inject(a: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add a faint 2-D sinusoid (a synthetic periodic generator fingerprint)."""
    h, w = a.shape[:2]
    fy, fx = rng.integers(2, 8), rng.integers(2, 8)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    phase = rng.uniform(0, 2 * np.pi)
    amp = rng.uniform(2.0, 6.0)
    pattern = amp * np.sin(2 * np.pi * (fy * yy / h + fx * xx / w) + phase)
    return _as_uint8(a.astype(np.float32) + pattern[:, :, None])


def diffusion_smooth(a: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Over-smooth to flatten heavy-tailed natural high-frequency statistics."""
    if rng.random() < 0.5:
        d = int(rng.choice([5, 7, 9]))
        out = cv2.bilateralFilter(a, d, sigmaColor=rng.uniform(40, 90),
                                  sigmaSpace=rng.uniform(40, 90))
    else:
        k = int(rng.choice([3, 5]))
        out = cv2.GaussianBlur(a, (k, k), sigmaX=rng.uniform(0.6, 1.4))
    return out


_FAKE_TRANSFORMS = {
    "upsample_fingerprint": upsample_fingerprint,
    "self_blend": self_blend,
    "double_jpeg": double_jpeg,
    "spectral_inject": spectral_inject,
    "diffusion_smooth": diffusion_smooth,
}


# --------------------------------------------------------------------------- #
# benign (pseudo-real) augmentation
# --------------------------------------------------------------------------- #


def benign_augment(a: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = a
    if rng.random() < 0.5:  # horizontal flip
        out = out[:, ::-1]
    if rng.random() < 0.7:  # mild brightness/contrast
        gain = rng.uniform(0.92, 1.08)
        bias = rng.uniform(-8, 8)
        out = _as_uint8(out.astype(np.float32) * gain + bias)
    if rng.random() < 0.5:  # a single light JPEG re-save (cameras do this)
        buf = io.BytesIO()
        Image.fromarray(out).save(buf, "JPEG", quality=int(rng.integers(88, 98)))
        buf.seek(0)
        out = np.asarray(Image.open(buf).convert("RGB"))
    return np.ascontiguousarray(out)


# --------------------------------------------------------------------------- #
# sample generation
# --------------------------------------------------------------------------- #


@dataclass
class PseudoSample:
    image: Image.Image
    label: int          # 0 = pseudo-real, 1 = pseudo-fake
    source: str         # originating source id (file, or file#patch)
    transforms: list[str] = field(default_factory=list)


def _load_rgb(path: Path, max_side: int = 1024) -> np.ndarray | None:
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return None
    w, h = img.size
    if max(w, h) > max_side:  # bound work on huge inputs
        s = max_side / max(w, h)
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
    return np.asarray(img)


def load_sources(
    paths: list[Path],
    patch: int | None = None,
    max_side: int = 1024,
) -> list[tuple[str, np.ndarray]]:
    """Load each path into one or more named RGB *sources*.

    With ``patch=None`` each file is a single source. With ``patch=P`` each file
    is tiled into non-overlapping ``P x P`` content patches, each becoming an
    independent source (``"name#rRcC"``). Patchifying multiplies the number of
    distinct *scenes* available for self-supervision and for leakage-free
    grouped evaluation, without any external data. Patches smaller than ``P``
    at the borders are dropped.
    """
    sources: list[tuple[str, np.ndarray]] = []
    for path in paths:
        arr = _load_rgb(path, max_side=max_side)
        if arr is None:
            continue
        if patch is None:
            sources.append((path.name, arr))
            continue
        h, w = arr.shape[:2]
        nr, nc = h // patch, w // patch
        if nr == 0 or nc == 0:                      # too small to tile -> whole
            sources.append((path.name, arr))
            continue
        for i in range(nr):
            for j in range(nc):
                tile = arr[i * patch:(i + 1) * patch, j * patch:(j + 1) * patch]
                sources.append((f"{path.stem}#r{i}c{j}", np.ascontiguousarray(tile)))
    return sources


def make_pseudo_fake(a: np.ndarray, rng: np.random.Generator,
                     n_ops: tuple[int, int] = (1, 2)) -> tuple[np.ndarray, list[str]]:
    """Apply a random 1-2 generation-style transforms in sequence."""
    k = int(rng.integers(n_ops[0], n_ops[1] + 1))
    names = list(rng.choice(list(_FAKE_TRANSFORMS), size=k, replace=False))
    out = a
    for name in names:
        out = _FAKE_TRANSFORMS[name](out, rng)
    return out, names


def generate(
    sources: list[tuple[str, np.ndarray]],
    per_image: int = 1,
    seed: int = 0,
) -> list[PseudoSample]:
    """Build a balanced self-supervised dataset from named RGB sources.

    For each source we emit ``per_image`` pseudo-real and ``per_image``
    pseudo-fake samples, giving a perfectly balanced set with no human labels.
    Sources are the output of :func:`load_sources` (whole images or patches).
    """
    rng = np.random.default_rng(seed)
    samples: list[PseudoSample] = []
    for name, base in sources:
        for _ in range(per_image):
            real = benign_augment(base, rng)
            samples.append(PseudoSample(Image.fromarray(real), 0, name, ["benign"]))
            fake_arr, ops = make_pseudo_fake(base, rng)
            # A pseudo-fake also undergoes a benign re-save, so the classifier
            # cannot cheat on the benign-augmentation distribution alone.
            fake_arr = benign_augment(fake_arr, rng)
            samples.append(PseudoSample(Image.fromarray(fake_arr), 1, name, ops))
    return samples
