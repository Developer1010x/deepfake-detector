"""Classical pattern-recognition checks for AI-generated images.

These are template-matching / structural-pattern algorithms that complement
the per-image statistical signals in detector.py:

  spectral_peak_pattern  - detect FFT peaks at frequencies typically
                           introduced by upsampling layers (f = N/2, N/4,
                           N/8 etc. of the Nyquist limit).
  block_heterogeneity    - per-block FFT-score spatial variance: real photos
                           have homogeneous statistics, AI images often have
                           regional inconsistencies.
  lbp_pattern            - Local Binary Pattern uniform-code mass deviation
                           from natural-image reference (~0.9 for real
                           photos; lower or higher for synthetics).

All pure numpy, no learned models.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from detector import _center_square, _grayscale, fft_score


def _uniform_lbp_codes() -> np.ndarray:
    """The 58 'uniform' LBP-8 codes: bit strings with at most 2 circular
    0-1 transitions. Natural-image LBP histograms are dominated by these
    (Ojala et al. 2002)."""
    codes = []
    for i in range(256):
        bits = format(i, "08b")
        rotated = bits[1:] + bits[0]
        if sum(1 for a, b in zip(bits, rotated) if a != b) <= 2:
            codes.append(i)
    return np.array(codes, dtype=np.int32)


_UNIFORM_LBP = _uniform_lbp_codes()


def spectral_peak_pattern(img: Image.Image) -> dict:
    """Look for radial-spectrum peaks at sub-Nyquist fractional frequencies.

    Most generative-model upsamplers (transpose conv, nearest-neighbor +
    conv, sub-pixel) tile a fixed kernel at strides 2 / 4 / 8 — this puts
    energy at f = N/2, N/4, N/8 of Nyquist. Real photos have a smooth
    1/f^α radial spectrum with no peaks at these frequencies.

    Implementation: compute radial profile, detrend with a quadratic in
    log-radius, then check for excess energy in a small window around each
    target frequency. Score = sum of peak prominences (in noise-σ units),
    capped at 1.
    """
    g = _center_square(_grayscale(img))
    s = g.shape[0]
    if s < 64:
        return {"score": 0.0, "peaks": []}

    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(g))))
    cy = cx = s // 2
    y, x = np.indices((s, s))
    r = np.hypot(x - cx, y - cy).astype(np.int32)
    radial = np.bincount(r.ravel(), spectrum.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    n = len(radial)

    # Detrend with a quadratic in log-radius.
    rr = np.log1p(np.arange(n))
    coeffs = np.polyfit(rr, radial, 2)
    trend = np.polyval(coeffs, rr)
    residual = radial - trend

    half = n // 2
    if half < 4:
        return {"score": 0.0, "peaks": []}
    noise = float(residual[half:].std()) + 1e-8

    target_fractions = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75)
    peaks = []
    for frac in target_fractions:
        idx = int(round(frac * half))
        lo, hi = max(0, idx - 2), min(n, idx + 3)
        window = residual[lo:hi]
        if window.size == 0:
            continue
        prom = float(window.max() / noise)
        if prom > 2.0:
            peaks.append({"frequency": frac, "prominence_sigma": round(prom, 2)})
    total_prom = sum(p["prominence_sigma"] for p in peaks)
    return {"score": float(np.clip(total_prom / 20.0, 0.0, 1.0)), "peaks": peaks}


def block_heterogeneity(img: Image.Image, n_blocks: int = 4) -> dict:
    """Spatial heterogeneity of per-block FFT scores.

    A real photograph captures one scene with one lens and one ISP — so its
    statistics are spatially homogeneous. Synthetic images, especially
    those generated in tiles or via patch-based denoising, often have
    spatially varying statistics. Score = std of per-block FFT scores,
    rescaled to [0, 1].
    """
    g = _grayscale(img)
    h, w = g.shape
    bh, bw = h // n_blocks, w // n_blocks
    if bh < 32 or bw < 32:
        return {"score": 0.0, "block_scores": [], "block_std": 0.0}

    block_scores = []
    for i in range(n_blocks):
        for j in range(n_blocks):
            tile = g[i * bh : (i + 1) * bh, j * bw : (j + 1) * bw]
            block_scores.append(fft_score(Image.fromarray(tile.astype(np.uint8))))
    std = float(np.std(block_scores))
    return {
        "score": float(np.clip(std * 5.0, 0.0, 1.0)),
        "block_scores": [round(b, 3) for b in block_scores],
        "block_std": round(std, 4),
    }


def lbp_pattern(img: Image.Image) -> dict:
    """Local Binary Pattern uniform-code mass deviation.

    For each pixel, build an 8-bit code from comparisons with its 8
    immediate neighbors (clockwise: TL, T, TR, R, BR, B, BL, L). Bit set
    if neighbor >= center. The histogram of these 256 codes characterizes
    local texture. Natural images concentrate ~0.85-0.95 of their LBP mass
    in the 58 'uniform' codes (those with <= 2 circular 0-1 transitions).
    Synthetic images often deviate from this — either too high (overly
    smooth) or too low (overly textured / patterned).

    Score = clip(5 * |uniform_mass - 0.9|, 0, 1).
    """
    g = _grayscale(img).astype(np.int32)
    h, w = g.shape
    if h < 32 or w < 32:
        return {"score": 0.0, "uniform_mass": 0.0, "entropy": 0.0}

    center = g[1:-1, 1:-1]
    lbp = np.zeros_like(center, dtype=np.uint8)
    bits = (
        (g[:-2, :-2],  1 << 7),
        (g[:-2, 1:-1], 1 << 6),
        (g[:-2, 2:],   1 << 5),
        (g[1:-1, 2:],  1 << 4),
        (g[2:, 2:],    1 << 3),
        (g[2:, 1:-1],  1 << 2),
        (g[2:, :-2],   1 << 1),
        (g[1:-1, :-2], 1 << 0),
    )
    for neigh, mask in bits:
        lbp |= ((neigh >= center).astype(np.uint8) * mask)

    hist = np.bincount(lbp.ravel(), minlength=256).astype(np.float64)
    hist /= max(hist.sum(), 1.0)
    uniform_mass = float(hist[_UNIFORM_LBP].sum())
    nz = hist[hist > 0]
    entropy = float(-(nz * np.log2(nz)).sum())
    score = float(np.clip(5.0 * abs(0.9 - uniform_mass), 0.0, 1.0))
    return {
        "score": score,
        "uniform_mass": round(uniform_mass, 4),
        "entropy": round(entropy, 4),
    }


def all_patterns(img: Image.Image) -> dict:
    """Run all three pattern checks and combine their scores."""
    sp = spectral_peak_pattern(img)
    bh = block_heterogeneity(img)
    lp = lbp_pattern(img)
    combined = 0.50 * sp["score"] + 0.30 * bh["score"] + 0.20 * lp["score"]
    return {
        "spectral_peaks":      sp,
        "block_heterogeneity": bh,
        "lbp":                 lp,
        "pattern_score":       float(np.clip(combined, 0.0, 1.0)),
    }
