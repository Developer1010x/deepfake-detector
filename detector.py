"""Classical deepfake / AI-image detector.

Seven independent signals, no learned models. Each returns a score in
[0, 1] where higher = more synthetic-looking.

Original signals:
  fft_score         - radial power-spectrum residual (upsampler artifacts)
  ela_score         - JPEG re-save error coefficient of variation
  benford_score     - chi-squared distance from Benford on DCT first digits

Advanced signals:
  cfa_score         - absence of Bayer / color-filter-array demosaicing trace
  wavelet_score     - kurtosis anomaly in Haar DWT detail sub-bands
  jpeg_ghost_score  - block-wise minimum-error quality inconsistency (Farid 2009)
  noise_score       - autocorrelation of high-pass residual at unit lags
"""

from __future__ import annotations

import io
import numpy as np
from PIL import Image


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _grayscale(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float32)


def _center_square(a: np.ndarray) -> np.ndarray:
    h, w = a.shape
    s = min(h, w)
    return a[(h - s) // 2 : (h - s) // 2 + s, (w - s) // 2 : (w - s) // 2 + s]


def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n).reshape(-1, 1)
    j = np.arange(n).reshape(1, -1)
    m = np.cos(np.pi * (2 * j + 1) * k / (2 * n))
    m[0, :] *= 1.0 / np.sqrt(n)
    m[1:, :] *= np.sqrt(2.0 / n)
    return m


_DCT8 = _dct_matrix(8)


def _haar_dwt(g: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = g.shape
    h, w = h - h % 2, w - w % 2
    g = g[:h, :w]
    a, b = g[0::2, 0::2], g[0::2, 1::2]
    c, d = g[1::2, 0::2], g[1::2, 1::2]
    return ((a + b + c + d) / 2.0,   # LL
            (a + b - c - d) / 2.0,   # LH
            (a - b + c - d) / 2.0,   # HL
            (a - b - c + d) / 2.0)   # HH


# --------------------------------------------------------------------------- #
# original signals
# --------------------------------------------------------------------------- #


def fft_score(img: Image.Image) -> float:
    g = _center_square(_grayscale(img))
    s = g.shape[0]
    if s < 32:
        return 0.5

    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(g))))
    cy = cx = s // 2
    y, x = np.indices((s, s))
    r = np.hypot(x - cx, y - cy).astype(np.int32)
    radial = np.bincount(r.ravel(), spectrum.ravel()) / np.maximum(np.bincount(r.ravel()), 1)

    hi = radial[len(radial) // 2 :]
    rr = np.log1p(np.arange(len(hi)))
    trend = np.polyval(np.polyfit(rr, hi, 1), rr)
    residual = hi - trend
    score = float(np.std(residual) / (np.std(hi) + 1e-8))
    return float(np.clip(score, 0.0, 1.0))


def ela_score(img: Image.Image, quality: int = 90) -> float:
    rgb = img.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf)
    diff = np.abs(np.asarray(rgb, dtype=np.float32) - np.asarray(resaved, dtype=np.float32))
    mean = float(diff.mean())
    std = float(diff.std())
    return std / (mean + std + 1e-6)


def benford_score(img: Image.Image) -> float:
    g = _grayscale(img)
    h, w = g.shape
    h8, w8 = h - h % 8, w - w % 8
    if h8 == 0 or w8 == 0:
        return 0.5
    g = g[:h8, :w8] - 128.0

    blocks = g.reshape(h8 // 8, 8, w8 // 8, 8).swapaxes(1, 2)
    coeffs = _DCT8 @ blocks @ _DCT8.T

    ac = np.abs(coeffs.reshape(-1, 64)[:, 1:]).ravel()
    ac = ac[ac >= 1.0]
    if ac.size == 0:
        return 0.5

    first = (ac / 10.0 ** np.floor(np.log10(ac))).astype(np.int32)
    observed = np.bincount(first, minlength=10)[1:10].astype(np.float64)
    observed /= observed.sum()
    expected = np.log10(1.0 + 1.0 / np.arange(1, 10))
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    return float(np.clip(chi2 / 0.5, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# advanced signals
# --------------------------------------------------------------------------- #


def cfa_score(img: Image.Image) -> float:
    """CFA / Bayer demosaicing trace.

    Camera images are demosaiced from a single-sensor Bayer pattern, which
    leaves a 2x2 periodic structure in the prediction residual: pixels in
    different positions of each 2x2 sub-grid have systematically different
    variance. Synthetic images, having never been demosaiced, show roughly
    uniform variance across the four sub-grids. Score = exp(-3 * mean
    coefficient-of-variation across channels) — uniform sub-grid variance
    pushes the score toward 1 (synthetic).
    """
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w, _ = rgb.shape
    if h < 16 or w < 16:
        return 0.5
    h, w = h - h % 2, w - w % 2
    rgb = rgb[:h, :w]

    asyms = []
    for c in range(3):
        ch = rgb[:, :, c]
        pred = (ch[:-2, 1:-1] + ch[2:, 1:-1] + ch[1:-1, :-2] + ch[1:-1, 2:]) / 4.0
        res = ch[1:-1, 1:-1] - pred
        h2, w2 = res.shape
        h2, w2 = h2 - h2 % 2, w2 - w2 % 2
        res = res[:h2, :w2]
        v = np.array([
            res[0::2, 0::2].var(),
            res[0::2, 1::2].var(),
            res[1::2, 0::2].var(),
            res[1::2, 1::2].var(),
        ])
        asyms.append(v.std() / (v.mean() + 1e-8))
    return float(np.clip(np.exp(-3.0 * float(np.mean(asyms))), 0.0, 1.0))


def wavelet_score(img: Image.Image) -> float:
    """Haar-DWT detail-band kurtosis anomaly.

    Natural-image wavelet detail coefficients are heavy-tailed: kurtosis in
    LH/HL/HH typically lands in 8-30 (Mallat, Field). Synthetic images
    often have less heavy tails, with kurtosis closer to Gaussian (3).
    Score maps low average kurtosis to high synthetic-likelihood.
    """
    g = _grayscale(img)
    if min(g.shape) < 16:
        return 0.5
    _, lh, hl, hh = _haar_dwt(g)
    kurts = []
    for band in (lh, hl, hh):
        b = band.ravel() - band.mean()
        std = b.std() + 1e-8
        kurts.append((b ** 4).mean() / std ** 4)
    avg_kurt = float(np.mean(kurts))
    return float(np.clip(1.0 - (avg_kurt - 3.0) / 20.0, 0.0, 1.0))


def jpeg_ghost_score(
    img: Image.Image,
    qualities: tuple[int, ...] = (60, 70, 80, 90),
    block: int = 16,
) -> float:
    """JPEG ghost (Farid 2009): re-encode at multiple qualities, measure
    spatial inconsistency in which quality minimizes block error.

    A single-source image: the quality-of-best-fit field is roughly
    constant across blocks. A spliced or generated image: different blocks
    pick different qualities, so the index field has high spatial variance.
    """
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w, _ = rgb.shape
    h, w = h - h % block, w - w % block
    if h < block * 2 or w < block * 2:
        return 0.5
    rgb = rgb[:h, :w]

    errors = []
    for q in qualities:
        buf = io.BytesIO()
        Image.fromarray(rgb.astype(np.uint8)).save(buf, "JPEG", quality=q)
        buf.seek(0)
        re = np.asarray(Image.open(buf), dtype=np.float32)
        diff = (rgb - re) ** 2
        err = diff.reshape(h // block, block, w // block, block, 3).mean(axis=(1, 3, 4))
        errors.append(err)
    errors = np.stack(errors, axis=0)
    min_q_idx = errors.argmin(axis=0).astype(np.float32)
    max_std = (len(qualities) - 1) / 2.0
    return float(np.clip(min_q_idx.std() / max(max_std, 1e-8), 0.0, 1.0))


def noise_score(img: Image.Image) -> float:
    """High-pass residual autocorrelation at unit lags.

    Sensor noise that survives a real ISP is approximately spatially
    uncorrelated. Synthetic-image residuals from upsampler stacks often
    carry pixel-to-pixel correlation. Compute residual = image - 3x3 mean,
    then average |r(1,0)|, |r(0,1)|, |r(1,1)|.
    """
    g = _grayscale(img)
    h, w = g.shape
    if min(h, w) < 16:
        return 0.5

    blur = (g[:-2, :-2] + g[:-2, 1:-1] + g[:-2, 2:] +
            g[1:-1, :-2] + g[1:-1, 1:-1] + g[1:-1, 2:] +
            g[2:, :-2] + g[2:, 1:-1] + g[2:, 2:]) / 9.0
    res = g[1:-1, 1:-1] - blur
    res = res - res.mean()
    var = res.var() + 1e-8

    a = res[:-1, :-1]
    rh = float(abs((a * res[:-1, 1:]).mean()) / var)
    rv = float(abs((a * res[1:, :-1]).mean()) / var)
    rd = float(abs((a * res[1:, 1:]).mean()) / var)
    avg_corr = (rh + rv + rd) / 3.0
    return float(np.clip(avg_corr * 3.0, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# combiner
# --------------------------------------------------------------------------- #


SIGNAL_NAMES = ("fft", "ela", "benford", "cfa", "wavelet", "jpeg_ghost", "noise")
DEFAULT_WEIGHTS = (0.20, 0.10, 0.20, 0.15, 0.10, 0.10, 0.15)


def all_scores(img: Image.Image) -> dict[str, float]:
    return {
        "fft": fft_score(img),
        "ela": ela_score(img),
        "benford": benford_score(img),
        "cfa": cfa_score(img),
        "wavelet": wavelet_score(img),
        "jpeg_ghost": jpeg_ghost_score(img),
        "noise": noise_score(img),
    }


def detect(img: Image.Image, weights: tuple[float, ...] = DEFAULT_WEIGHTS) -> dict:
    s = all_scores(img)
    combined = float(sum(w * s[name] for w, name in zip(weights, SIGNAL_NAMES)))
    return {**s, "combined": combined, "verdict": "synthetic" if combined > 0.5 else "real"}
