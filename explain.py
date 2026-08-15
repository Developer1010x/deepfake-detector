"""Spatial explanation maps — *where* in the image each forensic signal fired.

Three of the seven pixel signals are computed from a full two-dimensional field
and then collapsed to a single scalar before anybody sees them:

===========================  =========================================  ==============
signal                       field it computes                          what is kept
===========================  =========================================  ==============
``ela``                      per-pixel re-save error, ``(H, W, 3)``     one ratio
``jpeg_ghost``               per-block best-fit quality, ``(H/16, W/16)``  its std
``block_hetero``             per-tile FFT score, ``(n, n)``             its std
===========================  =========================================  ==============

Throwing the field away is what turns an explainable detector into a table of
numbers. This module renders each field as a colourised heat-map blended over a
desaturated copy of the input, so ``jpeg_ghost: 0.414`` becomes a picture in
which the inconsistent region is visible.

Everything here is pure numpy + Pillow — the same from-scratch constraint the
rest of the signal bank follows. Maps are returned as ``data:image/png;base64``
URIs so the Flask API can hand them straight to an ``<img>`` tag with no
temporary files and no static-file plumbing.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image

from detector import JPEG_GHOST_QUALITIES, ela_map, jpeg_ghost_field
from patterns import block_fft_field

#: Longest side of a rendered map, in pixels. Keeps the base64 payload small
#: enough to sit inside a JSON response without a second round-trip.
MAX_SIDE = 384

#: Overlay opacity is *modulated by the signal itself* — cold regions stay
#: nearly transparent so the photograph shows through, hot regions glow. A flat
#: field therefore renders as a plain desaturated photo, which is exactly the
#: right reading: "this signal found nothing spatially anomalous".
ALPHA_MIN, ALPHA_MAX = 0.06, 0.88

# Anchor colours of the `inferno` sequential map, sampled at 0, .25, .5, .75, 1.
# Hand-rolled rather than imported so the web path never pulls in matplotlib.
_INFERNO = np.array(
    [
        [0, 0, 4],
        [87, 16, 110],
        [188, 55, 84],
        [249, 142, 9],
        [252, 255, 164],
    ],
    dtype=np.float64,
)


def colormap(v: np.ndarray) -> np.ndarray:
    """Map a ``[0, 1]`` array to RGB uint8 via piecewise-linear `inferno`."""
    v = np.clip(np.nan_to_num(v, nan=0.0), 0.0, 1.0)
    pos = v * (len(_INFERNO) - 1)
    lo = np.clip(np.floor(pos).astype(int), 0, len(_INFERNO) - 2)
    frac = (pos - lo)[..., None]
    return np.clip(
        _INFERNO[lo] * (1.0 - frac) + _INFERNO[lo + 1] * frac, 0, 255
    ).astype(np.uint8)


def _normalise(
    field: np.ndarray,
    domain: tuple[float, float] | None = None,
    clip_pct: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """Scale a field to [0, 1], returning it with the (lo, hi) actually used.

    ``domain`` forces a fixed scale — used for the JPEG-ghost index, whose
    meaningful range is the quality list, not whatever this one image happened to
    span. ``clip_pct`` caps the top end at a percentile so a handful of hot
    pixels cannot wash the rest of an ELA map out to black.
    """
    field = np.nan_to_num(field.astype(np.float64), nan=0.0)
    if domain is not None:
        lo, hi = float(domain[0]), float(domain[1])
    else:
        lo = float(field.min())
        hi = float(np.percentile(field, clip_pct)) if clip_pct else float(field.max())
    if hi - lo < 1e-9:
        return np.zeros_like(field), lo, hi
    return np.clip((field - lo) / (hi - lo), 0.0, 1.0), lo, hi


def _base(img: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """Desaturated, dimmed copy of the input to blend the heat-map over."""
    g = np.asarray(img.convert("L").resize(size, Image.BILINEAR), dtype=np.float64)
    g = 0.30 + 0.55 * (g / 255.0)          # compress toward mid-grey
    return np.repeat((g * 255.0)[:, :, None], 3, axis=2)


def _target_size(img: Image.Image) -> tuple[int, int]:
    w, h = img.size
    scale = MAX_SIDE / float(max(w, h))
    if scale >= 1.0:
        return w, h
    return max(1, int(round(w * scale))), max(1, int(round(h * scale)))


def _to_data_uri(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def render_overlay(img: Image.Image, unit: np.ndarray, smooth: bool,
                   alpha_unit: np.ndarray | None = None) -> str:
    """Blend a ``[0, 1]``-scaled field over a desaturated ``img`` as a PNG data URI.

    ``smooth=False`` keeps block boundaries crisp (nearest-neighbour upsampling),
    which is what you want for the coarse 16-px JPEG-ghost grid: the blocks *are*
    the measurement, and interpolating them would imply spatial precision the
    signal does not have.

    ``alpha_unit`` separates *what colour* a cell gets from *how loudly* it is
    shown. They coincide for fields whose magnitude is the anomaly (ELA), but not
    for the JPEG-ghost index: there the colour is a quality label, and a field
    that is uniformly "quality 90" is the most innocent result there is — it must
    render as a quiet photo, not as a solid block of the brightest colour in the
    ramp.
    """
    size = _target_size(img)
    resample = Image.BILINEAR if smooth else Image.NEAREST
    heat = np.asarray(
        Image.fromarray(colormap(unit), mode="RGB").resize(size, resample),
        dtype=np.float64,
    )
    a = unit if alpha_unit is None else np.clip(alpha_unit, 0.0, 1.0)
    alpha = np.asarray(
        Image.fromarray((a * 255.0).astype(np.uint8), mode="L").resize(size, resample),
        dtype=np.float64,
    )[:, :, None] / 255.0
    alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * alpha
    blended = _base(img, size) * (1.0 - alpha) + heat * alpha
    return _to_data_uri(blended)


# --------------------------------------------------------------------------- #
# the three maps
# --------------------------------------------------------------------------- #


def _spread(field: np.ndarray) -> float:
    return round(float(np.std(field)), 4)


def ela_overlay(img: Image.Image) -> dict | None:
    field = ela_map(img).mean(axis=2)
    if field.size == 0:
        return None
    # Clip at the 99th percentile: a few saturated edge pixels otherwise flatten
    # the whole map to the bottom of the colour ramp.
    unit, lo, hi = _normalise(field, clip_pct=99.0)
    return {
        "key": "ela",
        "title": "Error Level Analysis",
        "caption": "Per-pixel error from a quality-90 re-save. Bright regions "
                   "have a different compression history from the rest of the "
                   "frame — the classic splice / inpaint tell.",
        "image": render_overlay(img, unit, smooth=True),
        "range": [round(lo, 3), round(hi, 3)],
        "spread": _spread(field),
        "unit": "abs. RGB difference (0-255), scaled to the 99th percentile",
    }


def jpeg_ghost_overlay(img: Image.Image) -> dict | None:
    field = jpeg_ghost_field(img)
    if field is None:
        return None
    # Fixed domain: the meaningful scale is the quality list, not this image's
    # own span, so the same colour means the same quality in every image.
    unit, _, _ = _normalise(field, domain=(0.0, len(JPEG_GHOST_QUALITIES) - 1.0))
    quals = "/".join(str(q) for q in JPEG_GHOST_QUALITIES)
    flat = bool(np.ptp(field) == 0)
    # Opacity is *disagreement with the dominant history*, in quality steps —
    # not the index itself. A single-source image is uniform at some quality and
    # must therefore render as a plain photo; only blocks that pick a different
    # quality from the rest of the frame are ghosts, and only they should glow.
    dominant = float(np.bincount(field.astype(np.int64).ravel()).argmax())
    deviation = np.abs(field - dominant) / max(1.0, len(JPEG_GHOST_QUALITIES) - 1.0)
    return {
        "key": "jpeg_ghost",
        "title": "JPEG ghost (best-fit quality)",
        "caption": f"For each 16x16 block, which of quality {quals} minimises the "
                   "re-encoding error. A single-source photo is one flat colour; a "
                   "mixed-history image is patchy (Farid 2009). Blocks are lit "
                   "in proportion to how far their best-fit quality is from the "
                   "rest of the frame."
                   + (" This field is uniform — no ghost." if flat else ""),
        "image": render_overlay(img, unit, smooth=False, alpha_unit=deviation),
        "range": [int(JPEG_GHOST_QUALITIES[0]), int(JPEG_GHOST_QUALITIES[-1])],
        "spread": _spread(field),
        "unit": "JPEG quality of best fit",
        "flat": flat,
    }


def block_fft_overlay(img: Image.Image, n_blocks: int = 8) -> dict | None:
    field = block_fft_field(img, n_blocks=n_blocks)
    if field is None:                       # image too small for an 8x8 grid
        field, n_blocks = block_fft_field(img, n_blocks=4), 4
    if field is None:
        return None
    unit, lo, hi = _normalise(field)
    return {
        "key": "block_hetero",
        "title": f"Spectral heterogeneity ({n_blocks}x{n_blocks} tiles)",
        "caption": "FFT spectral-residual score computed independently per tile, "
                   "scaled within this image. One lens and one sensor give a flat "
                   "field; tiled or patch-denoised generation gives hot spots.",
        "image": render_overlay(img, unit, smooth=False),
        "range": [round(lo, 3), round(hi, 3)],
        "spread": _spread(field),
        "unit": "fft signal (0-1), scaled within this image",
    }


def explanation_maps(img: Image.Image) -> list[dict]:
    """All available spatial explanations for one image, cheapest first.

    Each entry is ``{key, title, caption, image (data URI), range, unit}``. Maps
    that cannot be computed for the given size are omitted rather than faked.
    """
    maps = [ela_overlay(img), jpeg_ghost_overlay(img), block_fft_overlay(img)]
    return [m for m in maps if m is not None]


def save_explanations(img: Image.Image, out_dir: str | Path) -> list[Path]:
    """Write the maps as ``<out_dir>/<key>.png`` and return the written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for m in explanation_maps(img):
        raw = base64.b64decode(m["image"].split(",", 1)[1])
        p = out_dir / f"{m['key']}.png"
        p.write_bytes(raw)
        written.append(p)
    return written
