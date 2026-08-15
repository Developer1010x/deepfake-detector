"""Spatial explanation maps: does the picture say what the number says?

An overlay is a claim about *where* a signal fired, so the invariant that
matters is the boring one — a field with nothing in it must not look like a
finding. The JPEG-ghost map is the case that got this wrong: its colour encodes
a quality label rather than an anomaly, so a single-source photograph (uniformly
"quality 90", the most innocent result the signal can return) rendered as a
solid block of the brightest colour in the ramp and hid the photograph
underneath it. Opacity now tracks *disagreement with the dominant history*,
which is the thing the signal actually measures.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import explain

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "fake" / "alpha-dog.jpg"


def _decode(data_uri: str) -> np.ndarray:
    assert data_uri.startswith("data:image/png;base64,")
    raw = base64.b64decode(data_uri.split(",", 1)[1])
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.float64)


def _photo(seed=0, size=384) -> Image.Image:
    """A compressible pseudo-photograph (smooth, not white noise)."""
    rng = np.random.default_rng(seed)
    a = np.cumsum(rng.standard_normal((size, size, 3)), axis=0) * 6 + 128
    a = np.clip(a + np.linspace(0, 40, size)[None, :, None], 0, 255)
    return Image.fromarray(a.astype(np.uint8))


def _jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality)
    buf.seek(0)
    out = Image.open(buf)
    out.load()
    return out


def _spliced(seed=0) -> Image.Image:
    """One frame, two compression histories — what a ghost map should find."""
    base = _photo(seed)
    patch = _jpeg(base.crop((96, 96, 288, 288)), 40)
    out = base.copy()
    out.paste(patch, (96, 96))
    return _jpeg(out, 92)


# --------------------------------------------------------------------------- #
# colour ramp
# --------------------------------------------------------------------------- #


def test_colormap_is_in_range_and_ordered():
    v = np.linspace(0.0, 1.0, 64)
    rgb = explain.colormap(v)
    assert rgb.shape == (64, 3) and rgb.dtype == np.uint8
    # inferno runs dark -> bright: total luminance increases end to end.
    assert rgb[0].sum() < rgb[-1].sum()


def test_colormap_survives_nan_and_out_of_range():
    rgb = explain.colormap(np.array([np.nan, -3.0, 4.0]))
    assert np.isfinite(rgb).all()
    assert (rgb >= 0).all() and (rgb <= 255).all()


# --------------------------------------------------------------------------- #
# "nothing found" must look like nothing found
# --------------------------------------------------------------------------- #


def test_an_empty_field_renders_as_the_plain_photo():
    img = _photo(1, size=128)
    flat = np.zeros((8, 8))
    out = _decode(explain.render_overlay(img, flat, smooth=False))
    base = explain._base(img, explain._target_size(img))
    # alpha floor is ALPHA_MIN, so the tint can only move a pixel that far.
    assert np.abs(out - base).mean() < explain.ALPHA_MIN * 255


def _loudness(m, img) -> float:
    """How far the overlay moves the picture away from the plain photo."""
    return float(np.abs(_decode(m["image"])
                        - explain._base(img, explain._target_size(img))).mean())


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample image not in the tree")
def test_a_single_source_photo_gets_a_quiet_ghost_map():
    """The regression this file exists for.

    ``samples/fake/alpha-dog.jpg`` has one compression history: every block
    picks quality 90, so the field is uniform and the correct rendering is "here
    is your photo, nothing to see". Keying opacity to the index instead of to
    the deviation painted it solid pale yellow at 0.88 alpha.
    """
    img = Image.open(SAMPLE)
    m = explain.jpeg_ghost_overlay(img)
    assert m is not None and m["flat"] is True and m["spread"] == 0.0
    assert _loudness(m, img) < 10.0


def test_a_spliced_photo_lights_up_more_than_a_clean_one():
    clean, spliced = _jpeg(_photo(3), 92), _spliced(3)
    m_clean = explain.jpeg_ghost_overlay(clean)
    m_spliced = explain.jpeg_ghost_overlay(spliced)
    assert m_spliced["spread"] > m_clean["spread"]
    assert _loudness(m_spliced, spliced) > _loudness(m_clean, clean)


# --------------------------------------------------------------------------- #
# the bundle handed to the API
# --------------------------------------------------------------------------- #


def test_explanation_maps_are_complete_and_self_describing():
    maps = explain.explanation_maps(_jpeg(_photo(4), 88))
    assert [m["key"] for m in maps] == ["ela", "jpeg_ghost", "block_hetero"]
    for m in maps:
        assert m["title"] and m["caption"] and m["unit"]
        assert len(m["range"]) == 2
        arr = _decode(m["image"])
        assert max(arr.shape[:2]) <= explain.MAX_SIDE


def test_maps_that_cannot_be_computed_are_omitted_not_faked():
    tiny = Image.new("RGB", (12, 12), (90, 90, 90))
    keys = [m["key"] for m in explain.explanation_maps(tiny)]
    assert "jpeg_ghost" not in keys           # needs at least 2 blocks
    assert all(k in ("ela", "block_hetero") for k in keys)


def test_save_explanations_writes_one_png_per_map(tmp_path):
    written = explain.save_explanations(_jpeg(_photo(5), 90), tmp_path / "maps")
    assert [p.name for p in written] == ["ela.png", "jpeg_ghost.png",
                                         "block_hetero.png"]
    for p in written:
        assert p.exists() and Image.open(p).format == "PNG"
