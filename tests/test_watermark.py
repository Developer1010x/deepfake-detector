"""Watermark scanner: does it fire on real markers and stay quiet otherwise?

A provenance hit is a *hard override* — it replaces the calibrated probability
with "AI-generated, high confidence" (:func:`pipeline.final_verdict`). A false
positive is therefore the most damaging single failure this project can produce,
which is why both directions are pinned here:

  * markers a generator actually writes (C2PA, a diffusion ``parameters`` chunk,
    an EXIF generator name, a COM segment) must be detected;
  * things ordinary photographs and ordinary editors do — matching the libjpeg
    default quantization tables, carrying a ``Software`` tag, having heavy
    mid-band texture, or containing a generator name by chance inside the
    compressed scan data — must not be reported as a watermark.

The last one is not hypothetical: sweeping whole-file bytes found ``a1111``
inside the entropy stream of a stock photograph, and the mid-band kurtosis
heuristic fired on 21 of 160 real photographs before this was calibrated.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, PngImagePlugin

import watermark


def _photo(rng, size=(160, 160)) -> Image.Image:
    """A texture-rich pseudo-photograph — the kind that trips heuristics."""
    a = rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(a)


def _write_png(tmp_path, name, text: dict | None = None) -> str:
    img = Image.new("RGB", (80, 80), (120, 130, 140))
    info = PngImagePlugin.PngInfo()
    for k, v in (text or {}).items():
        info.add_text(k, v)
    p = tmp_path / name
    img.save(p, "PNG", pnginfo=info)
    return str(p)


# --------------------------------------------------------------------------- #
# metadata scoping: the entropy stream is not metadata
# --------------------------------------------------------------------------- #


def test_metadata_bytes_excludes_the_jpeg_scan_data():
    rng = np.random.default_rng(0)
    buf = io.BytesIO()
    _photo(rng).save(buf, "JPEG", quality=95, comment=b"hello from COM")
    raw = buf.getvalue()
    meta = watermark.metadata_bytes(raw)
    assert b"hello from COM" in meta          # the COM segment survives
    assert len(meta) < len(raw) / 4           # the scan data does not


def test_metadata_bytes_excludes_png_idat():
    rng = np.random.default_rng(1)
    buf = io.BytesIO()
    _photo(rng).save(buf, "PNG")
    raw = buf.getvalue()
    meta = watermark.metadata_bytes(raw)
    assert b"IDAT" not in meta
    assert len(meta) < len(raw) / 4


def test_a_generator_name_in_the_scan_data_is_not_a_hit(tmp_path):
    """The needle is in the compressed pixels, not in any metadata segment."""
    rng = np.random.default_rng(2)
    buf = io.BytesIO()
    _photo(rng).save(buf, "JPEG", quality=90)
    raw = bytearray(buf.getvalue())
    sos = raw.find(b"\xff\xda")
    assert sos > 0
    raw[sos + 40:sos + 45] = b"a1111"          # splice it into the scan data
    p = tmp_path / "spliced.jpg"
    p.write_bytes(bytes(raw))
    res = watermark.inspect(p)
    assert res["keyword_hits"] == []
    assert res["provenance"] is False


# --------------------------------------------------------------------------- #
# true positives: markers a generator writes
# --------------------------------------------------------------------------- #


def test_stable_diffusion_parameters_chunk_is_provenance(tmp_path):
    p = _write_png(tmp_path, "sd.png",
                   {"parameters": "a cat, Steps: 30, Sampler: DPM++ 2M, "
                                  "CFG scale: 7, Model hash: abc123"})
    res = watermark.inspect(p)
    assert res["provenance"] is True
    assert res["verdict"] == "AI provenance detected"
    assert "parameters" in res["png_ai_chunks"]


def test_c2pa_manifest_in_a_metadata_segment_is_provenance(tmp_path):
    rng = np.random.default_rng(3)
    buf = io.BytesIO()
    _photo(rng).save(buf, "JPEG", quality=92, comment=b"jumbf c2pa manifest")
    p = tmp_path / "c2pa.jpg"
    p.write_bytes(buf.getvalue())
    res = watermark.inspect(p)
    assert res["c2pa"] is True
    assert res["provenance"] is True


def test_exif_software_naming_a_generator_is_provenance(tmp_path):
    img = Image.new("RGB", (80, 80), (10, 20, 30))
    exif = img.getexif()
    exif[0x0131] = "Stable Diffusion WebUI (automatic1111)"   # Software
    p = tmp_path / "exif.jpg"
    img.save(p, "JPEG", exif=exif)
    res = watermark.inspect(p)
    assert res["provenance"] is True
    assert res["exif_ai_fields"]


# --------------------------------------------------------------------------- #
# false positives: what ordinary files do
# --------------------------------------------------------------------------- #


def test_a_plain_software_chunk_is_not_provenance(tmp_path):
    """Every editor writes Software/Comment. Presence alone proves nothing."""
    p = _write_png(tmp_path, "gimp.png", {"Software": "GIMP 2.10.34",
                                          "Comment": "holiday photo"})
    res = watermark.inspect(p)
    assert res["provenance"] is False
    assert res["png_ai_chunks"] == {}


def test_a_software_chunk_that_names_a_generator_is_provenance(tmp_path):
    """...but the same chunk *is* evidence when its content is a generator."""
    p = _write_png(tmp_path, "sw.png", {"Software": "Midjourney v6"})
    res = watermark.inspect(p)
    assert res["provenance"] is True


def test_the_circumstantial_clues_cannot_reach_the_reporting_threshold(tmp_path):
    """QT match + a saturated spectral anomaly still is not a watermark.

    0.15 + 0.25 = 0.40 by construction, which is why neither the verdict nor
    the pipeline override can be produced by statistics alone.
    """
    assert 0.15 + 0.25 < 0.5
    rng = np.random.default_rng(4)
    p = tmp_path / "textured.jpg"
    _photo(rng, (256, 256)).save(p, "JPEG", quality=85)     # Pillow default QTs
    res = watermark.inspect(p)
    assert res["jpeg_qt"]["matched"] is True                 # the clue fires
    assert res["provenance"] is False                        # the verdict does not
    assert res["score"] < 0.5


def test_spectral_anomaly_is_zero_on_an_ordinary_photograph():
    """Calibrated against the real corpus: p95 of natural mid-band kurtosis."""
    rng = np.random.default_rng(5)
    img = Image.fromarray(
        np.clip(np.cumsum(rng.standard_normal((128, 128, 3)), axis=0) * 8 + 128,
                0, 255).astype(np.uint8))
    assert watermark._spectral_watermark_anomaly(img) < 0.5


def test_scores_and_flags_stay_in_range(tmp_path):
    p = _write_png(tmp_path, "plain.png")
    res = watermark.inspect(p)
    assert 0.0 <= res["score"] <= 1.0
    assert isinstance(res["provenance"], bool)
    assert isinstance(res["evidence"], list)
    assert res["provenance"] == bool(res["evidence"])
