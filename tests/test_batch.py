"""Folder triage: collection, ranking, the CSV contract and error accounting.

``cli.py batch`` is the forensic path — point it at a directory nobody has
labelled and get an ordered worklist. The invariants that make such a list
trustworthy are cheap to state and easy to break:

  * every file handed in comes back out, including the ones that failed to
    decode (a triage tool that silently drops files is worse than none);
  * the order is by descending P(synthetic), with failures last;
  * the CSV has a fixed schema and no ragged rows;
  * the probability, verdict and abstention band match what the single-file
    path (``cli.py image``) would print for the same file.

All of this runs on the training-free classical combiner (``model=None``), so
the suite needs nothing beyond numpy + Pillow.
"""

from __future__ import annotations

import csv
import wave

import numpy as np
from PIL import Image

import batch
import pipeline


def _image(path, size=96, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    Image.fromarray(a).save(path)


def _wav(path, seconds=1.0, sr=16000):
    t = np.arange(int(sr * seconds)) / sr
    x = (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(x.tobytes())


# --------------------------------------------------------------------------- #
# collection
# --------------------------------------------------------------------------- #


def test_media_kind_classifies_by_extension():
    assert pipeline.media_kind("a.JPG") == "image"
    assert pipeline.media_kind("a.wav") == "audio"
    assert pipeline.media_kind("a.mkv") == "video"
    assert pipeline.media_kind("notes.txt") is None


def test_collect_finds_media_and_ignores_everything_else(tmp_path):
    _image(tmp_path / "a.png")
    _wav(tmp_path / "b.wav")
    (tmp_path / "notes.txt").write_text("not media")
    (tmp_path / "clip.mp4").write_bytes(b"\x00" * 16)
    found = batch.collect(tmp_path)
    assert [p.name for p in found] == ["a.png", "b.wav"]
    assert batch.count_skipped_videos(tmp_path) == 1


def test_collect_respects_recursion_and_accepts_a_single_file(tmp_path):
    (tmp_path / "sub").mkdir()
    _image(tmp_path / "top.png")
    _image(tmp_path / "sub" / "deep.png", seed=1)
    assert len(batch.collect(tmp_path, recursive=True)) == 2
    assert [p.name for p in batch.collect(tmp_path, recursive=False)] == ["top.png"]
    assert batch.collect(tmp_path / "top.png") == [tmp_path / "top.png"]
    assert batch.collect(tmp_path / "sub" / "deep.png", kinds=("audio",)) == []


# --------------------------------------------------------------------------- #
# ranking
# --------------------------------------------------------------------------- #


def test_triage_ranks_by_descending_probability_and_keeps_every_file(tmp_path):
    for i in range(3):
        _image(tmp_path / f"img{i}.png", seed=i)
    _wav(tmp_path / "clip.wav")
    (tmp_path / "broken.png").write_text("this is not a PNG")

    rows = batch.triage(batch.collect(tmp_path), model=None)

    assert len(rows) == 5                       # nothing is silently dropped
    scored = [r for r in rows if not r["error"]]
    failed = [r for r in rows if r["error"]]
    assert len(failed) == 1 and failed[0]["path"].endswith("broken.png")
    assert rows[-1] is failed[0]                # failures sort last
    probs = [r["prob_synthetic"] for r in scored]
    assert probs == sorted(probs, reverse=True)
    assert [r["rank"] for r in scored] == list(range(1, len(scored) + 1))
    assert failed[0]["rank"] == ""              # an unreadable file has no rank
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_every_row_carries_the_full_schema(tmp_path):
    _image(tmp_path / "a.png")
    _wav(tmp_path / "b.wav")
    for row in batch.triage(batch.collect(tmp_path), model=None):
        assert set(row) == set(batch.CSV_COLUMNS)


def test_the_verdict_matches_the_single_file_pipeline(tmp_path):
    """Batch and `cli.py image` must not disagree about the same file.

    Both sides resolve the model the same way (``model="auto"``), so this holds
    whether or not the fusion artifact is loadable in this environment — which
    is the invariant that matters: the two paths never disagree *with each
    other*.
    """
    _image(tmp_path / "a.png", size=128, seed=7)
    row = batch.triage(batch.collect(tmp_path))[0]
    single = pipeline.analyze_path(tmp_path / "a.png")
    assert row["prob_synthetic"] == single["prob_synthetic"]
    assert row["verdict"] == single["final_verdict"]
    assert row["confidence"] == single["final_confidence"]
    assert row["classical"] == single["classical_combined"]


def test_the_top_signal_is_the_largest_weighted_contribution(tmp_path):
    from detector import DEFAULT_WEIGHTS, SIGNAL_NAMES

    _image(tmp_path / "a.png", size=128, seed=3)
    row = batch.triage(batch.collect(tmp_path), model=None)[0]
    signals = pipeline.analyze_path(tmp_path / "a.png")["signals"]
    expected = max(zip(DEFAULT_WEIGHTS, SIGNAL_NAMES),
                   key=lambda ws: ws[0] * signals[ws[1]])[1]
    assert row["top_signal"] == expected


def test_skipping_the_watermark_scan_is_not_reported_as_a_clean_scan(tmp_path):
    _image(tmp_path / "a.png")
    scanned = batch.triage(batch.collect(tmp_path), model=None, watermark=True)[0]
    skipped = batch.triage(batch.collect(tmp_path), model=None, watermark=False)[0]
    assert scanned["watermark_score"] != ""
    assert skipped["watermark_score"] == ""      # renders as "-", not 0.00


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #


def test_write_csv_round_trips(tmp_path):
    _image(tmp_path / "a.png")
    _wav(tmp_path / "b.wav")
    rows = batch.triage(batch.collect(tmp_path), model=None)
    out = tmp_path / "out" / "triage.csv"
    batch.write_csv(rows, out)

    back = list(csv.DictReader(out.open()))
    assert len(back) == len(rows)
    assert list(back[0]) == list(batch.CSV_COLUMNS)
    assert float(back[0]["prob_synthetic"]) == rows[0]["prob_synthetic"]


def test_summarise_accounts_for_every_row(tmp_path):
    _image(tmp_path / "a.png")
    (tmp_path / "broken.png").write_text("nope")
    rows = batch.triage(batch.collect(tmp_path), model=None)
    s = batch.summarise(rows)
    assert s["total"] == len(rows)
    assert s["flagged"] + s["inconclusive"] + s["real"] + s["errors"] == s["total"]


def test_format_table_shows_the_header_and_honours_top(tmp_path):
    for i in range(4):
        _image(tmp_path / f"img{i}.png", seed=i)
    rows = batch.triage(batch.collect(tmp_path), model=None)
    text = batch.format_table(rows, top=2)
    assert "P(syn)" in text and "verdict" in text
    body = text.splitlines()[2:]
    assert len(body) == 2
