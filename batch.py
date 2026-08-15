"""Batch triage: rank a folder of *unlabelled* media by P(synthetic).

The evaluation path (``evaluate.py``) answers a research question — "how well
does this detector separate two labelled piles?" — and needs ``--real`` and
``--fake`` directories to do it. The forensic question is the other one: *here
are 500 files nobody has looked at; which twenty should a human open first?*
That is what this module answers, and it is the use case in which the project's
own honest caveat ("the strength is ranking, not thresholding") stops being an
apology and becomes the feature.

Two things make this more than a ``for`` loop over the single-file path:

  * **Batched feature extraction.** Images are processed in chunks of
    :data:`CHUNK`, so the 13-d hand-crafted block and the 1024-d deep embedding
    are each computed once per chunk through the vectorised
    :meth:`fusion.FusionModel.predict_proba` path, rather than one forward pass
    per file through ``predict_one``.
  * **One ranking across modalities.** Images and audio clips land in the same
    ordered table with the same calibrated probability, the same abstention band
    (:data:`pipeline.INCONCLUSIVE_MARGIN`) and the same watermark override
    (:func:`pipeline.final_verdict`), so a mixed evidence folder can be triaged
    in one pass.

Every row also carries *why*: ``top_signal`` is the forensic signal making the
largest weighted contribution to that file's classical score, which is the first
thing an analyst wants after the ranking itself.

Videos are deliberately not scored here — a per-frame sweep costs seconds per
file and would dominate the run — they are counted and reported as skipped.

Used by ``cli.py batch``; importable on its own::

    from batch import collect, triage, write_csv
    rows = triage(collect("evidence/"))
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import pipeline
from detector import DEFAULT_WEIGHTS, SIGNAL_NAMES

#: Images decoded and scored per feature-extraction call. Large enough to
#: amortise the per-call overhead of the deep stream, small enough to bound peak
#: memory: a chunk of full-resolution 24-megapixel photographs is ~570 MB
#: decoded, and that is the number this constant is really trading against.
CHUNK = 8

#: CSV schema. Stable column order — downstream scripts index by name, but a
#: fixed order keeps diffs of committed triage runs readable.
CSV_COLUMNS = (
    "rank", "path", "kind", "prob_synthetic", "verdict", "confidence",
    "decision_source", "classical", "top_signal", "top_signal_value",
    "watermark_score", "watermark_verdict", "duration_s", "error",
)

#: Modalities `triage` can score. Video is decoded per frame by `cli.py video`.
SCORABLE_KINDS = ("image", "audio")


def collect(root: str | Path, recursive: bool = True,
            kinds: tuple[str, ...] = SCORABLE_KINDS) -> list[Path]:
    """Every readable media file under ``root``, sorted, filtered by modality.

    Accepts a single file as well as a directory, so ``cli.py batch one.jpg``
    behaves sensibly instead of returning nothing.
    """
    root = Path(root)
    if root.is_file():
        return [root] if pipeline.media_kind(root) in kinds else []
    it = root.rglob("*") if recursive else root.glob("*")
    return sorted(p for p in it if p.is_file() and pipeline.media_kind(p) in kinds)


def count_skipped_videos(root: str | Path, recursive: bool = True) -> int:
    """How many video files were passed over, so the summary can say so."""
    return len(collect(root, recursive=recursive, kinds=("video",)))


def _row(path: Path, kind: str, **kw) -> dict:
    """A CSV row with every column present, so writers never see a gap."""
    row = {c: "" for c in CSV_COLUMNS}
    row.update(path=str(path), kind=kind, **kw)
    return row


def _top_signal(values: dict[str, float], names: tuple[str, ...],
                weights: tuple[float, ...]) -> tuple[str, float]:
    """The signal contributing most to the weighted classical score.

    Weighted, not raw: a signal that sits at 0.9 with weight 0.05 explains less
    of the verdict than one at 0.6 with weight 0.20, and the analyst wants the
    latter.
    """
    contrib = [(w * float(values[n]), n) for w, n in zip(weights, names)]
    _, name = max(contrib)
    return name, round(float(values[name]), 4)


def _image_rows(paths: list[Path], model, watermark: bool, chunk: int,
                progress=None, done: int = 0, total: int = 0) -> list[dict]:
    from fusion import deep_matrix, handcrafted_matrix
    from watermark import inspect as watermark_inspect

    weights = np.asarray(DEFAULT_WEIGHTS, dtype=np.float64)
    n_sig = len(SIGNAL_NAMES)
    rows: list[dict] = []

    for start in range(0, len(paths), chunk):
        group = paths[start:start + chunk]
        imgs: list[Image.Image] = []
        ok: list[Path] = []
        for p in group:
            try:
                im = Image.open(p)
                im.load()
                imgs.append(im.convert("RGB"))
                ok.append(p)
            except Exception as exc:  # noqa: BLE001 - a corrupt file is a result
                rows.append(_row(p, "image", error=f"{type(exc).__name__}: {exc}"))
            done += 1
            if progress:
                progress(done, total, p)
        if not imgs:
            continue

        X_hand = handcrafted_matrix(imgs)
        classical = X_hand[:, :n_sig] @ weights
        probs, source = classical, "classical"
        if model is not None:
            try:
                X_deep = (deep_matrix(imgs, spectral=model.config.spectral)
                          if model.uses_deep else None)
                probs = model.predict_proba(
                    X_hand if model.uses_handcrafted else None, X_deep)
                source = "fusion"
            except Exception as exc:  # noqa: BLE001 - degrade, but say so
                print(f"  WARNING: fusion model failed on this chunk "
                      f"({type(exc).__name__}: {exc}); using the classical "
                      f"combiner for these {len(imgs)} files", file=sys.stderr)
                probs, source = classical, "classical"

        for p, vec, prob, cls in zip(ok, X_hand, probs, classical):
            # An empty watermark score means "not scanned" (--no-watermark) and
            # renders as "-", which is not the same claim as "scanned, nothing
            # found" (0.0). The verdict fusion treats both as no evidence.
            wm: dict = {"score": "", "verdict": "", "provenance": False}
            if watermark:
                try:
                    wm = watermark_inspect(p)
                except Exception:  # noqa: BLE001 - provenance is best-effort
                    wm = {"score": 0.0, "verdict": "scan failed", "provenance": False}
            prob = float(prob)
            band = pipeline.confidence_band(prob)
            wm_score = float(wm["score"]) if wm["score"] != "" else 0.0
            verdict, conf = pipeline.final_verdict(prob, band, wm["provenance"])
            name, value = _top_signal(
                dict(zip(SIGNAL_NAMES, vec[:n_sig])), SIGNAL_NAMES, DEFAULT_WEIGHTS)
            rows.append(_row(
                p, "image",
                prob_synthetic=round(prob, 4), verdict=verdict, confidence=conf,
                decision_source=source, classical=round(float(cls), 4),
                top_signal=name, top_signal_value=value,
                watermark_score="" if wm["score"] == "" else round(wm_score, 3),
                watermark_verdict=wm.get("verdict", ""),
            ))
    return rows


def _audio_rows(paths: list[Path], progress=None, done: int = 0,
                total: int = 0) -> list[dict]:
    import audio_detect as ad

    rows: list[dict] = []
    for p in paths:
        try:
            res = pipeline.analyze_audio_path(p)
        except Exception as exc:  # noqa: BLE001
            res = {"error": f"{type(exc).__name__}: {exc}"}
        done += 1
        if progress:
            progress(done, total, p)
        if "error" in res:
            rows.append(_row(p, "audio", error=res["error"]))
            continue
        name, value = _top_signal(res["signals"], ad.SIGNAL_NAMES, ad.DEFAULT_WEIGHTS)
        rows.append(_row(
            p, "audio",
            prob_synthetic=res["prob_synthetic"], verdict=res["final_verdict"],
            confidence=res["final_confidence"],
            decision_source=res["decision_source"],
            classical=res["classical_combined"],
            top_signal=name, top_signal_value=value,
            duration_s=res["duration_s"],
        ))
    return rows


def triage(paths: list[Path], model="auto", watermark: bool = True,
           chunk: int = CHUNK, progress=None) -> list[dict]:
    """Score every path and return the rows ranked by P(synthetic), highest first.

    ``model="auto"`` loads ``models/fusion.joblib`` once for the whole run;
    pass ``model=None`` to force the training-free classical combiner (which is
    also what happens automatically when the artifact is absent or unusable —
    see :func:`pipeline.load_error`). Files that cannot be decoded are not
    dropped: they come back with an ``error`` column and sort to the bottom, so
    a triage run always accounts for every file it was given.
    """
    if model == "auto":
        model = pipeline.load_model()

    paths = list(paths)
    total = len(paths)
    images = [p for p in paths if pipeline.media_kind(p) == "image"]
    audio = [p for p in paths if pipeline.media_kind(p) == "audio"]

    rows = _image_rows(images, model, watermark, chunk, progress, 0, total)
    rows += _audio_rows(audio, progress, len(images), total)

    # Errors last; everything else by descending probability.
    rows.sort(key=lambda r: (bool(r["error"]),
                             -(r["prob_synthetic"] if r["prob_synthetic"] != "" else 0.0)))
    for i, r in enumerate(rows, 1):
        r["rank"] = "" if r["error"] else i
    return rows


def write_csv(rows: list[dict], dest: str | Path) -> None:
    """Write the ranked table as CSV. ``dest="-"`` writes to stdout."""
    if str(dest) == "-":
        w = csv.DictWriter(sys.stdout, fieldnames=list(CSV_COLUMNS))
        w.writeheader()
        w.writerows(rows)
        return
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        w.writeheader()
        w.writerows(rows)


def _short(path: str, width: int) -> str:
    """Right-anchored truncation: the filename matters more than the prefix."""
    return path if len(path) <= width else "..." + path[-(width - 3):]


#: Column widths for :func:`format_table`. VERDICT fits the longest verdict the
#: pipeline can emit ("inconclusive - insufficient evidence", 36) and SIGNAL the
#: longest signal name ("noisefloor_regularity", 21).
VERDICT_W, SIGNAL_W, PATH_W = 36, 21, 36


def format_table(rows: list[dict], top: int = 20, path_width: int = PATH_W) -> str:
    """Render the ranking as an aligned fixed-width table."""
    shown = [r for r in rows if not r["error"]][:top]
    errors = [r for r in rows if r["error"]]
    head = (f"{'#':>3}  {'P(syn)':>6}  {'kind':<5}  {'verdict':<{VERDICT_W}}  "
            f"{'top signal':<{SIGNAL_W}}  {'wm':>4}  file")
    lines = [head, "-" * len(head)]
    for r in shown:
        wm = f"{r['watermark_score']:.2f}" if r["watermark_score"] != "" else "-"
        lines.append(
            f"{r['rank']:>3}  {r['prob_synthetic']:>6.3f}  {r['kind']:<5}  "
            f"{r['verdict'][:VERDICT_W]:<{VERDICT_W}}  "
            f"{r['top_signal'][:SIGNAL_W]:<{SIGNAL_W}}  {wm:>4}  "
            f"{_short(r['path'], path_width)}"
        )
    for r in errors[:5]:
        lines.append(f"{'-':>3}  {'-':>6}  {r['kind']:<5}  "
                     f"{('ERROR: ' + r['error'])[:VERDICT_W]:<{VERDICT_W}}  "
                     f"{'-':<{SIGNAL_W}}  {'-':>4}  "
                     f"{_short(r['path'], path_width)}")
    if len(errors) > 5:
        lines.append(f"     ... and {len(errors) - 5} more unreadable files")
    return "\n".join(lines)


def summarise(rows: list[dict]) -> dict[str, int]:
    """Counts by outcome, for the one-line summary under the table."""
    out = {"total": len(rows), "flagged": 0, "inconclusive": 0, "real": 0,
           "watermarked": 0, "errors": 0}
    for r in rows:
        if r["error"]:
            out["errors"] += 1
        elif r["verdict"].startswith("AI-generated (provenance"):
            out["watermarked"] += 1
            out["flagged"] += 1
        elif r["verdict"].startswith("inconclusive"):
            out["inconclusive"] += 1
        elif r["prob_synthetic"] > 0.5:
            out["flagged"] += 1
        else:
            out["real"] += 1
    return out
