"""Unified CLI for every modality.

    python3 cli.py image     PATH [--explain DIR]
    python3 cli.py audio     PATH
    python3 cli.py video     PATH [--every N]
    python3 cli.py text      PATH | -
    python3 cli.py watermark PATH
    python3 cli.py batch     DIR  [--csv out.csv] [--top N]

Model artifacts resolve relative to the source tree (see ``pipeline.HERE``), so
these work from any working directory. Probabilities within
``pipeline.INCONCLUSIVE_MARGIN`` of 0.5 are reported as *inconclusive* rather
than forced into a real/synthetic call.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from detector import SIGNAL_NAMES
from pipeline import analyze_path, model_info
from watermark import inspect as watermark_inspect


def _print_watermark(w: dict) -> None:
    print(f"  watermark  : {w['score']:.3f}  -> {w['verdict']}")
    if w.get("jpeg_qt", {}).get("matched"):
        print(f"    - circumstantial: {w['jpeg_qt']['reason']}")
    if w["c2pa"]:
        print("    - C2PA / Content Credentials manifest detected")
    if w["png_ai_chunks"]:
        for k, v in w["png_ai_chunks"].items():
            preview = v[:120].replace("\n", " ")
            print(f"    - PNG chunk '{k}': {preview}{'...' if len(v) > 120 else ''}")
    if w["exif_ai_fields"]:
        for k, v in w["exif_ai_fields"].items():
            print(f"    - EXIF '{k}': {v[:120]}")
    if w["keyword_hits"]:
        print(f"    - keyword hits: {', '.join(w['keyword_hits'][:8])}"
              + (" ..." if len(w["keyword_hits"]) > 8 else ""))
    if w["spectral_anomaly"] > 0.05:
        print(f"    - circumstantial: mid-band spectral anomaly "
              f"{w['spectral_anomaly']:.3f}")


def _report_model_state(res: dict) -> None:
    """Print the learned-model line, distinguishing 'untrained' from 'broken'."""
    from pipeline import MODEL_PATH

    mi = model_info()
    if res["fusion_prob"] is not None and mi:
        print(f"  FUSION P(synthetic): {res['fusion_prob']:.3f}  "
              f"[{mi['mode']}/{mi['classifier']} model]")
        for w in mi.get("env_warnings", []):
            print(f"  WARNING: model-card mismatch -> {w}", file=sys.stderr)
    elif res.get("model_error"):
        # A present-but-unusable artifact is a *failure*, not an untrained state.
        print(f"  ERROR: {MODEL_PATH} exists but could not be used:\n"
              f"         {res['model_error']}", file=sys.stderr)
        print("  (falling back to the classical combiner)")
    elif not MODEL_PATH.exists():
        print(f"  (no trained fusion model at {MODEL_PATH} -> using classical "
              "combiner; run train_selfsup.py)")
    else:
        print("  (fusion model loaded but returned no probability -> classical "
              "combiner)")


def analyze_image(path: Path, explain_dir: Path | None = None) -> int:
    if not path.exists():
        print(f"file not found: {path}", file=sys.stderr)
        return 1
    res = analyze_path(path)
    print(f"{path.name}")

    # classical breakdown (interpretable signals)
    for name in SIGNAL_NAMES:
        print(f"  {name:11s}: {res['signals'][name]:.3f}")
    print(f"  pattern    : {res['patterns']['pattern_score']:.3f}")
    print(f"  classical  : {res['classical_combined']:.3f}")

    _report_model_state(res)
    _print_watermark(res["watermark"])
    print(f"  => {res['final_verdict']}  (confidence: {res['final_confidence']})")

    if explain_dir is not None:
        from PIL import Image as _Image

        from explain import save_explanations

        written = save_explanations(_Image.open(path), explain_dir)
        print(f"  explanation maps -> {', '.join(str(p) for p in written)}")
    return 0


def analyze_video(path: Path, sample_every: int) -> int:
    if not path.exists():
        print(f"file not found: {path}", file=sys.stderr)
        return 1
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"could not open video: {path}", file=sys.stderr)
        return 1

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{path.name}  ({total} frames, sampling every {sample_every})")

    from pipeline import analyze_image

    probs: list[float] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % sample_every == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            r = analyze_image(Image.fromarray(rgb))
            probs.append(r["prob_synthetic"])
            print(f"  frame {idx:5d}: P(synthetic)={r['prob_synthetic']:.2f} "
                  f"[{r['decision_source']}]")
        idx += 1
    cap.release()

    if not probs:
        print("no frames sampled", file=sys.stderr)
        return 1

    arr = np.array(probs)
    avg = float(arr.mean())
    temporal_std = float(arr.std())
    print(f"\nvisual : P(synthetic) mean={avg:.3f}  temporal_std={temporal_std:.3f}")

    # Audio-track forensic analysis (synthetic-voice cue), fused with the visual.
    from audio_io import extract_audio_from_video
    from pipeline import analyze_audio as analyze_audio_wave

    audio_prob = None
    a = extract_audio_from_video(path)
    if a is not None:
        ar = analyze_audio_wave(a[1], a[0])
        audio_prob = ar["prob_synthetic"]
        print(f"audio  : P(synthetic)={audio_prob:.3f}  [{ar['decision_source']}]")
    else:
        print("audio  : no decodable audio track (or ffmpeg unavailable)")

    # High temporal variance is itself a deepfake cue (flickering artifacts).
    visual_fake = avg > 0.5 or temporal_std > 0.15
    audio_fake = audio_prob is not None and audio_prob > 0.5
    is_fake = visual_fake or audio_fake
    fused = avg if audio_prob is None else 0.6 * avg + 0.4 * audio_prob
    cue = []
    if visual_fake:
        cue.append("visual")
    if audio_fake:
        cue.append("audio")
    verdict = "synthetic / deepfake" if is_fake else "real footage"
    print(f"  -> verdict: {verdict}  (fused={fused:.3f}"
          + (f", flagged by: {'+'.join(cue)}" if cue else "") + ")")
    return 0


def analyze_audio(path: Path) -> int:
    if not path.exists():
        print(f"file not found: {path}", file=sys.stderr)
        return 1
    from pipeline import analyze_audio_path

    res = analyze_audio_path(path)
    print(f"{path.name}")
    if "error" in res:
        print(f"  error: {res['error']}", file=sys.stderr)
        return 1
    from audio_detect import SIGNAL_NAMES as A_SIGNALS
    for name in A_SIGNALS:
        print(f"  {name:21s}: {res['signals'][name]:.3f}")
    print(f"  classical  : {res['classical_combined']:.3f}")
    if res["learned_prob"] is not None:
        print(f"  LEARNED P(synthetic): {res['learned_prob']:.3f}  [audio.joblib]")
    elif res.get("model_error"):
        print(f"  ERROR: models/audio.joblib could not be used:\n"
              f"         {res['model_error']}", file=sys.stderr)
        print("  (falling back to the classical combiner)")
    else:
        print("  (no trained audio model -> classical combiner; run train_audio.py)")
    print(f"  duration   : {res['duration_s']}s @ {res['sample_rate']} Hz")
    print(f"  => {res['final_verdict']}  (confidence: {res['final_confidence']})")
    return 0


def analyze_text(path: Path) -> int:
    text = sys.stdin.read() if str(path) == "-" else (
        path.read_text(errors="ignore") if path.exists() else None)
    if text is None:
        print(f"file not found: {path}", file=sys.stderr)
        return 1
    from pipeline import analyze_text as analyze_text_str

    res = analyze_text_str(text)
    from text_detect import SIGNAL_NAMES as T_SIGNALS
    label = "-" if str(path) == "-" else path.name
    print(f"{label}  ({len(text.split())} words)")
    for name in T_SIGNALS:
        print(f"  {name:20s}: {res['signals'][name]:.3f}")
    print(f"  P(AI-generated): {res['prob_ai']:.3f}")
    print(f"  => {res['final_verdict']}  (confidence: {res['final_confidence']})")
    return 0


def run_batch(root: Path, csv_path: str | None, top: int, recursive: bool,
              kinds: tuple[str, ...], limit: int, watermark: bool) -> int:
    """Rank a folder of unlabelled media by P(synthetic) — forensic triage."""
    import batch as batch_mod
    from pipeline import MODEL_PATH, load_model

    if not root.exists():
        print(f"path not found: {root}", file=sys.stderr)
        return 1

    paths = batch_mod.collect(root, recursive=recursive, kinds=kinds)
    skipped_video = (batch_mod.count_skipped_videos(root, recursive=recursive)
                     if "video" not in kinds else 0)
    if limit:
        paths = paths[:limit]
    if not paths:
        print(f"no {'/'.join(kinds)} files under {root}", file=sys.stderr)
        return 1

    model = load_model()
    if model is not None:
        mi = model_info() or {}
        print(f"model: {mi.get('mode', '?')}/{mi.get('classifier', '?')} "
              f"({MODEL_PATH.name}, n_train={mi.get('n_train', '?')})")
    else:
        from pipeline import load_error

        err = load_error()
        print(f"model: none — using the training-free classical combiner"
              + (f"\n  ({err})" if err else ""))

    print(f"scanning {len(paths)} files under {root}"
          + (f"  (+{skipped_video} video skipped — use `cli.py video`)"
             if skipped_video else ""))

    def progress(done: int, total: int, path: Path) -> None:
        print(f"\r  [{done:>4}/{total}] {path.name[:48]:<48}",
              end="", file=sys.stderr, flush=True)

    rows = batch_mod.triage(paths, model=model, watermark=watermark,
                            progress=progress)
    print("\r" + " " * 70 + "\r", end="", file=sys.stderr)

    print()
    print(batch_mod.format_table(rows, top=top))
    s = batch_mod.summarise(rows)
    shown = min(top, s["total"] - s["errors"])
    print(f"\n{shown} of {s['total']} shown  ·  flagged {s['flagged']} "
          f"(watermarked {s['watermarked']})  ·  inconclusive {s['inconclusive']}"
          f"  ·  likely real {s['real']}  ·  unreadable {s['errors']}")
    if csv_path:
        batch_mod.write_csv(rows, csv_path)
        if csv_path != "-":
            print(f"wrote {len(rows)} ranked rows to {csv_path}")
    return 0


def analyze_watermark(path: Path) -> int:
    if not path.exists():
        print(f"file not found: {path}", file=sys.stderr)
        return 1
    w = watermark_inspect(path)
    print(f"{path.name}")
    _print_watermark(w)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Label-free multimodal synthetic-media detector "
                    "(image / audio / video / text + watermark scan)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("image", help="full analysis on a single image")
    pi.add_argument("path", type=Path)
    pi.add_argument("--explain", type=Path, default=None, metavar="DIR",
                    help="also write spatial explanation heat-maps (ELA, JPEG "
                         "ghost, spectral heterogeneity) as PNGs into DIR")

    pv = sub.add_parser("video", help="analyze a video")
    pv.add_argument("path", type=Path)
    pv.add_argument("--every", type=int, default=15, help="sample every N frames (default 15)")

    pa = sub.add_parser("audio", help="synthetic-voice analysis on an audio file")
    pa.add_argument("path", type=Path)

    pt = sub.add_parser("text", help="AI/LLM-text analysis (FILE or - for stdin)")
    pt.add_argument("path", type=Path)

    pw = sub.add_parser("watermark", help="watermark / metadata scan only")
    pw.add_argument("path", type=Path)

    pb = sub.add_parser(
        "batch", help="rank a folder of unlabelled media by P(synthetic)",
        description="Forensic triage: score every image and audio file under "
                    "DIR and print them ranked, most synthetic first. Videos "
                    "are counted but not scored (use `cli.py video`).")
    pb.add_argument("path", type=Path, metavar="DIR")
    pb.add_argument("--csv", default=None, metavar="FILE",
                    help="write the full ranked table as CSV ('-' for stdout)")
    pb.add_argument("--top", type=int, default=20,
                    help="rows to print (default 20; the CSV always has all)")
    pb.add_argument("--kind", choices=["image", "audio", "both"], default="both",
                    help="which modalities to score (default: both)")
    pb.add_argument("--no-recursive", action="store_true",
                    help="do not descend into sub-directories")
    pb.add_argument("--limit", type=int, default=0,
                    help="stop after N files (0 = no limit)")
    pb.add_argument("--no-watermark", action="store_true",
                    help="skip the per-file provenance/watermark scan (faster)")

    args = p.parse_args()
    # Line-buffer stdout so that piping to a file or a pager keeps the progress
    # ticker (stderr) and the report (stdout) in the order they were written.
    sys.stdout.reconfigure(line_buffering=True)
    if args.cmd == "image":
        return analyze_image(args.path, explain_dir=args.explain)
    if args.cmd == "video":
        return analyze_video(args.path, args.every)
    if args.cmd == "audio":
        return analyze_audio(args.path)
    if args.cmd == "text":
        return analyze_text(args.path)
    if args.cmd == "batch":
        kinds = (("image", "audio") if args.kind == "both" else (args.kind,))
        return run_batch(args.path, args.csv, args.top,
                         recursive=not args.no_recursive, kinds=kinds,
                         limit=args.limit, watermark=not args.no_watermark)
    return analyze_watermark(args.path)


if __name__ == "__main__":
    sys.exit(main())
