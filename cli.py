"""CLI: `python cli.py image PATH` or `python cli.py video PATH [--every N]`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from detector import SIGNAL_NAMES, detect
from pipeline import analyze_path, model_info
from watermark import inspect as watermark_inspect


def _print_scores(r: dict) -> None:
    for name in SIGNAL_NAMES:
        print(f"  {name:11s}: {r[name]:.3f}")
    print(f"  combined   : {r['combined']:.3f}  -> {r['verdict']}")


def _print_watermark(w: dict) -> None:
    print(f"  watermark  : {w['score']:.3f}  -> {w['verdict']}")
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
        print(f"    - mid-band spectral anomaly: {w['spectral_anomaly']:.3f}")


def analyze_image(path: Path) -> int:
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

    # learned fusion verdict
    mi = model_info()
    if res["fusion_prob"] is not None and mi:
        print(f"  FUSION P(synthetic): {res['fusion_prob']:.3f}  "
              f"[{mi['mode']}/{mi['classifier']} model]")
    else:
        print("  (no trained fusion model -> using classical combiner; "
              "run train_selfsup.py)")
    _print_watermark(res["watermark"])
    print(f"  => {res['final_verdict']}  (confidence: {res['final_confidence']})")
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
    else:
        print("  (no trained audio model -> classical combiner; run train_audio.py)")
    print(f"  duration   : {res['duration_s']}s")
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
        description="Classical deepfake detector (7 signals + AI watermark scan)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("image", help="full analysis on a single image")
    pi.add_argument("path", type=Path)

    pv = sub.add_parser("video", help="analyze a video")
    pv.add_argument("path", type=Path)
    pv.add_argument("--every", type=int, default=15, help="sample every N frames (default 15)")

    pa = sub.add_parser("audio", help="synthetic-voice analysis on an audio file")
    pa.add_argument("path", type=Path)

    pt = sub.add_parser("text", help="AI/LLM-text analysis (FILE or - for stdin)")
    pt.add_argument("path", type=Path)

    pw = sub.add_parser("watermark", help="watermark / metadata scan only")
    pw.add_argument("path", type=Path)

    args = p.parse_args()
    if args.cmd == "image":
        return analyze_image(args.path)
    if args.cmd == "video":
        return analyze_video(args.path, args.every)
    if args.cmd == "audio":
        return analyze_audio(args.path)
    if args.cmd == "text":
        return analyze_text(args.path)
    return analyze_watermark(args.path)


if __name__ == "__main__":
    sys.exit(main())
