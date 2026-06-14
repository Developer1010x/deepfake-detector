"""Flask web frontend for the classical deepfake detector.

Accepts images (JPG, PNG, WebP, BMP, GIF, TIFF, HEIC, HEIF, AVIF, ICO) and
videos (MP4, MOV, AVI, MKV, WebM, M4V, MPEG, MPG, FLV, WMV, 3GP). Per-frame
analysis is run on videos with adaptive sampling.

Run locally:
    python3 app.py                          # dev server on :5000

Run for hosting:
    gunicorn -b 0.0.0.0:8000 -w 2 -t 120 app:app
"""

from __future__ import annotations

import os
import statistics
import tempfile
from pathlib import Path

import cv2
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

# Optional iPhone / modern-format support; fall through quietly if missing.
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass
try:
    import pillow_avif  # noqa: F401
except ImportError:
    pass

from detector import DEFAULT_WEIGHTS, SIGNAL_NAMES
from pipeline import (
    analyze_audio_path,
    analyze_image,
    analyze_path,
    analyze_text,
    model_info,
)
from watermark import inspect as watermark_inspect

IMAGE_EXTS = {"jpg", "jpeg", "png", "webp", "bmp", "gif", "tif", "tiff",
              "heic", "heif", "avif", "ico"}
VIDEO_EXTS = {"mp4", "mov", "avi", "mkv", "webm", "m4v", "mpeg", "mpg",
              "flv", "wmv", "3gp"}
AUDIO_EXTS = {"wav", "mp3", "m4a", "aac", "flac", "ogg", "opus", "wma"}
ALLOWED_EXTS = IMAGE_EXTS | VIDEO_EXTS | AUDIO_EXTS

MAX_BYTES = 100 * 1024 * 1024   # 100 MB to accommodate short videos
MAX_FRAMES = 24                 # cap frames analyzed per video for responsiveness

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES


def _agg(values: list[float]) -> dict:
    return {
        "mean": round(statistics.fmean(values), 4) if values else 0.0,
        "std":  round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0,
        "min":  round(min(values), 4) if values else 0.0,
        "max":  round(max(values), 4) if values else 0.0,
    }


def analyze_image_file(tmp_path: str, original_name: str) -> dict:
    try:
        res = analyze_path(tmp_path)
    except (UnidentifiedImageError, OSError) as e:
        return {"error": f"could not decode image: {e}"}

    return {
        "type": "image",
        "filename": original_name,
        "verdict": res["final_verdict"],
        "confidence": res["final_confidence"],
        "prob_synthetic": res["prob_synthetic"],
        "decision_source": res["decision_source"],
        "fusion_prob": res["fusion_prob"],
        "model": model_info(),
        "pixel": {**res["signals"], "combined": res["classical_combined"],
                  "verdict": res["verdict"]},
        "patterns": res["patterns"],
        "watermark": res["watermark"],
    }


def analyze_audio_file(tmp_path: str, original_name: str) -> dict:
    from audio_detect import SIGNAL_NAMES as A_SIGNALS

    res = analyze_audio_path(tmp_path)
    if "error" in res:
        return res
    return {
        "type": "audio",
        "filename": original_name,
        "verdict": res["final_verdict"],
        "confidence": res["final_confidence"],
        "prob_synthetic": res["prob_synthetic"],
        "decision_source": res["decision_source"],
        "learned_prob": res["learned_prob"],
        "duration_s": res["duration_s"],
        "signals": {k: res["signals"][k] for k in A_SIGNALS},
        "classical_combined": res["classical_combined"],
    }


def analyze_video_file(tmp_path: str, original_name: str) -> dict:
    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        return {"error": "could not open video"}

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if total <= 0:
        cap.release()
        return {"error": "video has no decodable frames"}

    sample_every = max(1, total // MAX_FRAMES)
    per_frame = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % sample_every == 0 and len(per_frame) < MAX_FRAMES:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            r = analyze_image(Image.fromarray(rgb))
            per_frame.append({
                "frame": idx,
                "time": round(idx / fps, 2),
                "prob_synthetic": r["prob_synthetic"],
                "decision_source": r["decision_source"],
                "classical_combined": r["classical_combined"],
                **{k: r["signals"][k] for k in SIGNAL_NAMES},
            })
        idx += 1
    cap.release()

    if not per_frame:
        return {"error": "no frames decoded"}

    cols = {k: [p[k] for p in per_frame]
            for k in ("prob_synthetic", "classical_combined", *SIGNAL_NAMES)}
    aggregate = {k: _agg(v) for k, v in cols.items()}

    avg = aggregate["prob_synthetic"]["mean"]
    tstd = aggregate["prob_synthetic"]["std"]

    # Audio-track forensic analysis (synthetic-voice cue), fused with the visual.
    from audio_io import extract_audio_from_video
    from pipeline import analyze_audio as analyze_audio_wave

    audio = None
    a = extract_audio_from_video(tmp_path)
    if a is not None:
        ar = analyze_audio_wave(a[1], a[0])
        audio = {"prob_synthetic": ar["prob_synthetic"],
                 "decision_source": ar["decision_source"],
                 "signals": ar["signals"]}

    # Temporal instability of the per-frame probability is itself a deepfake cue.
    visual_fake = avg > 0.5 or tstd > 0.15
    audio_fake = audio is not None and audio["prob_synthetic"] > 0.5
    fused = avg if audio is None else round(0.6 * avg + 0.4 * audio["prob_synthetic"], 4)
    if visual_fake or audio_fake:
        flags = "+".join(([("visual")] if visual_fake else []) + (["audio"] if audio_fake else []))
        verdict = f"likely AI-generated / deepfake (flagged by {flags})"
        confidence = "high" if (avg > 0.6 or tstd > 0.2 or audio_fake) else "medium"
    else:
        verdict = "likely real footage"
        confidence = "medium"

    wm = watermark_inspect(tmp_path)
    if wm["score"] >= 0.5:
        verdict = "AI watermark detected (synthetic)"
        confidence = "high"

    return {
        "type": "video",
        "filename": original_name,
        "verdict": verdict,
        "confidence": confidence,
        "prob_synthetic": round(avg, 4),
        "temporal_std": round(tstd, 4),
        "fused_prob": fused,
        "audio": audio,
        "decision_source": per_frame[0].get("decision_source", "fusion/classical"),
        "model": model_info(),
        "total_frames": total,
        "sampled_frames": len(per_frame),
        "fps": round(fps, 2),
        "duration_s": round(total / fps, 2),
        "aggregate": aggregate,
        "per_frame": per_frame,
        "watermark": wm,
    }


@app.route("/")
def index():
    return render_template(
        "index.html",
        signal_names=SIGNAL_NAMES,
        default_weights=DEFAULT_WEIGHTS,
        max_mb=MAX_BYTES // (1024 * 1024),
        image_exts=sorted(IMAGE_EXTS),
        video_exts=sorted(VIDEO_EXTS),
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    f = request.files.get("file") or request.files.get("image")
    if not f or not f.filename:
        return jsonify({"error": "no file uploaded"}), 400
    ext = Path(f.filename).suffix.lower().lstrip(".")
    if ext not in ALLOWED_EXTS:
        return jsonify({"error": f"unsupported extension: .{ext}"}), 400

    with tempfile.NamedTemporaryFile(suffix="." + ext, delete=False) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name
    try:
        if ext in IMAGE_EXTS:
            result = analyze_image_file(tmp_path, f.filename)
        elif ext in AUDIO_EXTS:
            result = analyze_audio_file(tmp_path, f.filename)
        else:
            result = analyze_video_file(tmp_path, f.filename)
        if "error" in result:
            return jsonify(result), 400
        return jsonify(result)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.route("/analyze_text", methods=["POST"])
def analyze_text_route():
    data = request.get_json(silent=True) or request.form
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "no text provided"}), 400
    if len(text.split()) < 15:
        return jsonify({"error": "need at least ~15 words for a reliable verdict"}), 400
    res = analyze_text(text)
    return jsonify({"type": "text", "n_words": len(text.split()), **res})


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": f"file exceeds {MAX_BYTES // (1024 * 1024)} MB limit"}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
