"""Flask web frontend and multimodal API for the deepfake detector.

All four advertised modalities are reachable from the browser, not just two:

  * images  (JPG, PNG, WebP, BMP, GIF, TIFF, HEIC, HEIF, AVIF, ICO) - returns the
    seven pixel signals, the pattern panel, the watermark scan, and the spatial
    explanation heat-maps from :mod:`explain`.
  * audio   (WAV, MP3, M4A, AAC, FLAC, OGG, Opus, WMA) - the eight voice signals.
  * video   (MP4, MOV, AVI, MKV, WebM, M4V, MPEG, MPG, FLV, WMV, 3GP) - per-frame
    fusion with adaptive sampling, temporal instability, and the audio track.
  * text    via ``POST /analyze_text`` and the Text tab - the eight LLM signals.

``/analyze_sample`` runs the media that ships in ``samples/`` by catalog key, so
``/?demo=alpha-dog`` reproduces any README screenshot from a URL.

Run locally:
    python3 app.py                          # dev server on :5000
    PORT=8080 python3 app.py                # different port
    FLASK_DEBUG=1 python3 app.py            # Werkzeug debugger, loopback only

Run for hosting:
    gunicorn -b 0.0.0.0:8000 -w 2 -t 120 app:app
"""

from __future__ import annotations

import os
import statistics
import tempfile
from pathlib import Path

import cv2
from flask import Flask, jsonify, render_template, request, send_file
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

import pipeline
from detector import DEFAULT_WEIGHTS, SIGNAL_NAMES
from pipeline import (
    analyze_audio_path,
    analyze_image,
    analyze_path,
    analyze_text,
    model_info,
)
from watermark import inspect as watermark_inspect

# The supported-format lists live in pipeline.py (with leading dots) so the CLI,
# the batch triage and this upload allowlist cannot drift apart. Werkzeug hands
# us a suffix without the dot, hence the strip.
IMAGE_EXTS = {e[1:] for e in pipeline.IMAGE_EXTS}
VIDEO_EXTS = {e[1:] for e in pipeline.VIDEO_EXTS}
AUDIO_EXTS = {e[1:] for e in pipeline.AUDIO_EXTS}
ALLOWED_EXTS = IMAGE_EXTS | VIDEO_EXTS | AUDIO_EXTS

MAX_BYTES = 100 * 1024 * 1024   # 100 MB to accommodate short videos
MAX_FRAMES = 24                 # cap frames analyzed per video for responsiveness

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES

HERE = Path(__file__).resolve().parent
SAMPLES_DIR = HERE / "samples"


def _build_sample_catalog() -> dict[str, dict]:
    """Fixed allowlist of the media that ships with the repository.

    Built once at import from files on disk, so ``/analyze_sample`` never has to
    turn a client-supplied string into a filesystem path — it only ever looks a
    key up in this dict. Anything not listed here is simply not addressable.
    """
    wanted = [
        ("cat-astronaut", "fake/cat-astronaut.jpeg", "image", "AI image"),
        ("alpha-dog", "fake/alpha-dog.jpg", "image", "AI image"),
        ("portrait", "fake/portrait.jpeg", "image", "AI image"),
        ("real-voice", "demo/pseudo-real-voice.wav", "audio", "benign voice"),
        ("fake-voice", "demo/pseudo-fake-voice.wav", "audio", "vocoded voice"),
        ("human-text", "demo/pseudo-human.txt", "text", "human prose"),
        ("ai-text", "demo/pseudo-ai.txt", "text", "LLM-style prose"),
    ]
    catalog: dict[str, dict] = {}
    for key, rel, kind, label in wanted:
        path = SAMPLES_DIR / rel
        if path.exists():
            catalog[key] = {"path": path, "kind": kind, "label": label,
                            "filename": path.name}
    return catalog


SAMPLE_CATALOG = _build_sample_catalog()


def _agg(values: list[float]) -> dict:
    return {
        "mean": round(statistics.fmean(values), 4) if values else 0.0,
        "std":  round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0,
        "min":  round(min(values), 4) if values else 0.0,
        "max":  round(max(values), 4) if values else 0.0,
    }


def analyze_image_file(tmp_path: str, original_name: str,
                       explain: bool = True) -> dict:
    try:
        res = analyze_path(tmp_path, explain=explain)
    except (UnidentifiedImageError, OSError) as e:
        return {"error": f"could not decode image: {e}"}

    return {
        "type": "image",
        "filename": original_name,
        "verdict": res["final_verdict"],
        "confidence": res["final_confidence"],
        "inconclusive": res["inconclusive"],
        "prob_synthetic": res["prob_synthetic"],
        "decision_source": res["decision_source"],
        "fusion_prob": res["fusion_prob"],
        "model": model_info(),
        "model_error": res.get("model_error"),
        "model_warnings": res.get("model_warnings") or [],
        "pixel": {**res["signals"], "combined": res["classical_combined"],
                  "verdict": res["verdict"]},
        "patterns": res["patterns"],
        "explanations": res.get("explanations", []),
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
        "inconclusive": res["inconclusive"],
        "prob_synthetic": res["prob_synthetic"],
        "decision_source": res["decision_source"],
        "learned_prob": res["learned_prob"],
        "model_error": res.get("model_error"),
        "model_warnings": res.get("model_warnings") or [],
        "duration_s": res["duration_s"],
        "sample_rate": res["sample_rate"],
        "signal_names": list(A_SIGNALS),
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
    if wm["provenance"]:
        # Hard evidence only: the statistical half of the watermark score fires
        # on ordinary footage and must not override the per-frame verdict.
        verdict = "AI provenance metadata detected (synthetic)"
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
    from audio_detect import SIGNAL_NAMES as A_SIGNALS
    from text_detect import SIGNAL_NAMES as T_SIGNALS

    return render_template(
        "index.html",
        signal_names=SIGNAL_NAMES,
        audio_signal_names=A_SIGNALS,
        text_signal_names=T_SIGNALS,
        default_weights=DEFAULT_WEIGHTS,
        max_mb=MAX_BYTES // (1024 * 1024),
        image_exts=sorted(IMAGE_EXTS),
        video_exts=sorted(VIDEO_EXTS),
        audio_exts=sorted(AUDIO_EXTS),
        min_text_words=MIN_TEXT_WORDS,
        model=model_info(),
        samples=[{"key": k, **{kk: vv for kk, vv in v.items() if kk != "path"}}
                 for k, v in SAMPLE_CATALOG.items()],
    )


@app.route("/analyze_sample", methods=["POST"])
def analyze_sample():
    """Analyze one of the media files that ship with the repo, by catalog key.

    Lets the UI offer "try a sample" buttons without a round-trip through an
    upload, and makes every screenshot in the README reproducible from a URL
    (``/?demo=cat-astronaut``).
    """
    data = request.get_json(silent=True) or request.form
    entry = SAMPLE_CATALOG.get((data.get("name") or "").strip())
    if entry is None:
        return jsonify({"error": "unknown sample"}), 404

    path = str(entry["path"])
    if entry["kind"] == "image":
        result = analyze_image_file(path, entry["filename"])
    elif entry["kind"] == "audio":
        result = analyze_audio_file(path, entry["filename"])
    elif entry["kind"] == "text":
        body = entry["path"].read_text()
        result = {"type": "text", "n_words": len(body.split()),
                  "text": body, **analyze_text(body)}
    else:
        result = analyze_video_file(path, entry["filename"])
    if "error" in result:
        return jsonify(result), 400
    return jsonify({**result, "sample": data.get("name")})


@app.route("/samples/<key>")
def sample_media(key: str):
    """Serve a catalogued sample so the UI can preview it inline."""
    entry = SAMPLE_CATALOG.get(key)
    if entry is None:
        return jsonify({"error": "unknown sample"}), 404
    return send_file(entry["path"])


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


#: Below this, several of the eight text signals pin to a bound (ttr -> 0,
#: ngram_repetition -> 0, transition_density -> 1) and the "verdict" is a
#: weighted sum of saturated constants. prepare_datasets.py uses the same floor
#: when building the HC3 evaluation set, so the demo matches the benchmark.
MIN_TEXT_WORDS = 25


@app.route("/analyze_text", methods=["POST"])
def analyze_text_route():
    data = request.get_json(silent=True) or request.form
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "no text provided"}), 400
    n_words = len(text.split())
    if n_words < MIN_TEXT_WORDS:
        return jsonify({"error": f"need at least {MIN_TEXT_WORDS} words for a "
                                 f"meaningful verdict (got {n_words}); below that "
                                 f"several signals saturate at their bounds"}), 400
    res = analyze_text(text)
    return jsonify({"type": "text", "n_words": n_words, **res})


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": f"file exceeds {MAX_BYTES // (1024 * 1024)} MB limit"}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # The Werkzeug interactive debugger is remote code execution for anyone who
    # can reach the port, so it is opt-in (FLASK_DEBUG=1) and, when enabled,
    # bound to loopback only. Default: off, and bound to 0.0.0.0 for LAN demos.
    debug = os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true", "yes")
    host = os.environ.get("HOST", "127.0.0.1" if debug else "0.0.0.0")
    app.run(host=host, port=port, debug=debug)
