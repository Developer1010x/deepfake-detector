"""Unified inference pipeline.

A single entry point used by the web app, the CLI and the video path. It layers
three evidence sources and degrades gracefully:

  1. **Learned fusion model** (``models/fusion.joblib``) - the hybrid
     deep+hand-crafted classifier from :mod:`fusion`, if it has been trained and
     (for deep/fusion modes) torch is available. Produces the headline
     calibrated ``P(synthetic)``.
  2. **Classical signals** - the seven statistical signals and three pattern
     checks, always computed (interpretable, no dependencies). Used as the
     headline probability when no learned model is present.
  3. **Watermark / provenance** - C2PA, PNG/EXIF AI tags, keyword and spectral
     scan from :mod:`watermark`. A positive provenance hit is a hard override.

The verdict logic is deliberately conservative and reports a confidence band so
the result is usable in a paper's qualitative analysis.
"""

from __future__ import annotations

import functools
from pathlib import Path

from PIL import Image

from detector import SIGNAL_NAMES, detect
from patterns import all_patterns

MODEL_PATH = Path("models/fusion.joblib")


@functools.lru_cache(maxsize=4)
def load_model(path: str = str(MODEL_PATH)):
    """Load the trained FusionModel if present and usable, else None."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        import deep_features
        from fusion import FusionModel

        model = FusionModel.load(p)
        if model.uses_deep and not deep_features.available():
            return None  # model needs torch but it's unavailable
        return model
    except Exception:
        return None


def model_info(path: str = str(MODEL_PATH)) -> dict | None:
    m = load_model(path)
    if m is None:
        return None
    return {"mode": m.config.mode, "classifier": m.config.classifier,
            **{k: v for k, v in m.train_meta_.items()
               if k in ("calibrated", "val_auc", "n_train")}}


def _band(prob: float) -> str:
    """Confidence band from distance to the decision boundary."""
    d = abs(prob - 0.5)
    return "high" if d >= 0.3 else "medium" if d >= 0.12 else "low"


def analyze_image(img: Image.Image, model_path: str = str(MODEL_PATH)) -> dict:
    """Analyze a single PIL image (no file-level watermark scan).

    Returns the classical breakdown plus, when available, the learned fusion
    probability; ``prob_synthetic`` is the headline value (fusion if present,
    else the classical combined score).
    """
    classical = detect(img)
    patterns = all_patterns(img)

    model = load_model(model_path)
    fusion_prob = None
    if model is not None:
        try:
            fusion_prob = model.predict_one(img)
        except Exception:
            fusion_prob = None

    prob = fusion_prob if fusion_prob is not None else classical["combined"]
    source = "fusion" if fusion_prob is not None else "classical"
    return {
        "prob_synthetic": round(float(prob), 4),
        "decision_source": source,
        "verdict": "synthetic" if prob > 0.5 else "real",
        "confidence": _band(prob),
        "fusion_prob": None if fusion_prob is None else round(fusion_prob, 4),
        "classical_combined": round(classical["combined"], 4),
        "signals": {k: round(classical[k], 4) for k in SIGNAL_NAMES},
        "patterns": patterns,
    }


AUDIO_MODEL_PATH = Path("models/audio.joblib")


@functools.lru_cache(maxsize=2)
def load_audio_model(path: str = str(AUDIO_MODEL_PATH)):
    """Load the trained audio classifier bundle if present, else None."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        import joblib
        return joblib.load(p)
    except Exception:
        return None


def analyze_audio(x, sr: int) -> dict:
    """Analyze a mono waveform: learned P(synthetic) if a model is present, else
    the training-free classical forensic combiner."""
    import numpy as np

    import audio_detect as ad

    scores = ad.all_scores(x, sr)
    classical = float(sum(w * scores[n]
                          for w, n in zip(ad.DEFAULT_WEIGHTS, ad.SIGNAL_NAMES)))
    bundle = load_audio_model()
    learned = None
    if bundle is not None:
        try:
            vec = np.asarray([[scores[n] for n in bundle["signal_names"]]])
            learned = float(bundle["clf"].predict_proba(vec)[0, 1])
        except Exception:
            learned = None
    prob = learned if learned is not None else classical
    return {
        "prob_synthetic": round(float(prob), 4),
        "decision_source": "learned" if learned is not None else "classical",
        "verdict": "synthetic" if prob > 0.5 else "real",
        "confidence": _band(prob),
        "learned_prob": None if learned is None else round(learned, 4),
        "classical_combined": round(classical, 4),
        "signals": {k: round(scores[k], 4) for k in ad.SIGNAL_NAMES},
    }


def analyze_audio_path(path: str | Path) -> dict:
    """Decode an audio file (any ffmpeg-readable format) and analyze it."""
    from audio_io import load_audio

    res = load_audio(path)
    if res is None:
        return {"error": "could not decode audio (non-WAV needs ffmpeg installed)"}
    sr, x = res
    out = analyze_audio(x, sr)
    out["duration_s"] = round(len(x) / 16000.0, 2)
    out["final_verdict"] = ("likely synthetic / cloned voice"
                            if out["prob_synthetic"] > 0.5 else "likely real voice")
    out["final_confidence"] = out["confidence"]
    return out


TEXT_MODEL_PATH = Path("models/text.joblib")


@functools.lru_cache(maxsize=2)
def load_text_model(path: str = str(TEXT_MODEL_PATH)):
    p = Path(path)
    if not p.exists():
        return None
    try:
        import joblib
        return joblib.load(p)
    except Exception:
        return None


def analyze_text(text: str) -> dict:
    """Analyze a text string for AI/LLM authorship via the forensic bank.

    Uses the learned (calibrated) model if ``models/text.joblib`` is present,
    else the training-free classical combiner.
    """
    import numpy as np

    import text_detect as td

    scores = td.all_scores(text)
    classical = float(sum(w * scores[n]
                          for w, n in zip(td.DEFAULT_WEIGHTS, td.SIGNAL_NAMES)))
    bundle = load_text_model()
    learned = None
    if bundle is not None:
        try:
            vec = np.asarray([[scores[n] for n in bundle["signal_names"]]])
            learned = float(bundle["clf"].predict_proba(vec)[0, 1])
        except Exception:
            learned = None
    prob = learned if learned is not None else classical
    return {
        "prob_ai": round(float(prob), 4),
        "decision_source": "learned" if learned is not None else "classical",
        "verdict": "AI-generated" if prob > 0.5 else "human-written",
        "confidence": _band(prob),
        "learned_prob": None if learned is None else round(learned, 4),
        "classical_combined": round(classical, 4),
        "signals": {k: round(float(scores[k]), 4) for k in td.SIGNAL_NAMES},
        "final_verdict": ("likely AI-generated" if prob > 0.5 else "likely human-written"),
        "final_confidence": _band(prob),
    }


def analyze_path(path: str | Path, model_path: str = str(MODEL_PATH)) -> dict:
    """Full single-image analysis: pixel/learned evidence + watermark fusion."""
    from watermark import inspect as watermark_inspect

    path = Path(path)
    img = Image.open(path)
    img.load()
    res = analyze_image(img, model_path=model_path)
    wm = watermark_inspect(path)
    res["watermark"] = wm

    # Fuse a hard provenance override with the probabilistic evidence.
    pixel_synth = res["prob_synthetic"] > 0.5
    wm_synth = wm["score"] >= 0.5
    if wm_synth:
        res["final_verdict"] = "AI-generated (provenance/watermark detected)"
        res["final_confidence"] = "high"
    elif pixel_synth and res["confidence"] == "high":
        res["final_verdict"] = "likely AI-generated"
        res["final_confidence"] = "high"
    elif pixel_synth:
        res["final_verdict"] = "possibly AI-generated"
        res["final_confidence"] = res["confidence"]
    else:
        res["final_verdict"] = "likely real"
        res["final_confidence"] = res["confidence"]
    return res
