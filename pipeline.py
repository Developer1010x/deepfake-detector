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

The verdict logic is deliberately conservative. When the calibrated probability
lands inside :data:`INCONCLUSIVE_MARGIN` of the 0.5 boundary the pipeline
**abstains** rather than forcing a call - see :func:`_band`. A forensic tool that
says "insufficient evidence" is more useful, and more honest, than one that
guesses at p=0.49.

Model artifacts are resolved relative to *this file*, never the working
directory, so the same input always produces the same verdict no matter where
the process was launched from.
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path

from PIL import Image

from detector import SIGNAL_NAMES, detect
from patterns import all_patterns

# Anchor every artifact path to the package directory. Using a CWD-relative path
# here means `cli.py image x.jpg` silently falls back to the classical combiner
# (and can return the opposite verdict) when run from anywhere but the repo root.
HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"
MODEL_PATH = MODELS_DIR / "fusion.joblib"

#: Half-width of the abstention band around the decision boundary. Probabilities
#: inside 0.5 +/- this are reported as "inconclusive" instead of real/synthetic.
INCONCLUSIVE_MARGIN = 0.12

# Which files this tool can read, by modality. Single source of truth: the web
# app derives its upload allowlist from these and `batch.collect` uses them to
# decide what to walk, so the CLI and the UI can never disagree about what is
# supported.
IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif",
                        ".tif", ".tiff", ".heic", ".heif", ".avif", ".ico"})
VIDEO_EXTS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v",
                        ".mpeg", ".mpg", ".flv", ".wmv", ".3gp"})
AUDIO_EXTS = frozenset({".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg",
                        ".opus", ".wma"})


def media_kind(path: str | Path) -> str | None:
    """``"image"`` / ``"audio"`` / ``"video"`` from the extension, else None."""
    ext = Path(path).suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in VIDEO_EXTS:
        return "video"
    return None

# Populated by load_model()/load_audio_model()/load_text_model() when an artifact
# is present but unusable, so callers can report the *real* reason instead of
# "no trained model".
_LOAD_ERRORS: dict[str, str] = {}

# Model-card mismatches: the artifact loaded, but it was pickled by a different
# library version than the one running. Not fatal, but the user is told.
_CARD_WARNINGS: dict[str, list[str]] = {}


def card_warnings(path: str | Path = MODEL_PATH) -> list[str]:
    """Environment mismatches between this artifact's model card and this run."""
    return _CARD_WARNINGS.get(str(Path(path)), [])


def load_error(path: str | Path = MODEL_PATH) -> str | None:
    """Why the artifact at ``path`` could not be loaded, or None.

    ``None`` also covers "the file simply is not there", which is a normal
    untrained state rather than a failure.
    """
    return _LOAD_ERRORS.get(str(Path(path)))


def _record_error(path: Path, exc: BaseException) -> None:
    _LOAD_ERRORS[str(path)] = f"{type(exc).__name__}: {exc}"


def _clear_error(path: Path) -> None:
    _LOAD_ERRORS.pop(str(path), None)


def _read_card(path: Path) -> dict:
    """The model card (``*.meta.json``) beside an artifact, or {}."""
    meta = path.with_suffix(".meta.json")
    try:
        return json.loads(meta.read_text())
    except Exception:
        return {}


def _check_env(card: dict) -> list[str]:
    """Compare the environment recorded at training time with the current one.

    Returns a list of human-readable mismatch strings (empty when the card has no
    ``env`` block, i.e. it predates model cards).
    """
    trained = card.get("env") or {}
    if not trained:
        return []
    warnings = []
    for pkg, want in trained.items():
        try:
            import importlib.metadata as md

            have = md.version(pkg)
        except Exception:
            have = None
        if have is None:
            warnings.append(f"{pkg}: trained with {want}, not installed now")
        elif have.split("+")[0] != str(want).split("+")[0]:
            warnings.append(f"{pkg}: trained with {want}, running {have}")
    return warnings


@functools.lru_cache(maxsize=4)
def load_model(path: str = str(MODEL_PATH)):
    """Load the trained FusionModel if present and usable, else None.

    Any failure is *recorded* (see :func:`load_error`) rather than swallowed, so
    a version mismatch or a corrupt pickle is not misreported to the user as an
    untrained model.
    """
    p = Path(path)
    if not p.exists():
        _clear_error(p)
        return None
    try:
        import deep_features
        from fusion import FusionModel

        model = FusionModel.load(p)
        if model.uses_deep and not deep_features.available():
            _LOAD_ERRORS[str(p)] = (
                "model was trained in 'fusion'/'deep' mode but torch/torchvision "
                "are unavailable; install them or retrain with --mode handcrafted"
            )
            return None
        if model.uses_deep and offline() and not deep_features.weights_cached():
            _LOAD_ERRORS[str(p)] = (
                "DEEPFAKE_OFFLINE=1 and the ~45 MB ResNet-18 backbone weights are "
                "not in the torch hub cache; unset DEEPFAKE_OFFLINE once to fetch "
                "them, or retrain with --mode handcrafted"
            )
            return None
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        _record_error(p, exc)
        return None
    _clear_error(p)
    _CARD_WARNINGS[str(p)] = _check_env(_read_card(p))
    return model


def model_info(path: str = str(MODEL_PATH)) -> dict | None:
    m = load_model(path)
    if m is None:
        return None
    card = _read_card(Path(path))
    info = {"mode": m.config.mode, "classifier": m.config.classifier,
            **{k: v for k, v in m.train_meta_.items()
               if k in ("calibrated", "val_auc", "n_train")}}
    if card.get("corpus"):
        info["corpus"] = card["corpus"]
    warnings = card_warnings(path)
    if warnings:
        info["env_warnings"] = warnings
    return info


def _band(prob: float) -> str:
    """Confidence band from distance to the decision boundary.

    ``"low"`` is the abstention band: the model is not separating the classes at
    this point and the caller should report "inconclusive".
    """
    d = abs(prob - 0.5)
    return "high" if d >= 0.3 else "medium" if d >= INCONCLUSIVE_MARGIN else "low"


#: Public name for the band function, for callers outside this module
#: (``batch.py`` ranks with it).
confidence_band = _band


def is_inconclusive(prob: float) -> bool:
    return abs(prob - 0.5) < INCONCLUSIVE_MARGIN


def final_verdict(prob: float, confidence: str, provenance: bool = False) -> tuple[str, str]:
    """Fuse the probabilistic evidence with a hard provenance hit.

    Returns ``(verdict, confidence)``. Shared by :func:`analyze_path` (single
    file) and :mod:`batch` (folder triage) so the two can never drift apart:
    a watermark overrides everything, the abstention band beats a forced call,
    and only a high-confidence positive is stated as "likely".

    ``provenance`` must be :data:`watermark.inspect`'s ``"provenance"`` flag -
    i.e. a marker somebody wrote (C2PA, a diffusion parameter chunk, an EXIF
    generator name), never the statistical part of the watermark score. Those
    heuristics fire on ordinary photographs, and overriding a calibrated
    probability with them produced *confident* false positives.
    """
    if provenance:
        return "AI-generated (provenance/watermark detected)", "high"
    if is_inconclusive(prob):
        # Abstain rather than commit to a coin-flip. |p - 0.5| < INCONCLUSIVE_MARGIN
        # means the model is not separating the classes for this input.
        return "inconclusive - insufficient evidence", "low"
    if prob <= 0.5:
        return "likely real", confidence
    return ("likely AI-generated" if confidence == "high"
            else "possibly AI-generated"), confidence


def analyze_image(img: Image.Image, model_path: str = str(MODEL_PATH),
                  explain: bool = False) -> dict:
    """Analyze a single PIL image (no file-level watermark scan).

    Returns the classical breakdown plus, when available, the learned fusion
    probability; ``prob_synthetic`` is the headline value (fusion if present,
    else the classical combined score). With ``explain=True`` the per-block
    forensic fields behind three of the signals are rendered as base64 PNG
    heatmaps under ``"explanations"``.
    """
    classical = detect(img)
    patterns = all_patterns(img)

    model = load_model(model_path)
    fusion_prob = None
    if model is not None:
        try:
            fusion_prob = model.predict_one(img)
        except Exception as exc:  # noqa: BLE001
            _record_error(Path(model_path), exc)
            fusion_prob = None

    prob = fusion_prob if fusion_prob is not None else classical["combined"]
    source = "fusion" if fusion_prob is not None else "classical"
    out = {
        "prob_synthetic": round(float(prob), 4),
        "decision_source": source,
        "verdict": ("inconclusive" if is_inconclusive(prob)
                    else "synthetic" if prob > 0.5 else "real"),
        "confidence": _band(prob),
        "inconclusive": is_inconclusive(prob),
        "fusion_prob": None if fusion_prob is None else round(fusion_prob, 4),
        "classical_combined": round(classical["combined"], 4),
        "signals": {k: round(classical[k], 4) for k in SIGNAL_NAMES},
        "patterns": patterns,
        "model_error": load_error(model_path),
        "model_warnings": card_warnings(model_path),
    }
    if explain:
        from explain import explanation_maps

        out["explanations"] = explanation_maps(img)
    return out


AUDIO_MODEL_PATH = MODELS_DIR / "audio.joblib"


@functools.lru_cache(maxsize=2)
def load_audio_model(path: str = str(AUDIO_MODEL_PATH)):
    """Load the trained audio classifier bundle if present, else None."""
    p = Path(path)
    if not p.exists():
        _clear_error(p)
        return None
    try:
        import joblib
        bundle = joblib.load(p)
    except Exception as exc:  # noqa: BLE001
        _record_error(p, exc)
        return None
    _clear_error(p)
    _CARD_WARNINGS[str(p)] = _check_env(_read_card(p))
    return bundle


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
        except Exception as exc:  # noqa: BLE001
            _record_error(AUDIO_MODEL_PATH, exc)
            learned = None
    prob = learned if learned is not None else classical
    return {
        "prob_synthetic": round(float(prob), 4),
        "decision_source": "learned" if learned is not None else "classical",
        "verdict": ("inconclusive" if is_inconclusive(prob)
                    else "synthetic" if prob > 0.5 else "real"),
        "confidence": _band(prob),
        "inconclusive": is_inconclusive(prob),
        "learned_prob": None if learned is None else round(learned, 4),
        "classical_combined": round(classical, 4),
        "signals": {k: round(scores[k], 4) for k in ad.SIGNAL_NAMES},
        "model_error": load_error(AUDIO_MODEL_PATH),
        "model_warnings": card_warnings(AUDIO_MODEL_PATH),
    }


def analyze_audio_path(path: str | Path) -> dict:
    """Decode an audio file (any ffmpeg-readable format) and analyze it."""
    from audio_io import load_audio

    res = load_audio(path)
    if res is None:
        return {"error": "could not decode audio (non-WAV needs ffmpeg installed)"}
    sr, x = res
    out = analyze_audio(x, sr)
    # `x` arrives at the file's NATIVE rate - resampling happens later, inside
    # the signal bank - so the duration must divide by `sr`, not the target rate.
    out["sample_rate"] = int(sr)
    out["duration_s"] = round(len(x) / float(sr), 2)
    if out["inconclusive"]:
        out["final_verdict"] = "inconclusive - insufficient evidence"
    else:
        out["final_verdict"] = ("likely synthetic / cloned voice"
                                if out["prob_synthetic"] > 0.5 else "likely real voice")
    out["final_confidence"] = out["confidence"]
    return out


TEXT_MODEL_PATH = MODELS_DIR / "text.joblib"


@functools.lru_cache(maxsize=2)
def load_text_model(path: str = str(TEXT_MODEL_PATH)):
    p = Path(path)
    if not p.exists():
        _clear_error(p)
        return None
    try:
        import joblib
        bundle = joblib.load(p)
    except Exception as exc:  # noqa: BLE001
        _record_error(p, exc)
        return None
    _clear_error(p)
    return bundle


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
        except Exception as exc:  # noqa: BLE001
            _record_error(TEXT_MODEL_PATH, exc)
            learned = None
    prob = learned if learned is not None else classical
    inconclusive = is_inconclusive(prob)
    return {
        "prob_ai": round(float(prob), 4),
        "decision_source": "learned" if learned is not None else "classical",
        "verdict": ("inconclusive" if inconclusive
                    else "AI-generated" if prob > 0.5 else "human-written"),
        "confidence": _band(prob),
        "inconclusive": inconclusive,
        "learned_prob": None if learned is None else round(learned, 4),
        "classical_combined": round(classical, 4),
        "signals": {k: round(float(scores[k]), 4) for k in td.SIGNAL_NAMES},
        "final_verdict": ("inconclusive - insufficient evidence" if inconclusive
                          else "likely AI-generated" if prob > 0.5
                          else "likely human-written"),
        "final_confidence": _band(prob),
        "model_error": load_error(TEXT_MODEL_PATH),
        "model_warnings": card_warnings(TEXT_MODEL_PATH),
    }


def analyze_path(path: str | Path, model_path: str = str(MODEL_PATH),
                 explain: bool = False) -> dict:
    """Full single-image analysis: pixel/learned evidence + watermark fusion."""
    from watermark import inspect as watermark_inspect

    path = Path(path)
    img = Image.open(path)
    img.load()
    res = analyze_image(img, model_path=model_path, explain=explain)
    wm = watermark_inspect(path)
    res["watermark"] = wm

    # Fuse a hard provenance override with the probabilistic evidence.
    res["final_verdict"], res["final_confidence"] = final_verdict(
        res["prob_synthetic"], res["confidence"], wm["provenance"])
    return res


#: Everything a fusion artifact's pickle and weights depend on.
FUSION_DEPS = ("scikit-learn", "numpy", "joblib", "torch", "torchvision")
#: The audio and text banks are pure numpy + sklearn - recording torch here would
#: emit a spurious mismatch warning on any install without it.
CLASSICAL_DEPS = ("scikit-learn", "numpy", "joblib")


def env_versions(packages: tuple[str, ...] = FUSION_DEPS) -> dict[str, str]:
    """Versions of the libraries a model artifact actually depends on.

    Written into the model card at train time and re-checked on load by
    :func:`_check_env`.
    """
    import importlib.metadata as md

    out: dict[str, str] = {}
    for pkg in packages:
        try:
            out[pkg] = md.version(pkg)
        except Exception:
            pass
    return out


def offline() -> bool:
    """True when the user has asked for a strictly offline run.

    The fusion model's deep stream downloads ~45 MB of ImageNet weights on first
    use; ``DEEPFAKE_OFFLINE=1`` keeps the classical path instead of hanging.
    """
    return os.environ.get("DEEPFAKE_OFFLINE", "").lower() in ("1", "true", "yes")
