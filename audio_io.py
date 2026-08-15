"""Audio decoding helpers shared by the CLI and web app.

WAV is read directly with the standard-library ``wave`` module (see
:func:`audio_detect.load_wav`). Any other container (MP3, M4A, FLAC, OGG, or the
audio track of a video) is transcoded to a temporary 16 kHz mono WAV with
``ffmpeg`` if it is installed; if ffmpeg is unavailable the caller degrades
gracefully (audio analysis is skipped, not faked).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from audio_detect import TARGET_SR, load_wav

#: WAV is the one container this module reads without shelling out to ffmpeg.
#: The full list of audio extensions the tool accepts lives in
#: ``pipeline.AUDIO_EXTS`` - one definition, used by the CLI, the web upload
#: allowlist and the batch walker alike.
WAV_EXTS = {".wav"}


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _ffmpeg_to_wav(src: str | Path, sr: int = TARGET_SR) -> tuple[int, np.ndarray] | None:
    """Transcode any ffmpeg-readable media to mono ``sr`` WAV and load it."""
    if not have_ffmpeg():
        return None
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out = tmp.name
    try:
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(src),
             "-ac", "1", "-ar", str(sr), "-vn", out],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        if proc.returncode != 0 or not Path(out).exists() or Path(out).stat().st_size == 0:
            return None
        return load_wav(out)
    finally:
        try:
            Path(out).unlink()
        except OSError:
            pass


def load_audio(path: str | Path) -> tuple[int, np.ndarray] | None:
    """Return (sample_rate, mono float waveform) for any supported audio file,
    or ``None`` if it cannot be decoded (e.g. non-WAV with no ffmpeg)."""
    path = Path(path)
    if path.suffix.lower() in WAV_EXTS:
        try:
            return load_wav(path)
        except Exception:
            return _ffmpeg_to_wav(path)
    return _ffmpeg_to_wav(path)


def extract_audio_from_video(path: str | Path) -> tuple[int, np.ndarray] | None:
    """Pull the audio track out of a video as mono 16 kHz, or ``None`` if the
    video has no audio track / ffmpeg is unavailable."""
    res = _ffmpeg_to_wav(path)
    if res is None:
        return None
    sr, x = res
    return None if x.size < TARGET_SR // 2 else (sr, x)   # <0.5s -> treat as silent
