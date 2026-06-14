"""Classical synthetic-voice / audio-deepfake detector.

Eight independent forensic signals, no learned models. Each returns a score in
[0, 1] where higher = more synthetic-looking (TTS / voice-conversion / vocoder).
The design mirrors :mod:`detector` for images: physically-grounded, interpretable
cues that target the *forensic consequences* of neural audio synthesis rather
than any one vocoder's signature.

Signals
  spectral_flatness   - over-smoothed (vocoder/TTS) magnitude envelope is flatter
  hf_deficit          - neural vocoders band-limit / roll off high frequencies
  phase_incoherence   - magnitude-only (Griffin-Lim-style) reconstruction leaves
                        inconsistent instantaneous frequency across frames
  jitter_deficit      - synthetic pitch is too regular: cycle-to-cycle period
                        micro-variation (jitter) is abnormally low
  shimmer_deficit     - amplitude micro-variation (shimmer) abnormally low
  hnr_anomaly         - harmonic-to-noise ratio abnormally high (too "clean")
  modulation_anomaly  - temporal energy modulation departs the natural 3-8 Hz
                        syllabic band
  noisefloor_regularity - synthesised pauses/silences are unnaturally clean+flat

All pure numpy; deterministic. Operates on a mono float waveform in [-1, 1].
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

TARGET_SR = 16000


# --------------------------------------------------------------------------- #
# IO + DSP helpers
# --------------------------------------------------------------------------- #


def load_wav(path: str | Path) -> tuple[int, np.ndarray]:
    """Read a PCM WAV file into (sample_rate, mono float32 in [-1, 1]).

    Only the standard-library ``wave`` module is used (PCM 8/16/24/32-bit).
    Non-WAV containers are converted to WAV upstream (see :mod:`audio_io`).
    """
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(n)
    if sw == 1:  # unsigned 8-bit
        a = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 2:
        a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 3:  # packed 24-bit little-endian
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = (b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16))
        v = np.where(v & 0x800000, v - (1 << 24), v)
        a = v.astype(np.float32) / float(1 << 23)
    elif sw == 4:
        a = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / float(1 << 31)
    else:
        raise ValueError(f"unsupported sample width: {sw} bytes")
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return sr, np.ascontiguousarray(a)


def resample(x: np.ndarray, sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    """Linear resampling to ``target_sr`` (adequate for forensic statistics)."""
    if sr == target_sr or x.size == 0:
        return x
    n_out = int(round(x.size * target_sr / sr))
    if n_out <= 1:
        return x
    xp = np.linspace(0.0, 1.0, x.size, endpoint=False)
    xq = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(xq, xp, x).astype(np.float32)


def _hann(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def stft(x: np.ndarray, n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    """Short-time Fourier transform -> complex array [n_frames, n_fft//2+1]."""
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))
    win = _hann(n_fft)
    n_frames = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx] * win[None, :]
    return np.fft.rfft(frames, axis=1)


def istft(spec: np.ndarray, n_fft: int = 1024, hop: int = 256,
          length: int | None = None) -> np.ndarray:
    """Inverse STFT with overlap-add (Hann window, COLA-normalised)."""
    win = _hann(n_fft)
    n_frames = spec.shape[0]
    frames = np.fft.irfft(spec, n=n_fft, axis=1) * win[None, :]
    out_len = n_fft + hop * (n_frames - 1)
    out = np.zeros(out_len, dtype=np.float64)
    wsum = np.zeros(out_len, dtype=np.float64)
    for i in range(n_frames):
        s = i * hop
        out[s:s + n_fft] += frames[i]
        wsum[s:s + n_fft] += win ** 2
    out /= np.maximum(wsum, 1e-8)
    if length is not None:
        out = out[:length] if out.size >= length else np.pad(out, (0, length - out.size))
    return out.astype(np.float32)


def _prep(x: np.ndarray, sr: int) -> np.ndarray:
    x = resample(np.asarray(x, dtype=np.float32), sr, TARGET_SR)
    m = float(np.max(np.abs(x))) if x.size else 0.0
    return x / m if m > 1e-9 else x


def _frame_energy(mag: np.ndarray) -> np.ndarray:
    return (mag ** 2).sum(axis=1)


def _voiced_mask(mag: np.ndarray, pct: float = 60.0) -> np.ndarray:
    """Frames whose energy exceeds the ``pct`` percentile (voiced/loud)."""
    e = _frame_energy(mag)
    if e.size == 0:
        return np.zeros(0, dtype=bool)
    return e >= np.percentile(e, pct)


# --------------------------------------------------------------------------- #
# signals
# --------------------------------------------------------------------------- #


def spectral_flatness(x: np.ndarray, sr: int) -> float:
    """Mean Wiener spectral flatness over voiced frames.

    Vocoder/TTS over-smoothing flattens the harmonic+formant structure, raising
    flatness (geometric/arithmetic mean ratio) toward that of noise.
    """
    g = _prep(x, sr)
    if g.size < 2048:
        return 0.5
    mag = np.abs(stft(g)) + 1e-8
    voiced = _voiced_mask(mag)
    if voiced.sum() < 2:
        return 0.5
    m = mag[voiced]
    flat = np.exp(np.log(m).mean(axis=1)) / m.mean(axis=1)
    val = float(np.nanmean(flat))
    return float(np.clip((val - 0.05) / 0.35, 0.0, 1.0))


def hf_deficit(x: np.ndarray, sr: int) -> float:
    """High-frequency energy deficit.

    Neural vocoders frequently band-limit or roll off energy above ~6-8 kHz,
    leaving less relative energy in the top quarter of the band than a genuine
    wideband recording.
    """
    g = _prep(x, sr)
    if g.size < 2048:
        return 0.5
    mag = np.abs(stft(g))
    voiced = _voiced_mask(mag)
    if voiced.sum() < 2:
        return 0.5
    m = mag[voiced]
    bins = m.shape[1]
    hi = m[:, int(0.7 * bins):].sum()
    tot = m.sum() + 1e-8
    ratio = float(hi / tot)
    return float(np.clip(1.0 - ratio / 0.12, 0.0, 1.0))


def phase_incoherence(x: np.ndarray, sr: int) -> float:
    """Instantaneous-frequency deviation across frames.

    For genuine tonal/voiced content the phase advance between consecutive STFT
    frames equals the bin's centre-frequency advance. Magnitude-only (Griffin-
    Lim-style) reconstruction breaks this, scattering the instantaneous-
    frequency deviation. We measure the dispersion of the wrapped phase
    advance over high-energy bins of voiced frames.
    """
    g = _prep(x, sr)
    n_fft, hop = 1024, 256
    if g.size < n_fft * 4:
        return 0.5
    spec = stft(g, n_fft, hop)
    mag = np.abs(spec)
    phase = np.angle(spec)
    if phase.shape[0] < 3:
        return 0.5
    bins = np.arange(mag.shape[1])
    expected = 2.0 * np.pi * hop * bins / n_fft           # per-frame phase advance
    dphi = phase[1:] - phase[:-1] - expected[None, :]
    dphi = np.angle(np.exp(1j * dphi))                    # wrap to (-pi, pi]
    # weight by magnitude so silent bins do not dominate
    w = mag[1:]
    thr = np.percentile(w, 80)
    sel = w >= max(thr, 1e-8)
    if sel.sum() < 8:
        return 0.5
    disp = float(np.sqrt((dphi[sel] ** 2).mean()))        # rad
    return float(np.clip(disp / 1.8, 0.0, 1.0))


def _frame_f0(frame: np.ndarray, sr: int, fmin: float = 70.0,
              fmax: float = 400.0) -> float:
    """Autocorrelation pitch estimate for one frame; 0.0 if unvoiced."""
    f = frame - frame.mean()
    if np.dot(f, f) < 1e-6:
        return 0.0
    ac = np.correlate(f, f, mode="full")[len(f) - 1:]
    lo, hi = int(sr / fmax), int(sr / fmin)
    if hi >= len(ac) or lo < 1:
        return 0.0
    seg = ac[lo:hi]
    lag = lo + int(np.argmax(seg))
    if ac[lag] < 0.3 * ac[0]:        # weak periodicity -> unvoiced
        return 0.0
    return sr / lag


def _f0_track(g: np.ndarray, sr: int, win: int = 1024, hop: int = 256) -> np.ndarray:
    n_frames = max(0, 1 + (len(g) - win) // hop)
    out = []
    for i in range(n_frames):
        out.append(_frame_f0(g[i * hop:i * hop + win], sr))
    return np.asarray(out)


def jitter_deficit(x: np.ndarray, sr: int) -> float:
    """Pitch-period jitter deficit.

    Natural phonation has ~0.3-2% cycle-to-cycle period variation. Synthetic
    voices are abnormally periodic; an unusually *low* jitter is a synthesis
    cue. We track F0 over voiced frames and measure relative successive
    variation of the period.
    """
    g = _prep(x, sr)
    if g.size < 4096:
        return 0.5
    f0 = _f0_track(g, sr)
    voiced = f0 > 0
    if voiced.sum() < 6:
        return 0.5
    period = 1.0 / f0[voiced]
    d = np.abs(np.diff(period))
    jitter = float(d.mean() / (period.mean() + 1e-9))
    return float(np.clip(1.0 - jitter / 0.02, 0.0, 1.0))


def shimmer_deficit(x: np.ndarray, sr: int) -> float:
    """Amplitude (shimmer) micro-variation deficit over voiced frames."""
    g = _prep(x, sr)
    if g.size < 4096:
        return 0.5
    mag = np.abs(stft(g))
    e = np.sqrt(_frame_energy(mag) + 1e-12)
    voiced = _voiced_mask(mag)
    if voiced.sum() < 6:
        return 0.5
    ev = e[voiced]
    d = np.abs(np.diff(ev))
    shimmer = float(d.mean() / (ev.mean() + 1e-9))
    return float(np.clip(1.0 - shimmer / 0.18, 0.0, 1.0))


def hnr_anomaly(x: np.ndarray, sr: int) -> float:
    """Harmonic-to-noise ratio abnormally high ("too clean").

    Real microphone speech carries aspiration/environmental noise between the
    harmonics. Some synthesisers produce an unnaturally high harmonic-to-noise
    ratio. HNR is estimated from the normalised autocorrelation peak per voiced
    frame.
    """
    g = _prep(x, sr)
    if g.size < 4096:
        return 0.5
    win, hop = 1024, 256
    n_frames = max(0, 1 + (len(g) - win) // hop)
    hnrs = []
    for i in range(n_frames):
        fr = g[i * hop:i * hop + win]
        f0 = _frame_f0(fr, sr)
        if f0 <= 0:
            continue
        f = fr - fr.mean()
        ac = np.correlate(f, f, mode="full")[len(f) - 1:]
        lag = int(round(sr / f0))
        if lag < 1 or lag >= len(ac):
            continue
        r = ac[lag] / (ac[0] + 1e-9)
        r = min(max(r, 1e-4), 0.999)
        hnrs.append(10.0 * np.log10(r / (1.0 - r)))
    if len(hnrs) < 6:
        return 0.5
    hnr = float(np.median(hnrs))
    return float(np.clip((hnr - 5.0) / 15.0, 0.0, 1.0))


def modulation_anomaly(x: np.ndarray, sr: int) -> float:
    """Departure of the temporal-envelope modulation from the syllabic band.

    Natural speech concentrates energy-envelope modulation around 3-8 Hz
    (syllable rate). Synthetic prosody can shift this distribution. Score is the
    deviation of the in-band modulation fraction from a natural reference.
    """
    g = _prep(x, sr)
    hop = 256
    if g.size < 8192:
        return 0.5
    mag = np.abs(stft(g, 1024, hop))
    env = np.sqrt(_frame_energy(mag) + 1e-12)
    env = env - env.mean()
    if env.size < 16:
        return 0.5
    fr = sr / hop                                  # envelope sample rate (Hz)
    spec = np.abs(np.fft.rfft(env * _hann(env.size)))
    freqs = np.fft.rfftfreq(env.size, d=1.0 / fr)
    band = (freqs >= 3.0) & (freqs <= 8.0)
    tot = spec.sum() + 1e-9
    frac = float(spec[band].sum() / tot)
    return float(np.clip(abs(frac - 0.45) / 0.45, 0.0, 1.0))


def noisefloor_regularity(x: np.ndarray, sr: int) -> float:
    """Unnaturally clean and flat noise floor in low-energy frames.

    Genuine recordings keep a textured background (room tone, self-noise) even
    in pauses; synthesised silence is near-digital-zero and spectrally flat. We
    inspect the quietest 20% of frames: an abnormally low and uniform residual
    energy is a synthesis cue.
    """
    g = _prep(x, sr)
    if g.size < 4096:
        return 0.5
    mag = np.abs(stft(g))
    e = _frame_energy(mag)
    if e.size < 8:
        return 0.5
    quiet = e <= np.percentile(e, 20)
    if quiet.sum() < 3:
        return 0.5
    eq = e[quiet]
    med_e = float(np.median(eq))
    tot_med = float(np.median(e) + 1e-12)
    rel = med_e / tot_med                      # quiet/median energy ratio
    cv = float(eq.std() / (eq.mean() + 1e-12))  # low CV -> very uniform silence
    quietness = np.clip(1.0 - rel / 0.02, 0.0, 1.0)
    uniformity = np.clip(1.0 - cv / 0.6, 0.0, 1.0)
    return float(np.clip(0.6 * quietness + 0.4 * uniformity, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# combiner
# --------------------------------------------------------------------------- #


SIGNAL_NAMES = ("spectral_flatness", "hf_deficit", "phase_incoherence",
                "jitter_deficit", "shimmer_deficit", "hnr_anomaly",
                "modulation_anomaly", "noisefloor_regularity")
DEFAULT_WEIGHTS = (0.15, 0.15, 0.20, 0.15, 0.10, 0.10, 0.05, 0.10)

_SIGNAL_FNS = {
    "spectral_flatness": spectral_flatness,
    "hf_deficit": hf_deficit,
    "phase_incoherence": phase_incoherence,
    "jitter_deficit": jitter_deficit,
    "shimmer_deficit": shimmer_deficit,
    "hnr_anomaly": hnr_anomaly,
    "modulation_anomaly": modulation_anomaly,
    "noisefloor_regularity": noisefloor_regularity,
}


def all_scores(x: np.ndarray, sr: int) -> dict[str, float]:
    return {name: float(fn(x, sr)) for name, fn in _SIGNAL_FNS.items()}


def detect(x: np.ndarray, sr: int,
           weights: tuple[float, ...] = DEFAULT_WEIGHTS) -> dict:
    s = all_scores(x, sr)
    combined = float(sum(w * s[name] for w, name in zip(weights, SIGNAL_NAMES)))
    return {**s, "combined": combined,
            "verdict": "synthetic" if combined > 0.5 else "real"}


if __name__ == "__main__":
    # Self-test: a clean harmonic tone (no jitter, flat noise floor) should
    # read more "synthetic" than the same tone with natural jitter + noise.
    rng = np.random.default_rng(0)
    sr = TARGET_SR
    t = np.arange(int(2.0 * sr)) / sr
    clean = sum(np.sin(2 * np.pi * 140 * k * t) / k for k in range(1, 8))
    natural = clean.copy()
    natural += 0.02 * rng.standard_normal(t.size)            # broadband noise
    print("clean  :", {k: round(v, 3) for k, v in all_scores(clean, sr).items()})
    print("natural:", {k: round(v, 3) for k, v in all_scores(natural, sr).items()})
    print("clean combined  :", round(detect(clean, sr)["combined"], 3))
    print("natural combined:", round(detect(natural, sr)["combined"], 3))
