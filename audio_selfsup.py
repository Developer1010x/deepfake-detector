"""Self-supervised pseudo-label generation for label-free audio-deepfake detection.

The audio analogue of :mod:`selfsup`. We need *no labelled synthetic speech*.
Instead we build an unlabelled corpus of natural-sounding voiced utterances with
a source-filter model (real pitch jitter, formant structure, broadband
aspiration noise, textured pauses, coherent phase), treat it as the "real"
manifold, and synthesise two views of each source:

  * **pseudo-real**  - benign edits a genuine recording undergoes (mild gain,
    a little additive noise, a light resample round-trip).
  * **pseudo-fake**  - the same utterance with one or two *neural-synthesis*
    artifacts injected, then the *same* benign edit, so the classifier cannot
    separate the classes on the benign distribution alone.

The five pseudo-fake transforms emulate, in a controlled self-supervised way,
the forensic traces real TTS / voice-conversion / vocoder pipelines leave:

  griffin_lim_phase   - magnitude-only re-synthesis (phase incoherence)
  mel_oversmooth      - spectral-envelope over-smoothing (flattened formants)
  band_limit          - high-frequency roll-off / band-limiting of vocoders
  harmonic_comb       - comb reinforcement -> abnormally high harmonic regularity
  noise_gate          - digitally clean, near-zero synthesised silences

The corpus generator is a *stand-in* (procedural rather than recorded), exactly
as the image pipeline uses a small image set as a corpus stand-in: the protocol
is identical for any real unlabelled speech directory. All numpy, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from audio_detect import TARGET_SR, istft, resample, stft

SR = TARGET_SR


# --------------------------------------------------------------------------- #
# natural-speech corpus synthesis (the unlabelled "real" manifold)
# --------------------------------------------------------------------------- #


def _formant_envelope(freqs: np.ndarray, formants: list[tuple[float, float]]) -> np.ndarray:
    """Sum of Lorentzian resonances at the given (centre, bandwidth) formants."""
    env = np.full_like(freqs, 0.02)
    for fc, bw in formants:
        env += 1.0 / (1.0 + ((freqs - fc) / (bw / 2.0)) ** 2)
    return env


def _voiced(dur: float, f0: float, formants, rng: np.random.Generator) -> np.ndarray:
    """One voiced segment via additive source-filter synthesis with real jitter
    and shimmer and coherent phase."""
    n = int(dur * SR)
    if n < 64:
        return np.zeros(0, dtype=np.float32)
    # F0 contour: slow drift + vibrato + random-walk micro-jitter (~0.5-1%).
    drift = f0 * (1.0 + 0.04 * np.sin(2 * np.pi * rng.uniform(0.5, 1.5) * np.arange(n) / SR))
    vibrato = 1.0 + 0.01 * np.sin(2 * np.pi * rng.uniform(4, 7) * np.arange(n) / SR)
    jitter = 1.0 + 0.008 * np.cumsum(rng.standard_normal(n)) / np.sqrt(np.arange(1, n + 1))
    f0_t = drift * vibrato * jitter
    phase = 2.0 * np.pi * np.cumsum(f0_t) / SR
    # Shimmer: slow random-walk amplitude modulation (~4%).
    shimmer = 1.0 + 0.04 * np.cumsum(rng.standard_normal(n)) / np.sqrt(np.arange(1, n + 1))
    sig = np.zeros(n, dtype=np.float64)
    k = 1
    while k * f0 < 0.95 * (SR / 2):
        fk = k * f0
        amp = (1.0 / k) * float(_formant_envelope(np.array([fk]), formants)[0])
        sig += amp * np.sin(k * phase + rng.uniform(0, 2 * np.pi))
        k += 1
    sig *= shimmer
    # Aspiration noise (broadband, high-passed) keeps genuine HF energy.
    noise = rng.standard_normal(n)
    noise -= np.convolve(noise, np.ones(5) / 5, mode="same")   # crude high-pass
    sig += 0.03 * noise
    # Smooth onset/offset.
    ramp = min(int(0.01 * SR), n // 2)
    if ramp > 0:
        w = np.ones(n)
        w[:ramp] = np.linspace(0, 1, ramp)
        w[-ramp:] = np.linspace(1, 0, ramp)
        sig *= w
    return sig.astype(np.float32)


def _pause(dur: float, rng: np.random.Generator) -> np.ndarray:
    """Textured room-tone pause (genuine recordings are never digital silence)."""
    n = int(dur * SR)
    return (0.004 * rng.standard_normal(n)).astype(np.float32) if n > 0 else np.zeros(0, np.float32)


def synth_utterance(rng: np.random.Generator) -> np.ndarray:
    """A natural-sounding utterance: alternating voiced segments and pauses with
    varying pitch and formants (mimicking syllables/phonemes)."""
    f0 = rng.uniform(95, 210)
    parts = [_pause(rng.uniform(0.08, 0.2), rng)]
    n_syl = rng.integers(4, 8)
    for _ in range(int(n_syl)):
        formants = [(rng.uniform(450, 850), rng.uniform(80, 130)),
                    (rng.uniform(1100, 2200), rng.uniform(120, 220)),
                    (rng.uniform(2500, 3500), rng.uniform(180, 300))]
        seg_f0 = f0 * rng.uniform(0.9, 1.12)
        parts.append(_voiced(rng.uniform(0.12, 0.26), seg_f0, formants, rng))
        if rng.random() < 0.6:
            parts.append(_pause(rng.uniform(0.04, 0.14), rng))
    x = np.concatenate(parts)
    m = float(np.max(np.abs(x))) if x.size else 0.0
    return (x / m * 0.9).astype(np.float32) if m > 1e-9 else x


def synth_corpus(n_sources: int = 40, seed: int = 0) -> list[tuple[str, np.ndarray]]:
    """Build ``n_sources`` named natural-speech utterances (the corpus stand-in)."""
    rng = np.random.default_rng(seed)
    return [(f"utt{i:03d}", synth_utterance(rng)) for i in range(n_sources)]


# --------------------------------------------------------------------------- #
# pseudo-fake transforms (neural-synthesis artifact injection)
# --------------------------------------------------------------------------- #


def griffin_lim_phase(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Magnitude-only re-synthesis: discard phase, recover it iteratively."""
    n_fft, hop = 1024, 256
    mag = np.abs(stft(x, n_fft, hop))
    angle = rng.uniform(-np.pi, np.pi, size=mag.shape)
    spec = mag * np.exp(1j * angle)
    for _ in range(int(rng.integers(2, 6))):
        y = istft(spec, n_fft, hop, length=len(x))
        ang = np.angle(stft(y, n_fft, hop))
        m = min(mag.shape[0], ang.shape[0])
        spec = mag[:m] * np.exp(1j * ang[:m])
    return istft(spec, n_fft, hop, length=len(x))


def mel_oversmooth(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Over-smooth the magnitude envelope along frequency (formant flattening)."""
    n_fft, hop = 1024, 256
    spec = stft(x, n_fft, hop)
    mag, phase = np.abs(spec), np.angle(spec)
    k = int(rng.choice([5, 7, 9, 11]))
    ker = np.ones(k) / k
    sm = np.apply_along_axis(lambda r: np.convolve(r, ker, mode="same"), 1, mag)
    return istft(sm * np.exp(1j * phase), n_fft, hop, length=len(x))


def band_limit(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Roll off energy above a vocoder-style cutoff (5.5-7.5 kHz)."""
    n_fft, hop = 1024, 256
    spec = stft(x, n_fft, hop)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / SR)
    cutoff = rng.uniform(5500, 7500)
    roll = 1.0 / (1.0 + (freqs / cutoff) ** 8)             # soft brick-wall
    return istft(spec * roll[None, :], n_fft, hop, length=len(x))


def harmonic_comb(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Comb-reinforce a fixed period -> abnormally regular harmonic structure."""
    period = int(SR / rng.uniform(110, 190))
    if period < 2 or period >= len(x):
        return x
    a = rng.uniform(0.5, 0.85)
    y = x.copy()
    y[period:] += a * x[:-period]
    y[:-period] += a * x[period:]
    m = float(np.max(np.abs(y))) or 1.0
    return (y / m * (float(np.max(np.abs(x))) or 1.0)).astype(np.float32)


def noise_gate(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Hard noise gate -> digitally clean (near-zero) synthesised silences."""
    n_fft, hop = 1024, 256
    mag = np.abs(stft(x, n_fft, hop))
    e = (mag ** 2).sum(axis=1)
    thr = np.percentile(e, rng.uniform(25, 40))
    gain = np.ones(len(x), dtype=np.float32)
    for i, en in enumerate(e):
        if en < thr:
            s = i * hop
            gain[s:s + n_fft] *= 0.03
    return (x * gain).astype(np.float32)


_FAKE_TRANSFORMS = {
    "griffin_lim_phase": griffin_lim_phase,
    "mel_oversmooth": mel_oversmooth,
    "band_limit": band_limit,
    "harmonic_comb": harmonic_comb,
    "noise_gate": noise_gate,
}


# --------------------------------------------------------------------------- #
# benign (pseudo-real) augmentation
# --------------------------------------------------------------------------- #


def benign_augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = x.astype(np.float32)
    if rng.random() < 0.8:                                  # mild gain
        out = out * rng.uniform(0.85, 1.1)
    if rng.random() < 0.7:                                  # a little real noise
        out = out + rng.uniform(0.002, 0.01) * rng.standard_normal(out.size).astype(np.float32)
    if rng.random() < 0.5:                                  # light resample round-trip
        inter = rng.uniform(0.9, 1.1)
        out = resample(resample(out, SR, int(SR * inter)), int(SR * inter), SR)
    m = float(np.max(np.abs(out))) if out.size else 0.0
    return (out / m * 0.9).astype(np.float32) if m > 1e-9 else out


# --------------------------------------------------------------------------- #
# sample generation
# --------------------------------------------------------------------------- #


@dataclass
class PseudoAudioSample:
    waveform: np.ndarray
    sr: int
    label: int          # 0 = pseudo-real, 1 = pseudo-fake
    source: str
    transforms: list[str] = field(default_factory=list)


def make_pseudo_fake(x: np.ndarray, rng: np.random.Generator,
                     n_ops: tuple[int, int] = (1, 2)) -> tuple[np.ndarray, list[str]]:
    k = int(rng.integers(n_ops[0], n_ops[1] + 1))
    names = list(rng.choice(list(_FAKE_TRANSFORMS), size=k, replace=False))
    out = x
    for name in names:
        out = _FAKE_TRANSFORMS[name](out, rng)
        m = float(np.max(np.abs(out))) if out.size else 0.0
        if m > 1e-9:
            out = (out / m * 0.9).astype(np.float32)
    return out, names


def generate(sources: list[tuple[str, np.ndarray]], per_source: int = 4,
             seed: int = 0) -> list[PseudoAudioSample]:
    """Balanced self-supervised dataset: ``per_source`` pseudo-real and
    ``per_source`` pseudo-fake views per source, no human labels."""
    rng = np.random.default_rng(seed + 777)
    out: list[PseudoAudioSample] = []
    for name, base in sources:
        for _ in range(per_source):
            real = benign_augment(base, rng)
            out.append(PseudoAudioSample(real, SR, 0, name, ["benign"]))
            fake, ops = make_pseudo_fake(base, rng)
            fake = benign_augment(fake, rng)               # same benign edit on both
            out.append(PseudoAudioSample(fake, SR, 1, name, ops))
    return out


if __name__ == "__main__":
    import audio_detect as ad

    corpus = synth_corpus(6, seed=0)
    samples = generate(corpus, per_source=2, seed=0)
    print(f"{len(corpus)} sources -> {len(samples)} samples "
          f"({sum(s.label == 0 for s in samples)} real / "
          f"{sum(s.label == 1 for s in samples)} fake)")
    real = [s for s in samples if s.label == 0][:4]
    fake = [s for s in samples if s.label == 1][:4]
    rc = np.mean([ad.detect(s.waveform, s.sr)["combined"] for s in real])
    fc = np.mean([ad.detect(s.waveform, s.sr)["combined"] for s in fake])
    print(f"mean classical combined  real={rc:.3f}  fake={fc:.3f}")
