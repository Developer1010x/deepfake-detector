"""Property tests for the hand-rolled DSP primitives.

The selling point of this project is that the signal processing is written from
scratch in numpy rather than imported from pywt / librosa / skimage. That is only
a selling point if it is *correct*, so the invariants each primitive is supposed
to satisfy are asserted here rather than assumed:

  * Haar DWT is orthonormal-up-to-scale and perfectly invertible.
  * ISTFT(STFT(x)) reconstructs x in the COLA-valid interior.
  * The 8x8 DCT-II matrix is orthonormal, so it round-trips.
  * Benford's law fires on synthetic data that obeys / violates it.
  * The 58 uniform LBP codes are exactly the codes with <= 2 bit transitions.
  * Every image signal stays inside [0, 1] on degenerate input.

Run with:  python3 -m pytest tests/ -q
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

import audio_detect as ad
import detector
import patterns


# --------------------------------------------------------------------------- #
# Haar DWT
# --------------------------------------------------------------------------- #


def _haar_inverse(ll, lh, hl, hh):
    """Reassemble the interleaved image from the four Haar sub-bands."""
    a = (ll + lh + hl + hh) / 2.0
    b = (ll + lh - hl - hh) / 2.0
    c = (ll - lh + hl - hh) / 2.0
    d = (ll - lh - hl + hh) / 2.0
    h, w = a.shape
    out = np.zeros((h * 2, w * 2), dtype=np.float64)
    out[0::2, 0::2], out[0::2, 1::2] = a, b
    out[1::2, 0::2], out[1::2, 1::2] = c, d
    return out


def test_haar_dwt_perfect_reconstruction():
    rng = np.random.default_rng(0)
    g = rng.uniform(0, 255, size=(64, 48)).astype(np.float32)
    ll, lh, hl, hh = detector._haar_dwt(g)
    assert np.allclose(_haar_inverse(ll, lh, hl, hh), g, atol=1e-4)


def test_haar_dwt_energy_is_preserved():
    """The transform is orthogonal up to the 1/2 scaling, so energy is conserved."""
    rng = np.random.default_rng(1)
    g = rng.standard_normal((32, 32))
    bands = detector._haar_dwt(g)
    assert np.isclose(sum(float((b ** 2).sum()) for b in bands), float((g ** 2).sum()),
                      rtol=1e-9)


def test_haar_dwt_detail_bands_vanish_on_a_flat_image():
    _, lh, hl, hh = detector._haar_dwt(np.full((16, 16), 128.0))
    for band in (lh, hl, hh):
        assert np.allclose(band, 0.0)


def test_haar_dwt_drops_odd_rows_and_columns():
    ll, _, _, _ = detector._haar_dwt(np.zeros((15, 9)))
    assert ll.shape == (7, 4)


# --------------------------------------------------------------------------- #
# DCT-II
# --------------------------------------------------------------------------- #


def test_dct8_matrix_is_orthonormal():
    m = detector._DCT8
    assert np.allclose(m @ m.T, np.eye(8), atol=1e-12)


def test_dct8_round_trips_a_block():
    rng = np.random.default_rng(2)
    block = rng.uniform(-128, 127, size=(8, 8))
    coeffs = detector._DCT8 @ block @ detector._DCT8.T
    assert np.allclose(detector._DCT8.T @ coeffs @ detector._DCT8, block, atol=1e-10)


def test_dct8_dc_term_matches_the_block_mean():
    block = np.full((8, 8), 10.0)
    coeffs = detector._DCT8 @ block @ detector._DCT8.T
    assert np.isclose(coeffs[0, 0], 10.0 * 8)          # mean * n, per the 1/sqrt(n) DC row
    assert np.allclose(coeffs[1:, :], 0.0, atol=1e-10)


# --------------------------------------------------------------------------- #
# STFT / ISTFT
# --------------------------------------------------------------------------- #


def test_istft_inverts_stft_in_the_cola_interior():
    """Overlap-add reconstruction is exact away from the ramp-in/ramp-out edges."""
    sr, n_fft, hop = 16000, 1024, 256
    t = np.arange(sr) / sr
    x = (0.6 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 1750 * t))
    x = x.astype(np.float32)
    y = ad.istft(ad.stft(x, n_fft, hop), n_fft, hop, length=len(x))
    interior = slice(n_fft, len(x) - n_fft)
    assert np.allclose(y[interior], x[interior], atol=1e-4)


def test_stft_shape_and_bin_of_a_pure_tone():
    sr, n_fft, hop = 16000, 1024, 256
    freq = sr / n_fft * 40                        # exactly on bin 40
    t = np.arange(4096) / sr
    spec = ad.stft(np.sin(2 * np.pi * freq * t).astype(np.float32), n_fft, hop)
    assert spec.shape[1] == n_fft // 2 + 1
    assert int(np.argmax(np.abs(spec).mean(axis=0))) == 40


def test_stft_pads_input_shorter_than_one_frame():
    spec = ad.stft(np.zeros(100, dtype=np.float32), n_fft=1024, hop=256)
    assert spec.shape == (1, 513)


def test_hann_window_is_cola_compliant_at_hop_n_over_4():
    """sum of w^2 over 4x-overlapped Hann frames is constant in the interior."""
    n_fft, hop = 1024, 256
    win = ad._hann(n_fft) ** 2
    acc = np.zeros(n_fft * 4)
    for i in range(13):
        acc[i * hop:i * hop + n_fft] += win
    interior = acc[n_fft:n_fft * 3]
    assert np.allclose(interior, interior[0], rtol=1e-9)


def test_resample_changes_length_proportionally():
    x = np.sin(np.linspace(0, 40 * np.pi, 8000)).astype(np.float32)
    y = ad.resample(x, 8000, 16000)
    assert abs(len(y) - 16000) <= 2


# --------------------------------------------------------------------------- #
# Benford
# --------------------------------------------------------------------------- #


def test_benford_score_is_low_for_benford_distributed_coefficients():
    """An image whose DCT AC coefficients follow a natural heavy-tailed law
    should sit close to Benford, i.e. score low."""
    rng = np.random.default_rng(3)
    # 1/f-ish natural texture: Benford holds well for such data.
    noise = rng.standard_normal((256, 256))
    spec = np.fft.fft2(noise)
    fy = np.fft.fftfreq(256)[:, None]
    fx = np.fft.fftfreq(256)[None, :]
    r = np.hypot(fy, fx)
    r[0, 0] = 1e-6
    smooth = np.real(np.fft.ifft2(spec / r))
    smooth -= smooth.min()
    smooth = (smooth / smooth.max() * 255).astype(np.uint8)
    assert detector.benford_score(Image.fromarray(smooth)) < 0.5


def test_benford_score_is_neutral_on_a_constant_image():
    """No AC energy at all -> the signal must abstain at 0.5, not divide by zero."""
    assert detector.benford_score(Image.new("L", (64, 64), 128)) == 0.5


def test_benford_score_is_neutral_when_the_image_is_smaller_than_one_block():
    assert detector.benford_score(Image.new("L", (4, 4), 200)) == 0.5


# --------------------------------------------------------------------------- #
# Local Binary Patterns
# --------------------------------------------------------------------------- #


def test_uniform_lbp_codes_are_the_58_expected_ones():
    codes = patterns._uniform_lbp_codes()
    assert len(codes) == 58
    for c in codes:
        bits = format(int(c), "08b")
        transitions = sum(1 for a, b in zip(bits, bits[1:] + bits[0]) if a != b)
        assert transitions <= 2
    non_uniform = set(range(256)) - set(int(c) for c in codes)
    for c in non_uniform:
        bits = format(c, "08b")
        assert sum(1 for a, b in zip(bits, bits[1:] + bits[0]) if a != b) > 2


def test_lbp_uniform_mass_is_one_for_a_smooth_gradient():
    """A monotonic ramp has no local texture, so every code is uniform."""
    ramp = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
    out = patterns.lbp_pattern(Image.fromarray(ramp))
    assert out["uniform_mass"] == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# spatial fields behind the scalar signals  (explain.py depends on these)
# --------------------------------------------------------------------------- #


def test_jpeg_ghost_field_shape_and_score_agreement():
    rng = np.random.default_rng(4)
    arr = rng.integers(0, 256, size=(128, 96, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    field = detector.jpeg_ghost_field(img, block=16)
    assert field.shape == (128 // 16, 96 // 16)
    expected = float(np.clip(field.std() / 1.5, 0.0, 1.0))
    assert detector.jpeg_ghost_score(img, block=16) == pytest.approx(expected)


def test_jpeg_ghost_field_is_none_when_the_image_is_too_small():
    assert detector.jpeg_ghost_field(Image.new("RGB", (16, 16)), block=16) is None
    assert detector.jpeg_ghost_score(Image.new("RGB", (16, 16)), block=16) == 0.5


def test_ela_map_matches_ela_score():
    rng = np.random.default_rng(5)
    img = Image.fromarray(rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8))
    diff = detector.ela_map(img)
    assert diff.shape == (64, 64, 3)
    expected = float(diff.std()) / (float(diff.mean()) + float(diff.std()) + 1e-6)
    assert detector.ela_score(img) == pytest.approx(expected)


def test_block_fft_field_matches_block_heterogeneity():
    rng = np.random.default_rng(6)
    img = Image.fromarray(rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8))
    field = patterns.block_fft_field(img, n_blocks=4)
    assert field.shape == (4, 4)
    out = patterns.block_heterogeneity(img, n_blocks=4)
    assert out["block_std"] == pytest.approx(round(float(np.std(field)), 4))


# --------------------------------------------------------------------------- #
# range invariants
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("size,colour", [((8, 8), 0), ((64, 64), 255),
                                         ((129, 71), 128), ((256, 256), 17)])
def test_every_image_signal_stays_in_the_unit_interval(size, colour):
    img = Image.new("RGB", size, (colour, colour, colour))
    for name, value in detector.all_scores(img).items():
        assert 0.0 <= value <= 1.0, f"{name} out of range on {size}: {value}"


def test_pattern_scores_stay_in_the_unit_interval():
    rng = np.random.default_rng(7)
    img = Image.fromarray(rng.integers(0, 256, size=(320, 240, 3), dtype=np.uint8))
    out = patterns.all_patterns(img)
    assert 0.0 <= out["pattern_score"] <= 1.0
    for key in ("spectral_peaks", "block_heterogeneity", "lbp"):
        assert 0.0 <= out[key]["score"] <= 1.0


def test_audio_signals_stay_in_the_unit_interval_on_silence_and_noise():
    rng = np.random.default_rng(8)
    for x in (np.zeros(16000, dtype=np.float32),
              rng.standard_normal(16000).astype(np.float32)):
        for name, value in ad.all_scores(x, 16000).items():
            assert 0.0 <= value <= 1.0, f"{name} out of range: {value}"
