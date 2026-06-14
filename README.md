# Label-Free Multimodal Synthetic-Media Detector (Image · Audio · Text · Video)

A research-grade, **CPU-only, label-free** detector that estimates whether **an image, an audio clip (synthetic/cloned voice), a text passage (LLM-generated), or a video** was produced by a generative AI model. One methodology — _unlabelled corpus → benign pseudo-real + artifact-injected pseudo-fake → interpretable forensic bank → calibrated classifier, under leave-source-out CV_ — is instantiated per modality. See **[Multimodal detection](#multimodal-detection-audio-text-video)** below and the full write-up in [`paper/paper.md`](paper/paper.md).

Headline results (leave-source-out CV; real-data external validation where available):

| Modality | Self-supervised AUC | External (real data) |
|---|---|---|
| Image (fusion) | **0.853** | 0.585 (CIFAKE, 32px — weak) |
| Audio (learned bank) | **0.957** (in-distribution) | — (no open set shipped) |
| Text (training-free bank) | 0.760 | **0.960** (HC3 real human-vs-ChatGPT) |

The **image** track (the original study) fuses **two complementary evidence families**:

1. **Interpretable forensic statistics** — thirteen hand-crafted signal-processing / statistical cues (the seven pixel signals + the pattern checks documented below). No training data needed; every score maps to a named physical phenomenon.
2. **Self-supervised deep features** — a frozen ImageNet-pretrained backbone applied in a **dual-stream** (spatial image + Fourier spectrum) configuration, combined with the forensic bank by a **calibrated learned classifier**.

The learned classifier is trained **without any labelled deepfakes**. It manufactures its own supervision: from an unlabelled image corpus it synthesises *pseudo-real* (benign edit) and *pseudo-fake* (generation-artifact injection) views and learns to separate them (cf. Self-Blended Images, CVPR 2022). The whole stack runs on **CPU**.

Under a leakage-free leave-source-out protocol, fusion reaches **0.853 ROC-AUC**, beating deep-only (0.822), hand-crafted-only (0.796), and the classical fixed-weight baseline (0.589). See [`paper/paper.md`](paper/paper.md) for the full write-up, methodology, and ablation.

### The AI pipeline at a glance

| Module | Role |
|---|---|
| `detector.py`, `patterns.py` | 13-d interpretable forensic feature bank (classical signals) |
| `deep_features.py` | frozen ResNet-18 dual-stream (spatial + spectral) deep embedding |
| `selfsup.py` | label-free pseudo-real / pseudo-fake generator (5 artifact families) |
| `fusion.py` | hybrid classifier (deep ⊕ hand-crafted), calibrated, joblib-persisted |
| `train_selfsup.py` | self-supervised training entry point → `models/fusion.joblib` |
| `evaluate.py` | leave-source-out CV, ablation, per-artifact analysis, figures → `paper/` |
| `pipeline.py` | unified inference (learned fusion → classical fallback → watermark override) |

### Quick start

```bash
pip install -r requirements.txt           # CPU torch + sklearn + matplotlib + pyarrow ; ffmpeg for audio decode
python3 train_selfsup.py --patch 128      # image: self-supervised training (no labels needed)
python3 train_audio.py                    # audio: train + ship models/audio.joblib
python3 evaluate.py --folds 5             # image metrics + figures into paper/
python3 cli.py image <path>               # inference — also: video / audio / text <path>  (text: - = stdin)
python3 app.py                            # web UI + multimodal API on :5000
```

---

## Multimodal detection (audio, text, video)

The same label-free recipe is instantiated across four modalities. Every track ships an interpretable forensic bank, a self-supervised pseudo-fake generator, and a leave-source-out evaluation harness.

| Modality | Detector / signals | Self-supervision | Eval harness | Inference |
|---|---|---|---|---|
| **Image** | `detector.py` (7 stats) + `patterns.py` + `deep_features.py` → `fusion.py` | `selfsup.py` (5 image artifacts) | `evaluate.py` | `cli.py image`, web upload |
| **Audio** | `audio_detect.py` (8 voice-forensic signals: spectral flatness, HF deficit, phase incoherence, pitch jitter, shimmer, HNR, modulation, noise-floor) | `audio_selfsup.py` (griffin-lim phase, mel over-smooth, band-limit, harmonic comb, noise gate) | `evaluate_audio.py` | `cli.py audio`, web upload, **fused into video** |
| **Text** | `text_detect.py` (8 LLM signals: burstiness, TTR, n-gram repetition, function-word regularity, punctuation diversity, rare-word rarity, transition density, token entropy) | `text_selfsup.py` (flatten burstiness, inject repetition/boilerplate, lexical smoothing, regularize punctuation) | `evaluate_text.py` | `cli.py text`, `POST /analyze_text` |
| **Video** | per-frame image fusion + temporal instability **+ extracted audio track** (`audio_io.py`) | — | — | `cli.py video`, web upload |

```bash
# new-modality CLI
python3 cli.py audio clip.wav                 # synthetic-voice forensic breakdown + verdict
python3 cli.py text essay.txt                 # AI/LLM-text breakdown  (or: echo "..." | python3 cli.py text -)
python3 cli.py video clip.mp4 --every 10      # visual + temporal + audio-track fusion

# real datasets + external validation (the credibility check)
python3 prepare_datasets.py                   # decode downloaded HC3 (text) + CIFAKE (image) into labelled folders
python3 evaluate_text.py  --real datasets/hc3/human  --fake datasets/hc3/ai      # -> 0.960 AUC on real ChatGPT vs human
python3 evaluate.py --real datasets/cifake/real --fake datasets/cifake/fake      # -> 0.585 AUC on CIFAKE (honest gap)
python3 evaluate_audio.py                                                         # -> 0.957 AUC in-distribution
```

**Honest scope of the new tracks.** The audio corpus is a *procedural stand-in* (no open, redistributable labelled synthetic-speech set was shipped), so its 0.957 AUC is in-distribution and optimistic — read it as the separability of the injected artifacts, not a deployment number. The text detector's strength is **ranking** (0.96 AUC on real HC3); single-sample verdicts at the 0.5 threshold are *indicative* (the training-free combiner is conservative and is the shipped default because it transfers to real data better than the learned model — see `paper/paper.md` §5.3). Datasets live under `datasets/` and are git-ignored.

---

The classical image signals below remain the **interpretable half** of the image detector — every one is a named, physically-motivated feature feeding the fusion model.

---

## What the problem is

A "deepfake" or AI-generated image is one synthesized by a model — GAN (StyleGAN, ProGAN), diffusion (Stable Diffusion, SDXL, Flux, Midjourney, DALL-E), or autoencoder. We want to look at a single image and decide: *was this produced by a camera capturing photons off a real scene, or by a neural network sampling from a learned distribution?*

A camera image carries a long, messy provenance: lens optics, sensor noise (PRNU), color filter array (CFA / Bayer pattern) demosaicing, in-camera ISP processing, JPEG compression. Each step leaves a statistical fingerprint. A generated image went through none of those — it came out of matrix multiplications and nonlinearities. So the two have *different statistics*, even when they look identical to a human. Classical detection is the art of finding those statistical differences.

## Why hybrid (interpretable forensics + self-supervised deep learning)

Production detectors throw a CNN at the problem (Microsoft Video Authenticator, Intel FakeCatcher). A purely classical detector, conversely, is interpretable but shallow. This project takes a **hybrid** stance, keeping the strengths of each:

1. **Interpretability is retained.** Every hand-crafted score corresponds to a named physical or statistical phenomenon, so when the detector fires you can point at *which* property was anomalous and *why* a generator would produce it. These 13 signals are first-class inputs to the fusion model, and `pipeline.py` reports them alongside every verdict.
2. **No labelled deepfakes are needed.** Rather than curating real/fake datasets (laborious, perpetually outdated), the learned classifier is trained **self-supervised**: it synthesises its own pseudo-real / pseudo-fake pairs from any unlabelled image corpus (`selfsup.py`). Supervision is defined by *artifact families*, not by a specific generator.
3. **Reduced model dependence.** A CNN trained on StyleGAN often fails on diffusion images because it memorised *that* generator's fingerprint. By targeting transferable artifact families (up-sampling periodicity, blending boundaries, double compression, flattened noise) and fusing them with frequency-aware deep features, the detector aims to degrade more gracefully against unseen generators.

**Measured benefit of fusion** (leave-source-out CV, full table in [`paper/results.md`](paper/results.md)):

| Model | ROC-AUC | F1 | Brier |
|---|---|---|---|
| classical (fixed weights) | 0.589 | 0.621 | 0.246 |
| hand-crafted (learned) | 0.796 | 0.708 | 0.186 |
| deep (dual-stream) | 0.822 | 0.736 | 0.179 |
| **fusion** | **0.853** | **0.771** | **0.165** |

The honest caveats live in the paper's *Limitations*: the self-supervised model detects the artifact families it synthesises, performance scales with corpus size, and adversarial/anti-forensic robustness is unevaluated. Use it as a strong, reproducible, explainable baseline — and validate on a labelled benchmark (`evaluate.py --real DIR --fake DIR`) before trusting it in production.

---

## The seven signals

The detector computes seven independent scores in `[0, 1]`. Higher = more synthetic-looking. The combined score is a weighted sum (default weights below) or, when labeled data is available, a logistic-regression weighting fitted by `tune.py`. Threshold at 0.5.

```
SIGNAL_NAMES    = ("fft", "ela", "benford", "cfa", "wavelet", "jpeg_ghost", "noise")
DEFAULT_WEIGHTS = (0.20,  0.10,  0.20,      0.15,  0.10,      0.10,         0.15)
```

### 1. FFT spectral analysis (`fft_score`)

**What it measures.** The radial power spectrum of the 2D Fourier transform, and how much it deviates from a smooth power-law decay at high frequencies.

**The physics.** Natural scenes, going back to Burton & Moorhead (1987) and Field (1987), have a remarkably consistent statistical property: the average power spectrum follows approximately

```
S(f) ~ 1 / f^α       (typically α ≈ 1.8 to 2.2)
```

where `f` is spatial frequency. This holds across photos of forests, faces, cities, microscopy — anywhere natural light hits a sensor. It is one of the most robust statistics of "natural images" we know of.

**Why generators violate it.** Almost every image-generation architecture has to upscale a low-resolution latent into a high-resolution image. The standard tools — transposed convolution, nearest-neighbor upsample + conv, sub-pixel conv — all introduce *periodic patterns* at high frequencies because they tile the same kernel across the image at a fixed stride. In the Fourier domain, periodic spatial patterns become *spectral peaks*. So a GAN's radial spectrum doesn't decay smoothly; it has bumps, ringing, or a flat shoulder where the upsampler's stride sets a characteristic frequency.

**Implementation.**
1. Convert to grayscale, center-crop to a square.
2. Compute `log(1 + |FFT2|)` (log-magnitude spectrum).
3. Compute the *radial profile* — average magnitude at each radius from the center frequency, via `np.bincount` weighted by spectrum values.
4. Take the high-frequency half of the radial profile.
5. Fit a linear trend (in log-radius) and compute the standard deviation of the residual.
6. Score = `std(residual) / std(profile)`, clipped to `[0, 1]`.

A smooth power-law decay produces tiny residuals (low score). Spectral peaks produce big residuals (high score).

**Key references.** Zhang, Karaman, Chang — *"Detecting and Simulating Artifacts in GAN Fake Images"* (WIFS 2019). Frank et al. — *"Leveraging Frequency Analysis for Deep Fake Image Recognition"* (ICML 2020).

### 2. Error Level Analysis (`ela_score`)

**What it measures.** Re-save the image as JPEG at a known quality, then compute the per-pixel difference from the original. Look at how *uneven* that error is.

**The physics.** JPEG compression is lossy and idempotent at convergence: once a region has been JPEG-compressed at quality Q, re-saving at the same Q changes it very little. So a pristine, single-pass JPEG photo has *uniform* re-save error — the difference image is roughly flat noise. But an image that was *spliced* from two sources, or had a region painted over and re-saved at a different quality, will show *hot spots* where the two compression histories meet.

For fully AI-generated images, the situation is subtler. Many generators output uncompressed PNG, which is then JPEG'd once at upload. That single JPEG step gives uniform error — *low* ELA, like an unedited photo. But generators that output JPEG directly, or images that have been edited in Photoshop after generation, show characteristic ELA patterns.

**Implementation.**
1. Convert to RGB.
2. Save to an in-memory buffer as JPEG at quality 90.
3. Reload and diff with the original.
4. Score = `std(diff) / (mean(diff) + std(diff) + ε)`. This is a coefficient-of-variation style metric in `[0, 1)`. Low std/mean = uniform error. High std with low mean = localized hot spots.

**Limitations.** ELA is famously noisy. If the input has been re-saved through JPEG multiple times at decreasing quality, you'll get hot-looking results from a perfectly real photo. That's why ELA carries the *lowest* default weight (0.2) in the combined score.

**Key reference.** Krawetz — *"A Picture's Worth"* (Black Hat 2007). The original ELA writeup, still the clearest one.

### 3. Benford's law on DCT coefficients (`benford_score`)

**What it measures.** The chi-squared distance between the actual distribution of leading digits in 8×8-block DCT AC coefficients and the distribution predicted by Benford's law.

**The math.** Benford's law (Newcomb 1881; Benford 1938) says that in many natural distributions, the leading digit `d ∈ {1, ..., 9}` follows

```
P(d) = log10(1 + 1/d)
```

So `P(1) ≈ 30.1%`, `P(2) ≈ 17.6%`, ..., `P(9) ≈ 4.6%`. This holds for populations of cities, lengths of rivers, stock prices, physical constants — and, as Pérez-González and collaborators showed in the 2000s, for the magnitudes of AC DCT coefficients of natural images.

The intuition for why image DCTs obey Benford: AC coefficients are roughly distributed as a generalized Gaussian / Laplacian centered at zero with a heavy tail spanning many orders of magnitude. *Any* distribution that spans multiple decades and has roughly log-uniform mass approximates Benford. Natural image statistics happen to satisfy this beautifully.

**Why generators violate it.** Generated images don't go through the same long imaging pipeline that produces these natural DCT statistics. The decoder of a VAE, the upsampler stack of a GAN, or the iterative denoising of a diffusion model produces correlated pixel structure that, when DCT'd, has a *different* coefficient distribution — often more concentrated around small values, or with a different tail shape. The leading-digit distribution shifts noticeably away from Benford.

**Implementation.**
1. Convert to grayscale, crop to a multiple of 8, subtract 128 (DC center).
2. Reshape into 8×8 blocks.
3. Compute the DCT-II of each block via the closed-form DCT matrix `D` (so `coeffs = D @ blocks @ Dᵀ`, fully vectorized in numpy — no per-block Python loop).
4. Take all AC coefficients (skip the DC component at `[0,0]`).
5. Take magnitudes ≥ 1.
6. Extract the leading digit of each: `d = floor(|c| / 10^floor(log10|c|))`.
7. Build the empirical histogram, normalize.
8. Compute chi-squared distance to the Benford expected vector.
9. Score = `min(1, chi² / 0.5)`. The 0.5 cap is empirical.

This is the most discriminative of the three signals on the test images so far. It carries the second-highest default weight (0.3).

**Key reference.** Pérez-González, Heileman, Abdallah — *"Benford's Law in Image Processing"* (ICIP 2007). And the broader literature on Benford detection of image manipulation.

### 4. CFA / Bayer demosaicing trace (`cfa_score`)

**What it measures.** Whether the image carries the periodic 2×2 fingerprint left by camera demosaicing.

**The physics.** Almost every consumer camera sensor uses a single chip with a Color Filter Array — most commonly a Bayer pattern (R-G-G-B in 2×2 tiles). Each pixel records only one of the three colors; the other two are interpolated by a *demosaicing* algorithm (bilinear, gradient-corrected, AHD, etc.). After demosaicing, "real" sample positions and "interpolated" positions have systematically different statistical properties — interpolated pixels are smoother, with smaller prediction residuals.

This shows up as variance asymmetry across the 2×2 sub-grids of the prediction residual. AI-generated images, having never been demosaiced, have approximately uniform variance across all four sub-grids.

**Implementation.**
1. For each color channel, predict every pixel from its 4 neighbors: `pred[i,j] = (im[i±1,j] + im[i,j±1]) / 4`.
2. Compute the residual `res = im - pred`.
3. Compute the variance of the residual on each of the four 2×2 sub-grids: `var(res[0::2, 0::2])`, `var(res[0::2, 1::2])`, etc.
4. Coefficient of variation across the four sub-grid variances measures asymmetry.
5. Score = `exp(-3 * mean_CV)`. Uniform variance (no demosaicing trace) → score near 1 (synthetic). Asymmetric variance (real Bayer trace) → score near 0.

**Limitation.** JPEG compression at quality < 95 partially destroys the Bayer trace because the 8×8 DCT block structure dominates over the 2×2 demosaicing structure. So this signal is most reliable on raw or lightly-compressed inputs and gets noisy on heavily-recompressed JPEGs (where it can falsely flag real photos as synthetic). The default weights account for this.

**Key references.** Popescu & Farid — *"Exposing Digital Forgeries in Color Filter Array Interpolated Images"* (IEEE TSP 2005). Dirik & Memon — *"Image tamper detection based on demosaicing artifacts"* (ICIP 2009).

### 5. Wavelet sub-band kurtosis (`wavelet_score`)

**What it measures.** Whether the detail-band coefficients of a Haar wavelet decomposition have the heavy-tailed distribution characteristic of natural images.

**The physics.** When you apply a wavelet transform to a natural photo, the detail sub-bands (LH = horizontal detail, HL = vertical detail, HH = diagonal detail) end up *highly sparse*: most coefficients are near zero, with occasional large values at edges and textures. The marginal distribution is heavy-tailed — well-modeled by a generalized Gaussian with shape parameter much less than 2, or equivalently, kurtosis much greater than 3. Mallat (1989) and Field (1987) made this one of the foundational observations of natural image statistics; whole compression schemes (JPEG2000, EZW, SPIHT) and texture models (Simoncelli's pyramid) are built on it.

Synthetic images, especially from generators with smooth latent decoders, often have *less* heavy-tailed wavelet stats — kurtosis closer to the Gaussian value of 3.

**Implementation.**
1. Manually compute one level of 2D Haar DWT via numpy slicing — sums and differences of 2×2 blocks. No `pywt` dependency.
2. For each of the three detail bands (LH, HL, HH), compute the kurtosis of the coefficient distribution: `μ_4 / σ^4`.
3. Average across the three bands.
4. Score = `clip(1 - (avg_kurtosis - 3) / 20, 0, 1)`. High kurtosis (natural) → low score; near-Gaussian kurtosis → high score.

**Limitation.** Kurtosis alone is a coarse summary of the distribution shape. A more rigorous version would fit a generalized Gaussian via maximum likelihood and use the shape parameter β as the score. Out of scope for this small project, but a natural extension.

**Key references.** Mallat — *"A Theory for Multiresolution Signal Decomposition"* (PAMI 1989). Field — *"Relations between the statistics of natural images..."* (1987). Simoncelli & Olshausen — *"Natural Image Statistics and Neural Representation"* (Annual Review of Neuroscience 2001).

### 6. JPEG ghost (`jpeg_ghost_score`)

**What it measures.** Spatial inconsistency in which JPEG quality factor minimizes block-level re-encoding error.

**The physics.** When you re-encode a JPEG at quality Q, the per-block reconstruction error reaches a minimum where Q matches the *original* compression quality (Farid 2009 — the "JPEG ghost" effect). For an unmodified single-source image, every 16×16 block agrees on which quality is best, so the "best-quality field" is spatially flat. For a spliced image (two regions with different JPEG histories) or for a generated image (which doesn't have a single coherent JPEG history at all), different blocks land on different best qualities, giving the field high spatial variance.

**Implementation.**
1. Re-encode the image at multiple quality factors (default: 60, 70, 80, 90).
2. For each quality, compute the squared error per 16×16 block.
3. For each block, find the quality index that minimizes its error.
4. Score = `std(min_quality_index)` across blocks, normalized to `[0, 1]`.

**Limitation.** Pure PNG inputs have no JPEG history at all, so the test compares re-encoding errors that were all introduced *by this analysis*. Behavior on PNG inputs is dominated by content rather than provenance. Most useful on JPEG inputs.

**Key reference.** Farid — *"Exposing Digital Forgeries from JPEG Ghosts"* (IEEE TIFS 2009).

### 7. Noise residual autocorrelation (`noise_score`)

**What it measures.** Whether the high-pass residual of the image is spatially uncorrelated, as real sensor noise tends to be.

**The physics.** A real camera image, after subtracting a smoothed version of itself, leaves a residual dominated by sensor noise (read noise, shot noise, thermal noise) plus some texture and JPEG quantization noise. The sensor-noise component is approximately spatially white — adjacent pixels have near-zero correlation — because the noise sources are independent at each photosite and the ISP doesn't perfectly remove that.

Synthetic images don't have sensor noise. Their high-pass residuals reflect the upsampling and decoding statistics of the generator, which typically leave *correlated* residuals — adjacent pixels in the residual are not independent because they came from overlapping receptive fields in the same convolutional layer.

**Implementation.**
1. Compute a 3×3 mean-filtered version of the grayscale image via numpy slicing.
2. Subtract to get the high-pass residual `res`.
3. Compute three lag-1 autocorrelations: horizontal `r(0,1)`, vertical `r(1,0)`, diagonal `r(1,1)`, each as `mean(res * shifted_res) / var(res)`.
4. Score = `clip(3 * mean(|r|), 0, 1)`. Spatially uncorrelated noise → score near 0 (real). Correlated residual → score near 1 (synthetic).

**Limitation.** Heavy denoising in the camera ISP (newer phones do a lot of this) can also produce correlated residuals, which leads to false positives. Conversely, generators that explicitly add Gaussian noise to their output (some adversarial defenses do this) can fool the test.

**Key reference.** Lukas, Fridrich, Goljan — *"Digital Camera Identification from Sensor Pattern Noise"* (IEEE TIFS 2006). Same family of techniques as PRNU sensor identification.

---

## AI watermark / provenance scanner (`watermark.py`)

A separate module that scans the image *file* (not just the pixels) for traces that AI generators leave behind. Five checks:

1. **C2PA / Content Credentials.** A growing standard — DALL-E 3, Adobe Firefly, Microsoft Designer, OpenAI's GPT-Image and others embed a JUMBF-wrapped manifest in the file describing how the image was made. Detection: search the raw bytes for the `jumbf` magic and `c2pa` / `contentcredentials` substrings.
2. **PNG text chunks.** Stable Diffusion frontends (Automatic1111, ComfyUI, InvokeAI) write the prompt, negative prompt, sampler, model hash, etc. into PNG `iTXt`/`tEXt` chunks named `parameters`, `prompt`, `workflow`. Detection: PIL exposes these via `img.info`.
3. **EXIF / XMP fields.** Generator names sometimes appear in `Software`, `Description`, `Comment`, or vendor-specific XMP namespaces.
4. **Raw-byte keyword sweep.** Any AI generator name, watermark scheme name, or distinctive parameter string (`Steps:`, `CFG scale:`, `Model hash:`, etc.) anywhere in the first 4 MB of the file.
5. **Mid-band spectral anomaly check.** Public spread-spectrum watermarks (the SD `invisible-watermark` library, a number of academic schemes) embed bits by perturbing mid-frequency DWT coefficients in a structured way. Without the secret key we can't read the bits, but we can flag anomalously peaked energy concentration in the mid-band annulus by computing the kurtosis of the DCT-magnitude distribution there.

The scanner returns a score in `[0, 1]` plus a structured breakdown (which markers were found, with snippets). When the watermark scanner says *yes* and the pixel signals say *no*, the image was probably saved through a pipeline that preserved metadata but happened to be a clever generator (high-quality diffusion). When pixel signals say *yes* and the watermark scanner says *no*, the image is probably synthetic but had its metadata stripped. When both agree → high confidence in the verdict.

**About truly secret watermarks.** Schemes like Google's SynthID embed a watermark in pixel space using a private key — they are *designed* to be undetectable without the verification key. No classical method, including this one, can read SynthID. We only flag SynthID *if its name appears in metadata*, which it sometimes does for compliance reasons.

---

## How the signals combine

Two combiners exist:

**1. Default weighted sum** (used by `cli.py` when no fitted model is present):

```python
combined = (0.20 * fft + 0.10 * ela + 0.20 * benford
            + 0.15 * cfa + 0.10 * wavelet + 0.10 * jpeg_ghost + 0.15 * noise)
verdict  = "synthetic" if combined > 0.5 else "real"
```

The weights are an informed prior: signals more robust to JPEG re-encoding (FFT, Benford) get the highest weight; signals known to be noisy on heavily-compressed inputs (CFA, ELA) get less.

**2. Logistic regression** (used by `tune.py` to fit weights from labeled data):

```
p(synthetic) = sigmoid(((x - μ) / σ) · w + b)
```

where `x` is the 7-dim score vector, `μ`/`σ` are the per-feature train-set mean and standard deviation, and `w`, `b` are fitted by gradient descent on binary cross-entropy + L2:

```
ℓ(w, b) = mean(- y log p - (1 - y) log(1 - p)) + λ ||w||²
∇w     = X.T @ (p - y) / n + 2λw
∇b     = mean(p - y)
```

Six lines of pure numpy, no sklearn, no torch. The output is interpretable: each learned coefficient tells you whether that signal pushed the prediction toward "synthetic" (positive) or "real" (negative), and how strongly.

**For video**, the per-frame scoring runs on every Nth frame (default 15). The clip is flagged synthetic if either the *mean* combined score is high OR the *temporal standard deviation* is high. The temporal-variance check catches face-swaps that drift between frames — a real clip should have stable per-frame statistics, a stitched fake usually doesn't.

---

## Project layout

```
deepfake-detector/
├── detector.py        # the seven pixel signals + default weighted-sum combiner
├── watermark.py       # AI watermark / metadata / provenance scanner
├── cli.py             # `image`, `video`, `watermark` subcommands
├── tune.py            # logistic-regression combiner training
├── requirements.txt   # numpy, pillow, opencv-python
├── samples/
│   ├── real/          # drop natural camera photos here
│   └── fake/          # drop AI-generated images here
└── README.md          # this file
```

## Usage

Full analysis on one image (all 7 signals + watermark scan):

```bash
python3 cli.py image path/to/image.jpg
```

Watermark / metadata scan only (fast — no pixel statistics):

```bash
python3 cli.py watermark path/to/image.png
```

Score a video, sampling every Nth frame:

```bash
python3 cli.py video path/to/clip.mp4 --every 10
```

Fit logistic-regression weights against your labeled set:

```bash
python3 tune.py                                  # default 70/30 split, seed 0
python3 tune.py --test-frac 0.2 --seed 42 --epochs 6000 --l2 0.1
```

`tune.py` does a stratified train/test split, fits a 7-feature logistic regression by gradient descent on the training split only, then reports both training and held-out test accuracy and prints the learned per-signal coefficients (sorted by magnitude, so you can read off which signals were most discriminative on your data). With fewer than 30 total samples it prints a warning that the weights will overfit.

---

## Honest limitations

This is a baseline detector. Specific failure modes you should know about:

- **Modern diffusion models** (anything trained with Karras-style anti-aliased upsamplers, JPEG augmentation, or explicit spectral regularization — i.e. most 2024+ models) deliberately avoid leaving the spectral fingerprints `fft_score` looks for.
- **JPEG round-trips destroy signal.** If an AI image was screenshotted, re-saved in Photoshop, or sent through messaging apps that re-encode, all three signals get noisier.
- **ELA is actively misleading on already-compressed inputs.** That's why it has the lowest weight; even so, it can pull a real photo over the threshold if the photo went through several JPEG generations.
- **Adversarial post-processing** — adding small Gaussian noise, applying a slight low-pass filter, or running through a JPEG cycle at quality ~75 — can drop all three scores. There's no defence against this in a detector that doesn't itself learn.
- **Benford fails on synthetic-looking real images** — close-up textures with little variation (a wall, the sky) have too few AC coefficients with magnitude ≥ 1 to give a stable first-digit histogram. The score defaults to 0.5 in that degenerate case, which is information-free.
- **The thresholds are eyeballed.** Until you've run `tune.py` on a labeled set of ≥ 20 + 20 images, the verdict is more "directional" than "calibrated."

What this tool *is* good for: building intuition for how generators differ from cameras at the signal level, sanity-checking suspicious images before deeper analysis, and serving as a starting point for a more rigorous classical pipeline (PRNU, demosaicing-trace analysis, chromatic-aberration consistency, etc.).

What it *isn't*: a court-admissible authenticity verdict.

---

## Further reading

- Burton, Moorhead — *"Color and spatial structure in natural scenes"* (1987) — natural-image power-law spectra.
- Field — *"Relations between the statistics of natural images and the response properties of cortical cells"* (1987) — same.
- Krawetz — *"A Picture's Worth"* (2007) — original ELA.
- Pérez-González et al. — *"Benford's Law in Image Processing"* (2007).
- Zhang, Karaman, Chang — *"Detecting and Simulating Artifacts in GAN Fake Images"* (2019).
- Frank et al. — *"Leveraging Frequency Analysis for Deep Fake Image Recognition"* (ICML 2020).
- Wang et al. — *"CNN-generated images are surprisingly easy to spot...for now"* (CVPR 2020).
