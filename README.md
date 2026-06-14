# Deepfake Detector — Label-Free Multimodal Synthetic-Media Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![CPU-only](https://img.shields.io/badge/runtime-CPU--only-green.svg)](#installation)

A research-grade, **CPU-only, label-free** detector that estimates whether an **image**, an **audio clip** (synthetic / cloned voice), a **text passage** (LLM-generated), or a **video** was produced by a generative AI model.

One methodology is instantiated per modality:

> **unlabelled corpus → benign pseudo-real + artifact-injected pseudo-fake → interpretable forensic bank → calibrated classifier, under leave-source-out cross-validation.**

No labelled deepfakes are required — the system manufactures its own supervision from any unlabelled corpus, so it targets transferable *artifact families* rather than memorising one generator's fingerprint. Every score maps to a named, physically-motivated phenomenon, so each verdict is explainable.

---

## Highlights

- **Four modalities** — image, audio, text, video — behind one CLI and one web API.
- **Label-free training** — synthesises pseudo-real / pseudo-fake pairs (cf. Self-Blended Images, CVPR 2022); no curated deepfake datasets needed.
- **Interpretable** — 13 hand-crafted forensic signals for images (+8 for audio, +8 for text), each a named signal-processing / statistical cue.
- **Hybrid for images** — frozen ImageNet backbone (dual-stream spatial + Fourier) fused with the forensic bank by a calibrated classifier.
- **Runs anywhere** — pure CPU, no GPU required.

### Headline results

Leave-source-out CV, with real-data external validation where an open set is available:

| Modality | Self-supervised AUC | External (real data) |
|---|---|---|
| Image (fusion) | **0.853** | 0.585 (CIFAKE, 32px — weak) |
| Audio (learned bank) | **0.957** (in-distribution) | — (no open set shipped) |
| Text (training-free bank) | 0.760 | **0.960** (HC3: real human vs. ChatGPT) |

Image fusion ablation (leave-source-out CV):

| Model | ROC-AUC | F1 | Brier |
|---|---|---|---|
| classical (fixed weights) | 0.589 | 0.621 | 0.246 |
| hand-crafted (learned) | 0.796 | 0.708 | 0.186 |
| deep (dual-stream) | 0.822 | 0.736 | 0.179 |
| **fusion** | **0.853** | **0.771** | **0.165** |

> **Honest scope.** The audio corpus is a *procedural stand-in* (no open, redistributable labelled synthetic-speech set ships here), so its 0.957 AUC is in-distribution and optimistic — read it as the separability of the injected artifacts, not a deployment number. The text track's strength is **ranking** (0.96 AUC on real HC3); single-sample verdicts at the 0.5 threshold are *indicative*. This is a strong, reproducible, explainable baseline — **not** a court-admissible authenticity verdict.

---

## Installation

```bash
git clone https://github.com/Developer1010x/deepfake-detector.git
cd deepfake-detector

pip install -r requirements.txt
# CPU-only torch is sufficient:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Optional system binary — decoding non-WAV audio and video audio-tracks needs ffmpeg:
#   sudo apt-get install ffmpeg
```

Python 3.9+ recommended.

---

## Quick start

```bash
# Train the image model (self-supervised, no labels needed)
python3 train_selfsup.py --patch 128      # -> models/fusion.joblib
python3 train_audio.py                     # -> models/audio.joblib

# Inference (CLI)
python3 cli.py image  path/to/image.jpg
python3 cli.py audio  path/to/clip.wav
python3 cli.py text   path/to/essay.txt        # or:  echo "..." | python3 cli.py text -
python3 cli.py video  path/to/clip.mp4 --every 10
python3 cli.py watermark path/to/image.png     # metadata / provenance scan only

# Web UI + multimodal API
python3 app.py                                 # serves on http://localhost:5000
```

### Web API

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Web UI (upload an image / audio / video) |
| `/analyze` | POST | Analyze an uploaded media file |
| `/analyze_text` | POST | Analyze a text passage |

---

## How it works

The same label-free recipe is instantiated across modalities — each track ships an interpretable forensic bank, a self-supervised pseudo-fake generator, and a leave-source-out evaluation harness.

| Modality | Detector / signals | Self-supervision | Eval | Inference |
|---|---|---|---|---|
| **Image** | `detector.py` (7 stats) + `patterns.py` + `deep_features.py` → `fusion.py` | `selfsup.py` (5 image artifacts) | `evaluate.py` | `cli.py image`, web upload |
| **Audio** | `audio_detect.py` (8 voice-forensic signals) | `audio_selfsup.py` (griffin-lim phase, mel over-smooth, band-limit, harmonic comb, noise gate) | `evaluate_audio.py` | `cli.py audio`, web upload, fused into video |
| **Text** | `text_detect.py` (8 LLM signals) | `text_selfsup.py` (flatten burstiness, inject repetition, lexical smoothing) | `evaluate_text.py` | `cli.py text`, `POST /analyze_text` |
| **Video** | per-frame image fusion + temporal instability + extracted audio track (`audio_io.py`) | — | — | `cli.py video`, web upload |

### The seven image signals

Each computes a score in `[0, 1]` (higher = more synthetic-looking), every one tied to a named physical phenomenon:

| Signal | What it catches |
|---|---|
| **FFT spectral** | upsampler periodicity — peaks/ringing that break the `1/f^α` natural-image power law |
| **Error Level Analysis** | uneven JPEG re-save error from splicing / mixed compression history |
| **Benford's law (DCT)** | leading-digit distribution of AC DCT coefficients drifting off Benford |
| **CFA / Bayer trace** | absence of the 2×2 demosaicing fingerprint left by camera sensors |
| **Wavelet kurtosis** | detail sub-bands that aren't as heavy-tailed as natural images |
| **JPEG ghost** | spatial inconsistency in the best-fit re-encoding quality |
| **Noise residual** | spatially *correlated* high-pass residual (real sensor noise is ~white) |

These 13 interpretable signals (7 pixel stats + 6 pattern checks in `patterns.py`) are first-class inputs to the fusion model and are reported alongside every verdict. A separate **watermark / provenance scanner** (`watermark.py`) inspects the file itself for C2PA Content Credentials, Stable-Diffusion PNG text chunks, EXIF/XMP generator names, and mid-band spectral watermark anomalies.

---

## Evaluation & external validation

```bash
# Self-supervised metrics + figures
python3 evaluate.py --folds 5

# Real datasets (the credibility check)
python3 prepare_datasets.py                                                   # decode HC3 (text) + CIFAKE (image)
python3 evaluate_text.py  --real datasets/hc3/human  --fake datasets/hc3/ai   # -> 0.960 AUC (real ChatGPT vs human)
python3 evaluate.py       --real datasets/cifake/real --fake datasets/cifake/fake  # -> 0.585 AUC (honest gap)
python3 evaluate_audio.py                                                      # -> 0.957 AUC in-distribution
```

Datasets live under `datasets/` and are git-ignored.

---

## Project layout

```
deepfake-detector/
├── cli.py                 # unified CLI: image / video / audio / text / watermark
├── app.py                 # Flask web UI + multimodal API
├── pipeline.py            # unified inference (fusion → classical fallback → watermark override)
│
├── detector.py            # 7 interpretable pixel signals + weighted-sum combiner
├── patterns.py            # additional pattern checks
├── deep_features.py       # frozen ResNet-18 dual-stream (spatial + spectral) embedding
├── fusion.py              # hybrid calibrated classifier (deep ⊕ hand-crafted)
├── selfsup.py             # label-free pseudo-real / pseudo-fake generator (image)
├── watermark.py           # AI watermark / metadata / provenance scanner
│
├── audio_detect.py        # 8 voice-forensic signals
├── audio_selfsup.py       # audio pseudo-fake generator
├── audio_io.py            # audio decode / video audio-track extraction
│
├── text_detect.py         # 8 LLM-text signals
├── text_selfsup.py        # text pseudo-fake generator
│
├── train_selfsup.py       # image self-supervised training  -> models/fusion.joblib
├── train_audio.py         # audio training                  -> models/audio.joblib
├── train_text.py          # text combiner training
├── evaluate.py            # image leave-source-out CV + ablation + figures
├── evaluate_audio.py      # audio evaluation
├── evaluate_text.py       # text evaluation
├── prepare_datasets.py    # decode HC3 / CIFAKE into labelled folders
├── tune.py                # logistic-regression combiner tuning
│
├── models/                # shipped trained artifacts (*.joblib)
├── samples/               # example real/ and fake/ inputs
├── templates/ · static/   # web UI
├── requirements.txt
└── Procfile               # gunicorn entry for deployment
```

---

## Honest limitations

- **Modern diffusion models** (Karras-style anti-aliased upsamplers, JPEG augmentation, spectral regularization — most 2024+ models) deliberately suppress the spectral fingerprints the FFT signal looks for.
- **JPEG round-trips destroy signal** — screenshots, re-saves, and messaging-app re-encoding make every signal noisier.
- **Adversarial post-processing** (slight noise, low-pass, a quality-75 JPEG cycle) can drop the scores; a non-learning detector has no defence against this.
- **Thresholds are calibrated, not absolute** — validate on a labelled benchmark (`evaluate.py --real DIR --fake DIR`) before trusting it in production.

What this tool *is*: a strong, reproducible, explainable baseline for understanding how generators differ from cameras / cloned voices / human writing at the signal level. What it *isn't*: a production authenticity guarantee.

---

## License

Released under the [MIT License](LICENSE).
