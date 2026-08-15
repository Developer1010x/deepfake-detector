# Deepfake Detector

Label-free, CPU-only detection of synthetic media across four modalities: images,
audio (synthetic or cloned voice), text (LLM-generated) and video.

One method is instantiated per modality:

    unlabelled corpus -> benign pseudo-real + artifact-injected pseudo-fake
                      -> interpretable forensic bank
                      -> calibrated classifier, leave-source-out cross-validation

No labelled deepfakes are needed. The system manufactures its own supervision from
any unlabelled corpus, so it targets transferable artifact families rather than
memorising one generator's fingerprint. Every score maps to a named, physically
motivated phenomenon, so each verdict can be explained rather than asserted.

## Install

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    # ffmpeg is needed only for non-WAV audio and for video audio tracks:
    #   sudo apt-get install ffmpeg

## Use

    python3 cli.py image  samples/fake/alpha-dog.jpg
    python3 cli.py audio  samples/demo/pseudo-fake-voice.wav
    python3 cli.py text   samples/demo/pseudo-ai.txt
    python3 cli.py video  clip.mp4 --every 12
    python3 cli.py watermark samples/fake/alpha-dog.jpg

    python3 cli.py image samples/fake/cat-astronaut.jpeg --explain /tmp/maps
    python3 batch.py DIR --csv ranked.csv        # rank a folder of unlabelled media
    python3 app.py                               # web UI + JSON API on :5000

## How it works

| Modality | Signals | Self-supervision | Eval |
|---|---|---|---|
| Image | `detector.py` (7 statistics), `patterns.py`, `deep_features.py`, fused by `fusion.py` | `selfsup.py`, 5 artifact families | `evaluate.py` |
| Audio | `audio_detect.py`, 8 voice-forensic signals | `audio_selfsup.py`: griffin-lim phase, mel over-smoothing, band-limit, harmonic comb, noise gate | `evaluate_audio.py` |
| Text | `text_detect.py`, 8 LLM signals | `text_selfsup.py`: flattened burstiness, injected repetition, lexical smoothing | `evaluate_text.py` |
| Video | per-frame image fusion + temporal instability + the separately scored audio track | inherited | inherited |

Each track ships a model card at `models/*.meta.json` recording the corpus it was
fitted on, its sample counts and the library versions that pickled it. `pipeline`
re-checks those versions on load and reports a mismatch rather than silently
degrading. A present-but-unloadable artifact is reported as an error with the real
exception, never as "no trained model".

**Abstention is deliberate.** Near p = 0.5 the detector returns
`inconclusive - insufficient evidence` instead of committing. A calibrated
classifier at p = 0.49 is saying it cannot separate the classes for this input,
and saying so is the difference between a forensic instrument and a coin flip.

## Results

Per-modality leave-source-out numbers, the figures behind them and the exact
commands are in `paper/`:

- `paper/results.md`, `paper/results.json` — image
- `paper/results_audio.md`, `paper/results_audio.json` — audio
- `paper/results_text.md`, `paper/results_text.json` — text
- `paper/figures/` — ROC, PR, calibration, confusion, per-artifact

Regenerate:

    python3 evaluate.py            # self-supervised metrics + figures
    python3 evaluate_audio.py
    python3 evaluate_text.py

## Sample media

`samples/fake/` and `samples/demo/` hold small stand-ins so every path can be run.
The audio, text and video demo assets are **generated, not captured**, and each is
labelled as a stand-in in `samples/demo/README.md`.

**No personal media is shipped in this repository.** The video demo is built from a
still you supply:

    python3 make_demo_media.py --still /path/to/photo.jpg

## Honest limitations

- **Modern diffusion models** (anti-aliased upsamplers, JPEG augmentation, spectral
  regularisation, most 2024+ models) deliberately suppress the spectral fingerprints
  the FFT signal looks for. `samples/fake/portrait.jpeg` is exactly this case and the
  shipped model scores it 0.316, i.e. wrong. It is kept in the repository rather than
  quietly removed.
- **CIFAKE is close to chance: 0.553 AUC, measured.** At 32x32 the radial spectrum has
  about 16 usable bins and the tiled fields collapse to a single block. The command
  that produces the number is documented precisely because the number is unflattering.
- **JPEG round-trips destroy signal.** Screenshots, re-saves and messaging-app
  re-encoding make every signal noisier.
- **Adversarial post-processing** (slight noise, low-pass, a quality-75 JPEG cycle) can
  drop the scores. A non-learning detector has no defence.
- **The audio corpus is procedural for both classes.** `models/audio.joblib` has never
  heard a human voice, and it under-detects `noise_gate`-only fakes (P ~ 0.29, measured).
- **Thresholds are calibrated, not absolute.** Validate on a labelled benchmark
  (`evaluate.py --real DIR --fake DIR`) before trusting this anywhere that matters.

What this is: a reproducible, explainable baseline for understanding how generators
differ from cameras, cloned voices and human writing at the signal level. What it is
not: a production authenticity guarantee, or a court-admissible verdict.

## Tests

    python3 -m pytest -q

## License

MIT. See `LICENSE`.
