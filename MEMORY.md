# Project memory — Label-Free Multimodal Synthetic-Media Detector

## Folder map (from `find . -maxdepth 2`, 2026-06-14)

```
detector.py        image: 7 forensic signals + fixed-weight combiner
patterns.py        image: 3 pattern checks (spectral-peak, block-heterogeneity, LBP)
deep_features.py   image: frozen ResNet-18 dual-stream (spatial + spectral) embedding
fusion.py          image: hand-crafted/deep/fusion learned classifier (+ calibration)
selfsup.py         image: corpus -> pseudo-real/pseudo-fake (5 artifact transforms)
train_selfsup.py   image: trains models/fusion.joblib
evaluate.py        image: leave-source-out CV + ablation + external (--real/--fake) -> paper/results*
watermark.py       provenance: C2PA / PNG-EXIF AI tags / keyword + spectral scan

audio_detect.py    audio: 8 voice-forensic signals + STFT/ISTFT/wav IO   [NEW]
audio_selfsup.py   audio: procedural speech corpus + 5 synthesis pseudo-fakes  [NEW]
train_audio.py     audio: trains models/audio.joblib  [NEW]
evaluate_audio.py  audio: leave-source-out CV ablation -> paper/results_audio*  [NEW]

text_detect.py     text: 8 LLM forensic signals (pure stdlib)  [NEW]
text_selfsup.py    text: human seed corpus + 5 LLM-style pseudo-fakes  [NEW]
train_text.py      text: trains models/text.joblib (OPTIONAL ablation; not shipped)  [NEW]
evaluate_text.py   text: CV ablation + external (--real/--fake HC3) -> paper/results_text*  [NEW]

audio_io.py        media: wav (stdlib) + ffmpeg transcode + video audio extraction  [NEW]
pipeline.py        unified inference: analyze_image/_path, analyze_audio(_path), analyze_text
cli.py             CLI: image | video | audio | text | watermark
app.py             Flask: /analyze (image/video/audio) + /analyze_text  [+audio,+text]
prepare_datasets.py decode datasets/cifake/*.parquet + datasets/hc3/all.jsonl -> labelled folders  [NEW]
paper/             paper.md (multimodal, authoritative) + paper_ieee.tex + results*.md + figures/
datasets/          git-ignored: hc3/ (text) + cifake/ (image) real external-validation data
models/            fusion.joblib (image, shipped) + audio.joblib (shipped)
```

## Dependency graph

```
cli.py / app.py  ->  pipeline.py  ->  {detector,patterns,fusion,deep_features}   (image)
                                  ->  audio_detect (+ audio_io, models/audio.joblib)  (audio)
                                  ->  text_detect  (classical default)                 (text)
cli.py video / app.py video      ->  audio_io.extract_audio_from_video -> pipeline.analyze_audio
evaluate*.py     ->  *_selfsup (corpus->pseudo) + *_detect/fusion (features) + sklearn GroupKFold
```

## Critical info / commands (system `python3`, NOT .venv)

- Env: numpy, scikit-learn, matplotlib, torch (CPU), scipy, joblib, **pyarrow**, PIL, cv2 all present; **ffmpeg** present (needed for non-wav audio + video audio extraction).
- Image eval (+external): `python3 evaluate.py --folds 5 --real datasets/cifake/real --fake datasets/cifake/fake`
- Audio: `python3 train_audio.py` then `python3 evaluate_audio.py` (feature extraction ~25s for 320 samples; train ~7min for 600 — f0 tracking is the bottleneck).
- Text: `python3 evaluate_text.py --real datasets/hc3/human --fake datasets/hc3/ai`
- Datasets: `python3 prepare_datasets.py` (decodes the downloaded HC3 + CIFAKE). Seeds fixed at 0 everywhere.

## Decisions

- **Text default = classical combiner, not the learned model.** Why: on real HC3 the training-free combiner reaches 0.960 AUC vs 0.825 for the learned model, and the learned model over-flags real human text as AI (overfit to injection artifacts). `train_text.py`/`models/text.joblib` kept as an optional ablation, NOT shipped. Rests on: external HC3 numbers measured this session.
- **Audio default = learned model** (`models/audio.joblib`): classical audio combiner is ~chance (0.455 AUC, below 0.5) on the corpus; learned reaches 0.957. Mirrors the image pipeline's shipped fusion model.
- **Audio corpus is procedural** (no open labelled synthetic-speech set shipped). 0.957 AUC is in-distribution/optimistic — documented as such in paper §6 and README.
- **`.tex` only partially synced**: title/abstract/keywords updated to multimodal; body prose still image-only; NOT compile-verified (no pdflatex). `paper/paper.md` is the authoritative multimodal write-up.

## Tried-and-failed

- `import pyarrow` initially missing -> `pip install pyarrow` (pandas 3.0.3 needs it for parquet).
- Real audio deepfake datasets (ASVspoof2019 LA, WaveFake, In-the-Wild) are gated/large/not openly on HF -> kept procedural corpus; ASVspoof2019 LA is the documented next benchmark.
- Shipping the learned text model made the demo worse (human text flagged AI) -> removed `models/text.joblib`.
