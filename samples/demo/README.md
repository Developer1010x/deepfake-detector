# Demo media — procedurally generated, not captured

Nothing in this folder is a real recording, and none of it is a real deepfake.
These six files exist so that `cli.py audio`, `cli.py video` and `cli.py text`
(and the corresponding web-UI paths) have something to run against; the
repository shipped no `.wav` or `.mp4` at all — both are git-ignored — which
meant two of the four advertised modalities could not be demonstrated.

Regenerate them at any time:

```bash
python3 make_demo_media.py
```

| File | What it is |
|---|---|
| `pseudo-human.txt` | A 4-sentence window of the human prose corpus in `text_selfsup`, after `benign_edit`. |
| `pseudo-ai.txt` | The **same passage** through `inject_boilerplate` then `lexical_smoothing`, then the same benign edit. Not output from any language model. |
| `pseudo-real-voice.wav` | 4 s of procedural formant speech from `audio_selfsup.synth_utterance`, then `benign_augment` (mild gain, a little noise, a resample round-trip) — the "camera-honest" edit. |
| `pseudo-fake-voice.wav` | The **same utterance**, put through `griffin_lim_phase` then `band_limit`, then the same benign edit. |
| `pseudo-real-clip.mp4` (not shipped, generate locally) | 48 frames (12 fps) of a Ken-Burns pan over a still you supply, each frame `benign_augment`-ed, muxed with `pseudo-real-voice.wav`. |
| `pseudo-fake-clip.mp4` (not shipped, generate locally) | The same pan with `selfsup.make_pseudo_fake` applied per frame (upsample fingerprint, self-blend, double JPEG, spectral injection, diffusion smoothing), muxed with `pseudo-fake-voice.wav`. |

## Why these artifact chains, and not random ones

`make_demo_media.TEXT_ARTIFACTS` pins the text fake to `inject_boilerplate →
lexical_smoothing`: boilerplate connectives ("Notably, ... Additionally, ...
Overall, ...") plus lexical flattening are the two most characteristic LLM
regularities, and together they produce prose that reads like an assistant. A
random draw can land on `inject_repetition`, whose output ("the wavefront that
the wavefront that") is trivially detectable but representative of nothing.

`make_demo_media.VOICE_ARTIFACTS` pins the audio fake to
`griffin_lim_phase → band_limit`: magnitude-only re-synthesis followed by a
vocoder-style high-frequency roll-off is the pair an actual neural TTS pipeline
leaves behind. A random draw from `audio_selfsup._FAKE_TRANSFORMS` can land on
`noise_gate` alone, and — measured, not assumed — the shipped `audio.joblib`
scores that at **P(synthetic) ≈ 0.29**, i.e. it misses it. That is a real gap in
the bank, worth stating rather than hiding behind a lucky seed.

## What these clips do and do not demonstrate

They demonstrate that each track runs end to end, that the signals move in the
expected direction, and what the per-frame timeline and the abstention band look
like on screen.

They do **not** demonstrate detection accuracy on real deepfakes. For that, see
`evaluate.py` (leave-source-out CV), `evaluate_text.py --real datasets/hc3/human
--fake datasets/hc3/ai` (real HC3 data) and the honest-limitations section of the
top-level README.
