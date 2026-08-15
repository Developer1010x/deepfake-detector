"""Generate the audio and video demo clips the CLI and web UI need.

Two of the four advertised modalities had nothing to run against: ``.wav`` and
``.mp4`` are git-ignored and no clip shipped, so ``cli.py audio`` and
``cli.py video`` could not be demonstrated at all. This script builds four small
clips from material already in the repository, using *the project's own*
artifact-injection transforms:

    samples/demo/pseudo-human.txt        human corpus prose, benign edit
    samples/demo/pseudo-ai.txt           same prose + LLM-style regularities
    samples/demo/pseudo-real-voice.wav   procedural formant speech, benign edit
    samples/demo/pseudo-fake-voice.wav   same utterance + vocoder artifacts
    samples/demo/pseudo-real-clip.mp4    Ken-Burns pan over samples/real/, benign
    samples/demo/pseudo-fake-clip.mp4    same pan + per-frame generation artifacts

These are honest *stand-ins*, not captured media, and they are labelled as such
everywhere they appear. The voice pair comes from ``audio_selfsup`` (the same
generator the shipped ``audio.joblib`` was trained on) and the video pair from
``selfsup`` (the image transforms). What they demonstrate is that each track
runs end to end and that the signals move in the expected direction — not
detection accuracy on real deepfakes, which is what ``evaluate.py`` is for.

    python3 make_demo_media.py
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import audio_selfsup as asup
import selfsup

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "samples" / "demo"
SOURCE_IMAGE = HERE / "samples" / "real" / "iphone-photo.jpg"

FPS = 12
N_FRAMES = 48
FRAME_W, FRAME_H = 640, 480


# --------------------------------------------------------------------------- #
# audio
# --------------------------------------------------------------------------- #


def write_wav(path: Path, x: np.ndarray, sr: int) -> None:
    """16-bit mono PCM, readable by the stdlib ``wave`` reader in audio_detect."""
    pcm = np.clip(x, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


#: The demo fake uses a *fixed, named* artifact chain rather than a random draw
#: from `audio_selfsup._FAKE_TRANSFORMS`: magnitude-only re-synthesis followed by
#: a vocoder-style band limit is the pair an actual neural TTS pipeline leaves
#: behind. A random draw can land on `noise_gate` alone, which is both
#: unrepresentative of TTS and (honestly) an artifact this bank barely detects
#: - see samples/demo/README.md.
VOICE_ARTIFACTS = ("griffin_lim_phase", "band_limit")


def build_voices(seconds: float, seed: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """One utterance, twice: benignly edited, and with synthesis artifacts."""
    rng = np.random.default_rng(seed)
    want = int(seconds * asup.SR)
    parts = []
    total = 0
    while total < want:
        u = asup.synth_utterance(rng)
        parts.append(u)
        total += len(u)
    base = np.concatenate(parts)[:want].astype(np.float32)

    real = asup.benign_augment(base, np.random.default_rng(seed + 1))

    fake_rng = np.random.default_rng(seed + 2)
    fake = base
    for name in VOICE_ARTIFACTS:
        fake = getattr(asup, name)(fake, fake_rng)
    fake = asup.benign_augment(fake, np.random.default_rng(seed + 3))
    return real, fake, list(VOICE_ARTIFACTS)


# --------------------------------------------------------------------------- #
# text
# --------------------------------------------------------------------------- #


#: Pinned for the same reason as VOICE_ARTIFACTS: boilerplate connectives
#: ("Notably, ... Additionally, ... Overall, ...") plus lexical smoothing are the
#: two most characteristic LLM regularities, and they produce prose that reads
#: like an assistant. A random draw can land on `inject_repetition`, whose output
#: ("the wavefront that the wavefront that") is detectable but not
#: representative of anything a modern model would emit.
TEXT_ARTIFACTS = ("inject_boilerplate", "lexical_smoothing")


def build_texts(seed: int) -> tuple[str, str, list[str]]:
    """One passage, twice: benignly edited, and with LLM-style regularities.

    Same construction as the audio and video pairs — the source prose is the
    human corpus in ``text_selfsup``, and the "AI" view is that prose put through
    the named injection transforms, not text from any language model.
    """
    import text_selfsup as tsup

    sources = tsup.build_sources(window=4)
    # Longest window: the eight text signals saturate on short input, so the demo
    # passage must clear the 25-word API floor comfortably.
    _, base = max(sources, key=lambda kv: len(kv[1].split()))
    human = tsup.benign_edit(base, np.random.default_rng(seed))

    ai_rng = np.random.default_rng(seed + 1)
    ai = base
    for name in TEXT_ARTIFACTS:
        ai = getattr(tsup, name)(ai, ai_rng)
    ai = tsup.benign_edit(ai, np.random.default_rng(seed + 2))
    return human, ai, list(TEXT_ARTIFACTS)


# --------------------------------------------------------------------------- #
# video
# --------------------------------------------------------------------------- #


def ken_burns(arr: np.ndarray, n_frames: int, w: int, h: int) -> list[np.ndarray]:
    """Slow zoom-and-pan crop sequence over one still, so frames differ."""
    H, W = arr.shape[:2]
    frames = []
    for i in range(n_frames):
        t = i / max(1, n_frames - 1)
        zoom = 1.0 - 0.18 * t                     # crop tightens over time
        cw, ch = int(W * zoom), int(H * zoom)
        x0 = int((W - cw) * (0.15 + 0.7 * t))
        y0 = int((H - ch) * (0.65 - 0.4 * t))
        crop = arr[y0:y0 + ch, x0:x0 + cw]
        frames.append(cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA))
    return frames


def write_mp4(path: Path, frames: list[np.ndarray], fps: int) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))
    if not vw.isOpened():
        raise RuntimeError(f"OpenCV could not open a writer for {path}")
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()


def mux(video: Path, audio: Path, out: Path) -> bool:
    """Mux an audio track in with ffmpeg (H.264 so browsers can preview it)."""
    if shutil.which("ffmpeg") is None:
        return False
    proc = subprocess.run(
        # crf 18 is visually near-lossless: at the default 23 the H.264 stage
        # itself dominates the forensic signals and both clips read as synthetic,
        # which would tell you about the encoder, not about the injected artifacts.
        ["ffmpeg", "-v", "error", "-y", "-i", str(video), "-i", str(audio),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "slow", "-crf", "18",
         "-c:a", "aac", "-b:a", "96k", "-shortest", str(out)],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        print(proc.stderr.decode()[:400], file=sys.stderr)
        return False
    return True


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--seconds", type=float, default=N_FRAMES / FPS)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    # Line-buffer stdout: these jobs run for minutes and their progress is
    # useless if it sits in a 4 KB block buffer until the process exits.
    sys.stdout.reconfigure(line_buffering=True)

    if not SOURCE_IMAGE.exists():
        print(f"ERROR: missing source still {SOURCE_IMAGE}", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    # ---- audio pair ----
    real_v, fake_v, ops = build_voices(args.seconds, args.seed)
    real_wav = args.out / "pseudo-real-voice.wav"
    fake_wav = args.out / "pseudo-fake-voice.wav"
    write_wav(real_wav, real_v, asup.SR)
    write_wav(fake_wav, fake_v, asup.SR)
    print(f"audio: {real_wav.name} (benign) / {fake_wav.name} "
          f"(artifacts: {', '.join(ops)})")

    # ---- text pair ----
    human_txt, ai_txt, text_ops = build_texts(args.seed)
    (args.out / "pseudo-human.txt").write_text(human_txt.strip() + "\n")
    (args.out / "pseudo-ai.txt").write_text(ai_txt.strip() + "\n")
    print(f"text: pseudo-human.txt ({len(human_txt.split())} words) / "
          f"pseudo-ai.txt ({len(ai_txt.split())} words, "
          f"artifacts: {', '.join(text_ops)})")

    # ---- video pair ----
    still = np.asarray(Image.open(SOURCE_IMAGE).convert("RGB"))
    base_frames = ken_burns(still, N_FRAMES, FRAME_W, FRAME_H)

    rng_r = np.random.default_rng(args.seed + 11)
    real_frames = [selfsup.benign_augment(f, rng_r) for f in base_frames]

    rng_f = np.random.default_rng(args.seed + 12)
    fake_frames, used = [], set()
    for f in base_frames:
        g, names = selfsup.make_pseudo_fake(f, rng_f)
        used.update(names)
        fake_frames.append(selfsup.benign_augment(g, rng_f))

    tmp_dir = args.out / ".tmp"
    tmp_dir.mkdir(exist_ok=True)
    ok = True
    for frames, wav, name in ((real_frames, real_wav, "pseudo-real-clip.mp4"),
                              (fake_frames, fake_wav, "pseudo-fake-clip.mp4")):
        silent = tmp_dir / ("silent-" + name)
        write_mp4(silent, frames, FPS)
        if mux(silent, wav, args.out / name):
            print(f"video: {name} ({len(frames)} frames @ {FPS} fps, with audio)")
        else:
            shutil.move(str(silent), str(args.out / name))
            print(f"video: {name} ({len(frames)} frames @ {FPS} fps, NO audio "
                  "- ffmpeg unavailable)", file=sys.stderr)
            ok = False
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"video artifacts injected: {', '.join(sorted(used))}")
    return 0 if ok else 0


if __name__ == "__main__":
    sys.exit(main())
