"""Download and decode the real-world datasets used for external validation.

The self-supervised numbers in the README are only worth something if they are
checked against genuinely labelled data. That check needs two public datasets,
and this script now *fetches* them as well as decoding them — previously it read
files that nothing downloaded, so the entire "Evaluation & external validation"
section was unrunnable as written.

  CIFAKE  (image)  parquet  -> datasets/cifake/real/*.png  (CIFAR-10 photos)
                              datasets/cifake/fake/*.png  (Stable-Diffusion)
  HC3     (text)   jsonl    -> datasets/hc3/human/*.txt
                              datasets/hc3/ai/*.txt        (ChatGPT answers)

CIFAKE label convention (from the dataset card): 0 = FAKE, 1 = REAL.
These folders feed `evaluate.py --real ... --fake ...` (image) and
`evaluate_text.py --real ... --fake ...` (text).

    python3 prepare_datasets.py                       # download + decode
    python3 prepare_datasets.py --n-image 300 --n-text 300
    python3 prepare_datasets.py --no-download         # decode already-present files
    python3 prepare_datasets.py --hc3-file all.jsonl  # the full 73 MB HC3 split

By default HC3 comes from the small ``open_qa`` split (~3 MB) rather than the
73 MB ``all.jsonl``: it is the same human-vs-ChatGPT data and 300 samples per
class come out of it comfortably.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

DS = Path(__file__).resolve().parent / "datasets"

# Hugging Face resolve URLs. Both are public; no token required.
CIFAKE_REPO = "dragonintelligence/CIFAKE-image-dataset"
CIFAKE_FILE = "data/test-00000-of-00001.parquet"          # ~8 MB
HC3_REPO = "Hello-SimpleAI/HC3"
HC3_DEFAULT_FILE = "open_qa.jsonl"                        # ~3 MB
HF_URL = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"

_UA = {"User-Agent": "deepfake-detector/1.0 (+dataset fetch)"}
TIMEOUT = 60


def download(url: str, dest: Path) -> bool:
    """Stream ``url`` to ``dest``; returns False (with a message) on failure."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  have {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  fetching {url}")
    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r, tmp.open("wb") as out:
            while chunk := r.read(1 << 20):
                out.write(chunk)
    except (urllib.error.URLError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        print(f"  ERROR: download failed: {exc}", file=sys.stderr)
        return False
    tmp.replace(dest)
    print(f"  saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return True


def _image_bytes(cell) -> bytes:
    """Raw bytes from a Hugging Face image column (a dict with 'bytes', or raw)."""
    return cell["bytes"] if isinstance(cell, dict) else cell


def prepare_cifake(n_per_class: int, parquet: Path) -> tuple[int, int]:
    import pandas as pd
    from PIL import Image

    df = pd.read_parquet(parquet)
    real_dir = DS / "cifake" / "real"
    fake_dir = DS / "cifake" / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    counts = {1: 0, 0: 0}                       # 1=REAL, 0=FAKE
    targets = {1: real_dir, 0: fake_dir}
    for _, row in df.iterrows():
        lab = int(row["label"])
        if counts.get(lab, n_per_class) >= n_per_class:
            if all(c >= n_per_class for c in counts.values()):
                break
            continue
        img = Image.open(io.BytesIO(_image_bytes(row["image"]))).convert("RGB")
        img.save(targets[lab] / f"{counts[lab]:04d}.png")
        counts[lab] += 1
    return counts[1], counts[0]


def prepare_hc3(n_per_class: int, jsonl: Path) -> tuple[int, int]:
    human_dir = DS / "hc3" / "human"
    ai_dir = DS / "hc3" / "ai"
    human_dir.mkdir(parents=True, exist_ok=True)
    ai_dir.mkdir(parents=True, exist_ok=True)
    nh = na = 0
    with open(jsonl) as f:
        for line in f:
            if nh >= n_per_class and na >= n_per_class:
                break
            rec = json.loads(line)
            for ans in rec.get("human_answers", []):
                if nh < n_per_class and len(ans.split()) >= 25:
                    (human_dir / f"{nh:04d}.txt").write_text(ans.strip())
                    nh += 1
            for ans in rec.get("chatgpt_answers", []):
                if na < n_per_class and len(ans.split()) >= 25:
                    (ai_dir / f"{na:04d}.txt").write_text(ans.strip())
                    na += 1
    return nh, na


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-image", type=int, default=300)
    ap.add_argument("--n-text", type=int, default=300)
    ap.add_argument("--hc3-file", default=HC3_DEFAULT_FILE,
                    help=f"HC3 split to use (default: {HC3_DEFAULT_FILE}; "
                         "'all.jsonl' is the full 73 MB set)")
    ap.add_argument("--no-download", action="store_true",
                    help="decode files already under datasets/ instead of fetching")
    ap.add_argument("--only", choices=["image", "text"], default=None)
    args = ap.parse_args()
    # Line-buffer stdout: these jobs run for minutes and their progress is
    # useless if it sits in a 4 KB block buffer until the process exits.
    sys.stdout.reconfigure(line_buffering=True)

    parquet = DS / "cifake" / "test.parquet"
    jsonl = DS / "hc3" / args.hc3_file
    rc = 0

    if args.only != "text":
        print("CIFAKE (image):")
        ok = parquet.exists() if args.no_download else download(
            HF_URL.format(repo=CIFAKE_REPO, path=CIFAKE_FILE), parquet)
        if ok:
            r, f = prepare_cifake(args.n_image, parquet)
            print(f"  -> {r} real, {f} fake images in {DS / 'cifake'}/")
        else:
            print(f"  SKIPPED: {parquet} not available", file=sys.stderr)
            rc = 1

    if args.only != "image":
        print("HC3 (text):")
        ok = jsonl.exists() if args.no_download else download(
            HF_URL.format(repo=HC3_REPO, path=args.hc3_file), jsonl)
        if ok:
            h, a = prepare_hc3(args.n_text, jsonl)
            print(f"  -> {h} human, {a} ai texts in {DS / 'hc3'}/")
        else:
            print(f"  SKIPPED: {jsonl} not available", file=sys.stderr)
            rc = 1

    return rc


if __name__ == "__main__":
    sys.exit(main())
