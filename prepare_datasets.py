"""Decode the downloaded real-world datasets into labelled folders for external
validation.

  CIFAKE  (image)  parquet  -> datasets/cifake/real/*.png  (CIFAR-10 photos)
                              datasets/cifake/fake/*.png  (Stable-Diffusion)
  HC3     (text)   jsonl    -> datasets/hc3/human/*.txt
                              datasets/hc3/ai/*.txt        (ChatGPT answers)

CIFAKE label convention (from the dataset card): 0 = FAKE, 1 = REAL.
These folders feed `evaluate.py --real ... --fake ...` (image) and
`evaluate_text.py --real ... --fake ...` (text) for genuinely-labelled external
testing of the self-supervised detectors.

    python3 prepare_datasets.py --n-image 300 --n-text 300
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

DS = Path("datasets")


def prepare_cifake(n_per_class: int) -> tuple[int, int]:
    import pandas as pd
    from PIL import Image

    df = pd.read_parquet(DS / "cifake" / "test.parquet")
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
        img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        img.save(targets[lab] / f"{counts[lab]:04d}.png")
        counts[lab] += 1
    return counts[1], counts[0]


def prepare_hc3(n_per_class: int) -> tuple[int, int]:
    human_dir = DS / "hc3" / "human"
    ai_dir = DS / "hc3" / "ai"
    human_dir.mkdir(parents=True, exist_ok=True)
    ai_dir.mkdir(parents=True, exist_ok=True)
    nh = na = 0
    with open(DS / "hc3" / "all.jsonl") as f:
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-image", type=int, default=300)
    ap.add_argument("--n-text", type=int, default=300)
    args = ap.parse_args()

    r, f = prepare_cifake(args.n_image)
    print(f"CIFAKE  -> {r} real, {f} fake images in {DS/'cifake'}/")
    h, a = prepare_hc3(args.n_text)
    print(f"HC3     -> {h} human, {a} ai texts in {DS/'hc3'}/")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
