"""Fetch a reproducible corpus of real photographs for label-free training.

The self-supervised recipe needs *one* thing from the outside world: a set of
ordinary camera photographs to treat as the real manifold. It does not need
labels, and it must not need deepfakes. This script fetches such a corpus from
Lorem Picsum (which serves Unsplash photographs) by stable numeric id, so the
same command reproduces the same corpus on any machine.

    python3 fetch_corpus.py --n 160 --width 1024

Writes ``corpus/<id>.jpg`` plus ``corpus/manifest.json`` recording every photo's
id, photographer and Unsplash page — the provenance the model card points at.
``corpus/`` is git-ignored; the trained artifact is what ships.

Why real photographs and not the four images in ``samples/``: the shipped model
learns "what a camera sensor's statistics look like" from this corpus, and four
images cannot span that. See ``models/fusion.meta.json`` for what the released
artifact was actually fitted on.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPUS_DIR = HERE / "corpus"

LIST_URL = "https://picsum.photos/v2/list?page={page}&limit={limit}"
IMAGE_URL = "https://picsum.photos/id/{id}/{w}/{h}"
_UA = {"User-Agent": "deepfake-detector/1.0 (+corpus fetch)"}
TIMEOUT = 30


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return r.read()


def list_photos(n: int) -> list[dict]:
    """Metadata for the first ``n`` photos, in stable id order."""
    out: list[dict] = []
    page = 1
    while len(out) < n:
        try:
            batch = json.loads(_get(LIST_URL.format(page=page, limit=100)))
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            print(f"ERROR: could not list photos ({exc})", file=sys.stderr)
            break
        if not batch:
            break
        out.extend(batch)
        page += 1
    return out[:n]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=160, help="number of photographs")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--out", type=Path, default=CORPUS_DIR)
    args = ap.parse_args()
    # Line-buffer stdout: these jobs run for minutes and their progress is
    # useless if it sits in a 4 KB block buffer until the process exits.
    sys.stdout.reconfigure(line_buffering=True)

    args.out.mkdir(parents=True, exist_ok=True)
    photos = list_photos(args.n)
    if not photos:
        print("ERROR: no photo metadata retrieved; is the network reachable?",
              file=sys.stderr)
        return 1

    manifest, ok, skipped = [], 0, 0
    for meta in photos:
        pid = meta["id"]
        dest = args.out / f"{pid}.jpg"
        entry = {"id": pid, "author": meta.get("author"),
                 "unsplash_url": meta.get("url"), "file": dest.name}
        if dest.exists() and dest.stat().st_size > 0:
            manifest.append(entry)
            skipped += 1
            continue
        try:
            dest.write_bytes(_get(IMAGE_URL.format(id=pid, w=args.width, h=args.height)))
        except (urllib.error.URLError, OSError) as exc:
            print(f"  skip id={pid}: {exc}", file=sys.stderr)
            continue
        manifest.append(entry)
        ok += 1
        if ok % 20 == 0:
            print(f"  fetched {ok} ...")

    (args.out / "manifest.json").write_text(json.dumps(
        {"source": "Lorem Picsum (Unsplash photographs)",
         "list_url": LIST_URL, "image_url": IMAGE_URL,
         "size": [args.width, args.height],
         "n": len(manifest), "photos": manifest}, indent=2))
    print(f"corpus: {len(manifest)} photographs in {args.out}/ "
          f"({ok} downloaded, {skipped} already present)")
    print(f"manifest -> {args.out / 'manifest.json'}")
    return 0 if manifest else 1


if __name__ == "__main__":
    sys.exit(main())
