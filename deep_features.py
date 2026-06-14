"""Deep, self-supervised feature extraction for hybrid deepfake detection.

This module provides the *learned* half of the hybrid detector. It pairs a
frozen ImageNet-pretrained convolutional backbone (no deepfake labels are ever
used to train it) with a dual-stream design:

  * spatial stream  - the RGB image, capturing semantic / texture artifacts.
  * spectral stream - the log-magnitude 2-D Fourier spectrum, rendered as a
                      3-channel image, capturing the periodic up-sampling and
                      GAN/diffusion "fingerprint" artifacts that live in
                      frequency space (Wang et al. 2020; Frank et al. 2020).

Both streams pass through the same backbone; their global-average-pooled
embeddings are concatenated. The result is a fixed-length descriptor that is
combined with the hand-crafted forensic signals in ``fusion.py``.

Design notes for reproducibility:
  * The backbone is *frozen* (eval mode, ``requires_grad=False``). We use it
    purely as a fixed feature map, so no GPU and no labelled training data are
    required - consistent with the label-free, self-supervised setting.
  * torch / torchvision are imported lazily. If they are unavailable the rest
    of the project (classical signals, CLI, web app) still runs; callers should
    check :func:`available` first.
  * Everything runs on CPU and is deterministic given an input image.
"""

from __future__ import annotations

import functools
import threading

import numpy as np
from PIL import Image

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #

_INPUT_SIZE = 224
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Backbone embedding width (ResNet18 global-pool = 512). Two streams -> 1024.
BACKBONE_DIM = 512
DEEP_DIM = 2 * BACKBONE_DIM  # spatial + spectral

_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# availability / lazy model construction
# --------------------------------------------------------------------------- #


@functools.lru_cache(maxsize=1)
def available() -> bool:
    """True if torch + torchvision can be imported (deep stream usable)."""
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401

        return True
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def _backbone():
    """Build and cache the frozen feature backbone.

    Returns a callable mapping a pre-processed float tensor batch
    ``(N, 3, 224, 224)`` to an ``(N, 512)`` embedding. Cached for the process
    lifetime; the first call downloads the ImageNet weights (~45 MB).
    """
    import torch
    import torch.nn as nn
    from torchvision.models import ResNet18_Weights, resnet18

    torch.manual_seed(0)
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # Drop the 1000-way classifier; keep everything up to global average pool.
    feature_net = nn.Sequential(*list(model.children())[:-1])  # -> (N, 512, 1, 1)
    feature_net.eval()
    for p in feature_net.parameters():
        p.requires_grad_(False)
    torch.set_num_threads(max(1, (torch.get_num_threads() or 1)))
    return feature_net


# --------------------------------------------------------------------------- #
# pre-processing
# --------------------------------------------------------------------------- #


def _to_input_array(img: Image.Image) -> np.ndarray:
    """Resize an RGB image to the network input and scale to [0, 1] HWC."""
    rgb = img.convert("RGB").resize((_INPUT_SIZE, _INPUT_SIZE), Image.BILINEAR)
    return np.asarray(rgb, dtype=np.float32) / 255.0


def _spectral_image(img: Image.Image) -> np.ndarray:
    """Render the log-magnitude Fourier spectrum as a 3-channel [0,1] image.

    The grayscale image's centered 2-D FFT log-magnitude is min-max
    normalised and tiled across three channels so the same RGB-trained
    backbone can ingest it. This exposes the radial/periodic spectral
    structure that distinguishes generator up-samplers from camera optics.
    """
    g = np.asarray(img.convert("L").resize((_INPUT_SIZE, _INPUT_SIZE), Image.BILINEAR),
                   dtype=np.float32)
    spec = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(g))))
    lo, hi = float(spec.min()), float(spec.max())
    spec = (spec - lo) / (hi - lo + 1e-8)
    return np.repeat(spec[:, :, None], 3, axis=2)


def _normalise(hwc: np.ndarray) -> np.ndarray:
    """ImageNet-normalise an HWC [0,1] array and return CHW float32."""
    chw = ((hwc - _IMAGENET_MEAN) / _IMAGENET_STD).transpose(2, 0, 1)
    return np.ascontiguousarray(chw, dtype=np.float32)


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #


def embedding(img: Image.Image, spectral: bool = True) -> np.ndarray:
    """Return the deep descriptor for one image.

    With ``spectral=True`` (default) returns a 1024-d vector
    ``[spatial(512) || spectral(512)]``; otherwise the 512-d spatial part.
    Raises ``RuntimeError`` if torch/torchvision are unavailable.
    """
    return embeddings([img], spectral=spectral)[0]


def embeddings(imgs: list[Image.Image], spectral: bool = True, batch_size: int = 16) -> np.ndarray:
    """Vectorised :func:`embedding` over a list of images.

    Returns an ``(N, DEEP_DIM)`` array (or ``(N, BACKBONE_DIM)`` if
    ``spectral=False``). Batched to bound CPU memory.
    """
    if not available():
        raise RuntimeError(
            "torch/torchvision unavailable - install them or use spectral=False "
            "is not enough; the deep stream requires the backbone."
        )
    import torch

    net = _backbone()
    out: list[np.ndarray] = []
    with _lock, torch.no_grad():
        for start in range(0, len(imgs), batch_size):
            chunk = imgs[start : start + batch_size]
            spatial_batch = np.stack([_normalise(_to_input_array(im)) for im in chunk])
            feats = [net(torch.from_numpy(spatial_batch)).reshape(len(chunk), -1).numpy()]
            if spectral:
                spec_batch = np.stack([_normalise(_spectral_image(im)) for im in chunk])
                feats.append(net(torch.from_numpy(spec_batch)).reshape(len(chunk), -1).numpy())
            out.append(np.concatenate(feats, axis=1).astype(np.float32))
    return np.concatenate(out, axis=0)


def feature_names(spectral: bool = True) -> list[str]:
    """Names for each deep dimension (used by the fusion feature schema)."""
    names = [f"deep_spatial_{i}" for i in range(BACKBONE_DIM)]
    if spectral:
        names += [f"deep_spectral_{i}" for i in range(BACKBONE_DIM)]
    return names
