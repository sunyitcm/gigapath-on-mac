"""Core utilities for GigaPath tile-level feature extraction on macOS (CPU / Apple MPS).

This module is the shared foundation for all CLI scripts in this repository.
Pipeline stages (tissue detection -> MPP-matched dense tiling -> batch encoding
-> HDF5) are deliberately separated from downstream aggregation/modeling so that
extracted features can be cached and reused.

v2 (2026-08): dense tiling (stride = READ) with per-tile mask-footprint check,
replacing the v1 thumbnail-window-center sampling which covered only ~3% of
tissue on large slides. Adds an optional marker-ink filter.
"""
from __future__ import annotations

import time

import cv2
import h5py
import numpy as np
import openslide
import timm
import torch
from PIL import Image

# HuggingFace repo id in timm "hf_hub:" notation (gated; accept the license on HF first).
HF_MODEL_ID = "hf_hub:prov-gigapath/prov-gigapath"

# ImageNet normalization used by the GigaPath tile encoder.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]

TILE_PX = 224          # model input edge length (pixels)
TARGET_MPP = 0.5       # GigaPath training resolution: 0.5 microns/pixel (20x equivalent)
DEFAULT_MPP = 0.25     # fallback if the SVS lacks mpp metadata (typical for 40x scans)
MASK_STRIDE = 16       # tissue-mask resolution: 1 mask px = 16 level-0 px


def pick_device() -> str:
    """Return 'mps' on Apple Silicon with a working Metal backend, else 'cpu'."""
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_tile_encoder(device: str | None = None, fp16: bool | None = None):
    """Load the frozen GigaPath tile encoder (ViT-G/14, ~1B params).

    fp16 defaults to True on MPS (measured up to 3.7x faster on M5 Max with
    per-tile cosine similarity > 0.9999 vs fp32; see verify_fp16.py).
    """
    device = device or pick_device()
    fp16 = (device == "mps") if fp16 is None else fp16
    model = timm.create_model(HF_MODEL_ID, pretrained=True)
    if fp16:
        model = model.half()
    return model.to(device).eval(), device, fp16


def tissue_mask(slide: openslide.OpenSlide, mask_stride: int = MASK_STRIDE):
    """Binary tissue mask at reduced resolution via HSV-saturation Otsu.

    Saturation (not grayscale) is thresholded: tissue is pink/purple (high
    saturation), glass background and adipose are pale (low saturation).
    A median blur and 3x3 morphological closing suppress noise and pinholes.

    Returns (mask, t_otsu): mask[my, mx] is True where tissue is present
    (one mask pixel covers mask_stride x mask_stride level-0 pixels);
    t_otsu is the Otsu threshold actually used, for logging/reproducibility.
    """
    W, H = slide.dimensions
    mw = (W + mask_stride - 1) // mask_stride
    mh = (H + mask_stride - 1) // mask_stride
    thumb = np.asarray(slide.get_thumbnail((mw, mh)).convert("RGB"))
    s = cv2.cvtColor(thumb, cv2.COLOR_RGB2HSV)[:, :, 1]
    s = cv2.medianBlur(s, 5)
    t_otsu, _ = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (s > t_otsu).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return mask, t_otsu


def read_size_for_mpp(slide: openslide.OpenSlide, target_mpp: float = TARGET_MPP):
    """Physical-scale matching: pixels to read at level 0 so that, after resizing
    to 224x224, the patch matches the 0.5-MPP training distribution.

    A 40x scan (mpp~0.22) must read ~510 px, not 224 px.
    """
    mpp = float(slide.properties.get("openslide.mpp-x", DEFAULT_MPP))
    return max(round(TILE_PX * target_mpp / mpp), TILE_PX), mpp


def dense_tile_coords(mask: np.ndarray, dims: tuple[int, int], read_size: int,
                      tissue_frac: float = 0.25,
                      mask_stride: int = MASK_STRIDE) -> list[tuple[int, int]]:
    """Dense tiling over the whole slide (stride = read_size, non-overlapping).

    A tile is kept if at least `tissue_frac` of its footprint in the mask is
    tissue. This is the CLAM-style coverage strategy; v1 sampled one tile per
    thumbnail window and covered only ~3% of tissue on large slides.
    """
    W, H = dims
    fpx = max(1, read_size // mask_stride)
    coords = []
    for y0 in range(0, H, read_size):
        my = y0 // mask_stride
        for x0 in range(0, W, read_size):
            mx = x0 // mask_stride
            fp = mask[my:my + fpx, mx:mx + fpx]
            if fp.size and fp.mean() >= tissue_frac:
                coords.append((x0, y0))
    return coords


def is_ink(tile_rgb: np.ndarray, dark_thresh: int = 60,
           frac_thresh: float = 0.30) -> bool:
    """Marker-ink heuristic: True if > frac_thresh of pixels are near-black.

    Known limitation: ink mixed with folded tissue may pass; acceptable under
    weakly-supervised MIL (attention downweights such tiles). For strict
    artifact removal use a segmentation model (e.g. TRIDENT DeepLabV3).
    """
    small = np.asarray(Image.fromarray(tile_rgb).resize((64, 64)))
    v = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)[:, :, 2]
    return (v < dark_thresh).mean() > frac_thresh


@torch.no_grad()
def extract_features(
    slide_path: str,
    model: torch.nn.Module | None = None,
    device: str | None = None,
    fp16: bool | None = None,
    batch_size: int | None = None,
    max_tiles: int | None = None,
    tissue_frac: float = 0.25,
    pen_filter: bool = True,
    mask_stride: int = MASK_STRIDE,
    target_mpp: float = TARGET_MPP,
    verbose: bool = True,
):
    """Extract tile features from one WSI (v2 dense-tiling pipeline).

    Returns (features, coords, stats):
        features: float32 array (N, 1536)
        coords:   int array (N, 2), level-0 pixel coordinates of each tile origin
        stats:    dict with timing / counts
    """
    t_start = time.time()
    if model is None:
        model, device, fp16 = load_tile_encoder(device, fp16)
    device = device or pick_device()
    fp16 = (device == "mps") if fp16 is None else fp16
    batch_size = batch_size or (64 if device == "mps" else 8)

    slide = openslide.OpenSlide(slide_path)
    read_size, mpp = read_size_for_mpp(slide, target_mpp)
    mask, t_otsu = tissue_mask(slide, mask_stride)
    all_coords = dense_tile_coords(mask, slide.dimensions, read_size,
                                   tissue_frac, mask_stride)
    if max_tiles:
        all_coords = all_coords[:max_tiles]
    if verbose:
        print(f"slide={slide.dimensions} mpp={mpp:.4f} read_size={read_size} "
              f"tissue={mask.mean() * 100:.1f}% (Otsu t={t_otsu:.0f}) "
              f"candidates={len(all_coords)} device={device} fp16={fp16} "
              f"batch={batch_size}", flush=True)

    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    feats, coords, imgs, kept = [], [], [], []
    t_read, t_enc, n_ink = 0.0, 0.0, 0

    def flush():
        nonlocal t_enc
        if not imgs:
            return
        t = torch.from_numpy(np.stack(imgs)).float().to(device) / 255.0
        t = (t - mean) / std
        if fp16:
            t = t.half()
        t0 = time.time()
        out = model(t)
        if device == "mps":
            torch.mps.synchronize()
        t_enc += time.time() - t0
        feats.append(out.float().cpu().numpy())
        coords.extend(kept)
        imgs.clear()
        kept.clear()
        if verbose:
            print(f"  encoded {len(coords)} tiles", flush=True)

    for (x, y) in all_coords:
        t0 = time.time()
        tile = (slide.read_region((x, y), 0, (read_size, read_size))
                .convert("RGB").resize((TILE_PX, TILE_PX), Image.LANCZOS))
        arr = np.array(tile)
        t_read += time.time() - t0
        if pen_filter and is_ink(arr):
            n_ink += 1
            continue
        imgs.append(arr.transpose(2, 0, 1))
        kept.append([x, y])
        if len(imgs) >= batch_size:
            flush()
    flush()
    slide.close()

    features = np.concatenate(feats) if feats else np.zeros((0, 1536), np.float32)
    coords = np.array(coords[: len(features)], dtype=np.int32)
    stats = {
        "n_tiles": len(features),
        "n_ink_dropped": n_ink,
        "read_resize_s": round(t_read, 2),
        "encode_s": round(t_enc, 2),
        "total_s": round(time.time() - t_start, 2),
        "mpp": mpp,
        "read_size": read_size,
    }
    if verbose:
        print(f"done: {stats}")
    return features, coords, stats


def save_h5(path: str, features: np.ndarray, coords: np.ndarray,
            attrs: dict | None = None) -> None:
    """One slide = one HDF5 file (CLAM-compatible convention)."""
    with h5py.File(path, "w") as h:
        h.create_dataset("features", data=features, compression="gzip")
        h.create_dataset("coords", data=coords)
        h.attrs["version"] = "v2"
        for k, v in (attrs or {}).items():
            h.attrs[k] = v
