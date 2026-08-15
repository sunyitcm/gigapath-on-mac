#!/usr/bin/env python3
"""Verify fp16-vs-fp32 feature fidelity on identical tiles (same coords, same input).

Acceptance criterion used in the accompanying article:
mean per-tile cosine similarity > 0.999.

Usage:
    HF_HUB_OFFLINE=1 python scripts/verify_fp16.py --wsi /path/to/slide.svs [--n 200]
"""
import argparse

import numpy as np
import openslide
import timm
import torch
from PIL import Image

from gigapath_wsi import (HF_MODEL_ID, IMAGENET_MEAN, IMAGENET_STD, TILE_PX,
                          dense_tile_coords, read_size_for_mpp, tissue_mask)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wsi", required=True)
    p.add_argument("--n", type=int, default=200, help="number of tissue tiles to compare")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # 1) deterministic tile coordinates (shared by both precisions)
    slide = openslide.OpenSlide(args.wsi)
    read_size, _ = read_size_for_mpp(slide)
    mask, _ = tissue_mask(slide)
    coords = dense_tile_coords(mask, slide.dimensions, read_size)[: args.n]
    print(f"tiles compared: {len(coords)}")

    # 2) preprocess ONCE in fp32 so preprocessing differences cannot leak in
    tiles = []
    for x, y in coords:
        t = (slide.read_region((x, y), 0, (read_size, read_size))
             .convert("RGB").resize((TILE_PX, TILE_PX), Image.LANCZOS))
        tiles.append(np.array(t).transpose(2, 0, 1))
    slide.close()
    T = torch.from_numpy(np.stack(tiles)).float() / 255.0
    T = (T - IMAGENET_MEAN) / IMAGENET_STD

    def encode(fp16: bool) -> np.ndarray:
        m = timm.create_model(HF_MODEL_ID, pretrained=True)
        if fp16:
            m = m.half()
        m = m.to(device).eval()
        outs = []
        for i in range(0, len(T), args.batch_size):
            x = T[i:i + args.batch_size].to(device)
            if fp16:
                x = x.half()
            with torch.no_grad():
                outs.append(m(x).float().cpu())
            if device == "mps":
                torch.mps.synchronize()
        del m
        return torch.cat(outs).numpy()

    f32, f16 = encode(False), encode(True)
    per_tile = cosine(f32, f16)
    slide_cos = float(cosine(f32.mean(0)[None], f16.mean(0)[None])[0])

    print(f"per-tile cosine: mean={per_tile.mean():.6f} "
          f"min={per_tile.min():.6f} median={np.median(per_tile):.6f}")
    print(f"slide-level (mean-pool) cosine: {slide_cos:.6f}")
    print("verdict:", "PASS (> 0.999)" if per_tile.mean() > 0.999 else "BELOW 0.999 - report honestly")


if __name__ == "__main__":
    main()
