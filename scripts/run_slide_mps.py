#!/usr/bin/env python3
"""run_slide_mps.py — single-WSI production pipeline on Apple Silicon (MPS).

Thin CLI wrapper around gigapath_wsi.extract_features (v2 dense tiling +
mask-footprint check + marker-ink filter), plus an automatic QC overlay.

Verified on 40x, MPP 0.220 WSIs (2026-08):
  - biopsy strip  (28,341 x 35,884):   602 tiles, 18 s
  - resection     (105,133 x 101,467): 13,646 tiles, 289 s

Usage:
    HF_HUB_OFFLINE=1 python scripts/run_slide_mps.py data/your_slide.svs
"""
import sys

import openslide
from PIL import ImageDraw

from gigapath_wsi import extract_features, read_size_for_mpp, save_h5


def main(path):
    features, coords, stats = extract_features(path)

    # save CLAM-compatible h5 next to the slide
    out_h5 = path.rsplit(".", 1)[0] + ".h5"
    save_h5(out_h5, features, coords,
            attrs={"mpp": stats["mpp"], "read": stats["read_size"]})
    print(f"done: {features.shape} -> {out_h5} "
          f"({stats['n_ink_dropped']} ink tiles dropped, {stats['total_s']}s)")

    # QC overlay: boxes reflect the true READ footprint (before resize to 224)
    slide = openslide.OpenSlide(path)
    W, H = slide.dimensions
    read_size, _ = read_size_for_mpp(slide)
    ow = 1024
    sc = ow / W
    oh = int(H * sc)
    base = slide.get_thumbnail((ow, oh)).convert("RGB")
    slide.close()
    d = ImageDraw.Draw(base)
    bw = max(1, int(read_size * sc))
    for (x0, y0) in coords:
        d.rectangle([x0 * sc, y0 * sc, x0 * sc + bw, y0 * sc + bw],
                    outline=(255, 0, 0))
    out_png = path.rsplit(".", 1)[0] + "_check.png"
    base.save(out_png)
    print(f"QC overlay -> {out_png}")


if __name__ == "__main__":
    main(sys.argv[1])
