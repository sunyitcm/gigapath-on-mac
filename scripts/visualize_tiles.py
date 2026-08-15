#!/usr/bin/env python3
"""Overlay extracted tile locations (from a .h5) on the WSI thumbnail for QC.

ALWAYS run this after extraction on at least a few slides: silent coordinate
bugs (e.g., tiles landing on blank glass or marker ink) are only visible here.

Usage:
    python scripts/visualize_tiles.py \
        --wsi /path/to/slide.svs --h5 /path/to/slide.h5 --out tiles_check.png
"""
import argparse

import h5py
import openslide
from PIL import ImageDraw

from gigapath_wsi import read_size_for_mpp


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wsi", required=True)
    p.add_argument("--h5", required=True)
    p.add_argument("--out", default="tiles_check.png")
    p.add_argument("--thumb-size", type=int, default=1024)
    args = p.parse_args()

    slide = openslide.OpenSlide(args.wsi)
    read_size, _ = read_size_for_mpp(slide)
    thumb = slide.get_thumbnail((args.thumb_size, args.thumb_size)).convert("RGB")
    sx = slide.dimensions[0] / thumb.size[0]
    sy = slide.dimensions[1] / thumb.size[1]

    with h5py.File(args.h5, "r") as h:
        coords = h["coords"][:]

    draw = ImageDraw.Draw(thumb)
    for i, (x, y) in enumerate(coords):
        # rectangle reflects the true READ footprint (before resize to 224)
        box = [x / sx, y / sy, (x + read_size) / sx, (y + read_size) / sy]
        draw.rectangle(box, outline="red", width=2)
        if i < 50:  # number only the first 50 tiles; labels blur at N > ~1000
            draw.text((x / sx, y / sy), str(i + 1), fill="red")

    thumb.save(args.out)
    print(f"saved {args.out} with {len(coords)} tile boxes (read_size={read_size})")


if __name__ == "__main__":
    main()
