#!/usr/bin/env python3
"""Extract GigaPath tile features from a single WSI into one HDF5 file.

Usage:
    HF_HUB_OFFLINE=1 python scripts/extract_features.py \
        --wsi /path/to/slide.svs --out /path/to/slide.h5 [--max-tiles 500]
"""
import argparse

from gigapath_wsi import extract_features, save_h5


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wsi", required=True, help="path to the input .svs file")
    p.add_argument("--out", required=True, help="path of the output .h5 file")
    p.add_argument("--max-tiles", type=int, default=None,
                   help="cap the number of tissue tiles (default: no cap)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="encoding batch size (default: 64 on MPS, 8 on CPU)")
    args = p.parse_args()

    features, coords, stats = extract_features(
        args.wsi, max_tiles=args.max_tiles, batch_size=args.batch_size)
    save_h5(args.out, features, coords,
            attrs={"mpp": stats["mpp"], "read": stats["read_size"]})
    print(f"saved {stats['n_tiles']} tiles {features.shape} -> {args.out}")


if __name__ == "__main__":
    main()
