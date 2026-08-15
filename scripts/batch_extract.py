#!/usr/bin/env python3
"""Batch-extract GigaPath features for a folder of WSIs (one .h5 per slide).

The model is loaded once and reused across slides (~15 s one-time cost);
slides whose .h5 already exists are skipped, so interrupted runs resume cleanly.

Usage:
    HF_HUB_OFFLINE=1 python scripts/batch_extract.py \
        --svs-dir /path/to/svs_folder --out-dir /path/to/h5_out [--n-files 10]
"""
import argparse
import glob
import os
import time

import h5py

from gigapath_wsi import extract_features, load_tile_encoder, save_h5


def is_valid_h5(path: str) -> bool:
    """Guard against resuming from a truncated .h5 left by a killed process."""
    try:
        with h5py.File(path, "r") as h:
            return "features" in h and "coords" in h
    except Exception:
        return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--svs-dir", required=True, help="folder containing .svs files")
    p.add_argument("--out-dir", required=True, help="folder for output .h5 files")
    p.add_argument("--n-files", type=int, default=None,
                   help="process only the first N slides (default: all)")
    p.add_argument("--batch-size", type=int, default=None)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.svs_dir, "*.svs")))
    if args.n_files:
        files = files[: args.n_files]
    print(f"{len(files)} slide(s) queued")

    t0 = time.time()
    model, device, fp16 = load_tile_encoder()          # one-time load (~15 s)
    print(f"model loaded on {device} (fp16={fp16})")

    for i, path in enumerate(files, 1):
        name = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(args.out_dir, name + ".h5")
        if os.path.exists(out):
            if is_valid_h5(out):
                print(f"[{i}/{len(files)}] skip (valid): {name}")
                continue
            print(f"[{i}/{len(files)}] re-do (corrupt h5): {name}")
        print(f"[{i}/{len(files)}] processing: {name}")
        features, coords, stats = extract_features(
            path, model=model, device=device, fp16=fp16,
            batch_size=args.batch_size, verbose=False)
        save_h5(out, features, coords,
                attrs={"mpp": stats["mpp"], "read": stats["read_size"]})
        print(f"    {stats['n_tiles']} tiles ({stats['n_ink_dropped']} ink dropped) "
              f"in {stats['total_s']}s -> {out}")

    print(f"all done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
