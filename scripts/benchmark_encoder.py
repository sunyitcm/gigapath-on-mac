#!/usr/bin/env python3
"""Benchmark the GigaPath tile encoder across batch sizes (source of the
numbers reported in the article). Random 224x224 inputs, single run.

Usage:
    HF_HUB_OFFLINE=1 python scripts/benchmark_encoder.py [--fp16] [--batches 16 32 64 128]
"""
import argparse
import time

import timm
import torch

from gigapath_wsi import HF_MODEL_ID, TILE_PX


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fp16", action="store_true", help="benchmark in half precision")
    p.add_argument("--batches", type=int, nargs="+", default=[16, 32, 64, 128])
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    m = timm.create_model(HF_MODEL_ID, pretrained=True)
    if args.fp16:
        m = m.half()
    m = m.to(device).eval()
    print(f"device={device} fp16={args.fp16}")

    def sync():
        if device == "mps":
            torch.mps.synchronize()

    # warmup: absorb one-time Metal shader compilation / caching
    x = torch.randn(16, 3, TILE_PX, TILE_PX).to(device)
    if args.fp16:
        x = x.half()
    with torch.no_grad():
        m(x)
    sync()

    # 3 runs per batch size, report the median (matches the article protocol)
    for bs in args.batches:
        x = torch.randn(bs, 3, TILE_PX, TILE_PX).to(device)
        if args.fp16:
            x = x.half()
        times = []
        for _ in range(3):
            sync()
            t0 = time.perf_counter()
            with torch.no_grad():
                m(x)
            sync()
            times.append(time.perf_counter() - t0)
        dt = sorted(times)[1]
        print(f"batch={bs}: {dt:.3f}s ({1000 * dt / bs:.1f} ms/tile, median of 3)")


if __name__ == "__main__":
    main()
