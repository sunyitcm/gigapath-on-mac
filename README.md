# GigaPath on Mac

**Running a 1B-parameter pathology foundation model (GigaPath) locally without an NVIDIA GPU: deployment, benchmarks, and a complete WSI feature-extraction pipeline on Intel CPU and Apple Silicon (MPS).**

[中文文档 (Chinese)](README_CN.md) · [Deployment manuals](docs/) · [Benchmarks](#benchmarks)

---

## Why this repo

Every major pathology foundation model (GigaPath, UNI, Virchow) documents NVIDIA + CUDA as the deployment target, and the only Mac-related trace upstream is an unanswered installation-failure issue ([prov-gigapath#36](https://github.com/prov-gigapath/prov-gigapath/issues)). This repository provides, to our knowledge, the first measured account of running a billion-parameter pathology foundation model on macOS:

- a working deployment recipe for **both Intel (CPU-only) and Apple Silicon (MPS)** Macs;
- **measured** tile-encoding throughput, fp16-vs-fp32 fidelity, and end-to-end WSI timing;
- a production-ready extraction pipeline (tissue detection, physical-scale/MPP matching, HDF5 caching) that outputs **CLAM-compatible** feature files;
- a troubleshooting guide for macOS-specific failure modes (HF cache structure, TLS-via-system-, OpenSlide dylib).

## Key findings

1. Apple MPS delivers **12–58 ms per tile** for the GigaPath ViT-G/14 tile encoder depending on chip generation (M5 Max fp16: 12 ms; older generation: 58 ms) — 15–40× faster than Intel CPU† and roughly RTX-4070-class territory on the newest chip.
2. Inference throughput is **independent of unified-memory capacity** (verified across 16–256 GB) and is decided by the chip's matrix-acceleration architecture: the 40-core M5 Max beats the 80-core M3 Ultra by 2.4× on fp16, because only the newer generation has per-core neural accelerators (fp16 gain 3.7× vs 1.15×).
3. fp16 inference is **numerically safe**: per-tile cosine similarity vs fp32 ≥ 0.9995 (slide-level 0.999997).
4. A 200-slide cohort (~600k tiles) extracts **overnight on a single Mac** (~11 h); aggregation and downstream MIL modeling are CPU-cheap.
5. Unified memory (64–128 GB addressable by the GPU) removes the VRAM ceiling that constrains consumer NVIDIA cards.

† Intel CPU figure is an order-of-magnitude estimate from interactive runs, not an identical-input benchmark; see [measurement notes](#measurement-notes).

## Benchmarks

### Tile-encoding latency (GigaPath ViT-G/14, random 224×224 inputs, single run)

| Platform | Precision | Batch | Wall time | ms/tile |
|---|---|---|---|---|
| Intel CPU (16 GB)† | fp32 | 8 | — | 1000–3000 |
| Apple MPS (64 GB) | fp32 | 16 | 1.22 s | 76 |
| Apple MPS | fp32 | 64 | 4.70 s | 73 |
| Apple MPS | fp32 | 128 | 9.93 s | 77 |
| Apple MPS | fp16 | 16 | 1.39 s | 87 |
| Apple MPS | fp16 | 32 | 1.91 s | 60 |
| Apple MPS | fp16 | 64 | **3.73 s** | **58** |
| Apple MPS | fp16 | 128 | 8.36 s | 65 |

Reproduce: `python scripts/benchmark_encoder.py [--fp16]`

### fp16 vs fp32 fidelity (101 identical tissue tiles)

| Metric | Value |
|---|---|
| Per-tile cosine, mean | 0.999949 |
| Per-tile cosine, min | 0.999466 |
| Slide-level (mean-pool) cosine | 0.999997 |

Reproduce: `python scripts/verify_fp16.py --wsi <slide.svs>`

### End-to-end WSI timing (40× biopsy WSI, 24,296×26,654 px, ~4% tissue, 101 tiles; MPS fp16, batch 64)

| Stage | Time | Per tile |
|---|---|---|
| Tissue detection (Otsu) | 0.1 s | — |
| Tile read + resize | 0.7 s | 7 ms |
| MPS encoding | 5.8 s | 57 ms |
| Model load (one-time) | ~15 s | — |
| **Total** | **21.5 s** | **marginal ≈ 64 ms/tile** |

Note the two cost conventions: amortizing the one-time model load over 101 tiles gives an apparent 213 ms/tile; the marginal cost (what matters for batch processing) is ≈ 64 ms/tile.

**Known limitation (dense tiling v2):** the marker-ink filter (drop tiles with >30% near-black pixels) does not catch ink mixed with folded tissue; such tiles pass through. This is acceptable for weakly-supervised MIL (attention downweights them), and most standard pipelines (CLAM et al.) do not filter pen marks either. For strict artifact removal, replace the Otsu mask with a DeepLabV3 tissue segmenter (e.g. TRIDENT).

#### Four-platform comparison (fp32/fp16 at batch 64, warmup protocol, median of 3)

| Platform | Chip | GPU cores | fp32 | fp16 | fp16 gain | fp16 ms/tile | fp16 cosine |
|---|---|---|---|---|---|---|---|
| Intel 16 GB | Intel CPU | — | — | n/a | — | ~1–3 s/tile † | — |
| Mac 64 GB | Apple Silicon (M2 Max)  | 30 | 4.70 s | 3.73 s | 1.26× | 58 | 0.999949 |
| Mac 128 GB | Apple M5 Max | 40 | 2.85 s | **0.77 s** | **3.7×** | **12** | 0.999947 |
| Mac 256 GB | Apple M3 Ultra  | 80 | 2.08 s | 1.81 s | 1.15× | 28 | 0.999947 |


Three conclusions: (1) throughput is **independent of RAM capacity** (16–256 GB spans 16×, ranking does not follow memory); (2) the deciding variable is the **generational matrix-acceleration architecture** — the 40-core M5 Max (per-core neural accelerators, fp16 gain 3.7×) beats the 80-core M3 Ultra (fp16 gain 1.15×) by 2.4× on fp16 inference; (3) under fp32, where neither generation has a dedicated path, core count wins as expected (M3 Ultra 2.08 s < M5 Max 2.85 s) — ruling out confounders. fp16 fidelity is reproducible across machines (cosine 0.999947–0.999949), so features extracted on different machines can be pooled.

### Real-world WSI throughput (dense tiling v2, M5 Max 128 GB, fp16, batch 64, 40x, MPP 0.220)

| Slide | Dimensions | Tissue | Tiles | Wall time | tiles/s |
|---|---|---|---|---|---|
| Biopsy strip | 28,341 × 35,884 | 13.0% | 602 | 18 s | 33 |
| Resection | 105,133 × 101,467 | ~30% | 13,646 | 289 s | 47 |

Real-world marginal cost ≈ 21–30 ms/tile including disk I/O and resizing (vs 12 ms in synthetic benchmarks — I/O overhead ≈ 4 ms/tile on the encoder side, the rest is `read_region` on large files). QC overlays confirmed dense tissue coverage and correct exclusion of adipose regions.

### Rough comparison with NVIDIA GPUs (engineering estimates, ±50%, NOT measured)

| Hardware | ms/tile | 10k-tile WSI |
|---|---|---|
| Apple Silicon (MPS fp16, this work) | ~58 | ~10 min |
| RTX 4060 class (est.) | ~20–30 | 3–5 min |
| RTX 4090 (est.) | ~5–8 | ~1 min |
| A100 (est.) | ~2–4 | <1 min |

## Installation (Apple Silicon)

```bash
conda create -n gigapath python=3.11 -y && conda activate gigapath
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio
python -m pip install -r requirements.txt

# weights are gated: accept the license at
# https://huggingface.co/prov-gigapath/prov-gigapath first, then
export HF_TOKEN=hf_your_token
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('prov-gigapath/prov-gigapath'))"
```

Smoke test (offline once weights are cached):

```bash
HF_HUB_OFFLINE=1 python -c "
import torch, timm
m = timm.create_model('hf_hub:prov-gigapath/prov-gigapath', pretrained=True).to('mps').eval()
print(m(torch.randn(1,3,224,224).to('mps')).shape)   # torch.Size([1, 1536])
"
```

Detailed, error-by-error guides for both platforms: [docs/](docs/).

## Quickstart

```bash
conda activate gigapath

# production path (v2): dense tiling + mask footprint + ink filter + QC overlay
HF_HUB_OFFLINE=1 python scripts/run_slide_mps.py /path/to/slide.svs

# library-based alternative -> one .h5 (features: (N,1536), coords: (N,2))
HF_HUB_OFFLINE=1 python scripts/extract_features.py \
    --wsi /path/to/slide.svs --out /path/to/slide.h5

# ALWAYS visually QC tile placement on a few slides
python scripts/visualize_tiles.py \
    --wsi /path/to/slide.svs --h5 /path/to/slide.h5 --out tiles_check.png

# batch a folder (model loads once; existing .h5 files are skipped)
HF_HUB_OFFLINE=1 python scripts/batch_extract.py \
    --svs-dir /path/to/svs --out-dir /path/to/h5_out
```

## Repository layout

```
├── README.md / README_CN.md    # this file / full Chinese article
├── docs/                       # platform-specific deployment manuals (EN-ready CN)
├── scripts/
│   ├── gigapath_wsi.py         # core library (shared by all CLIs)
│   ├── run_slide_mps.py        # v2 production CLI: dense tiling + mask footprint + ink filter
│   ├── extract_features.py     # single WSI -> .h5 (library-based)
│   ├── batch_extract.py        # folder of WSIs -> one .h5 each
│   ├── visualize_tiles.py      # tile-placement QC overlay
│   ├── verify_fp16.py          # fp16-vs-fp32 cosine fidelity check
│   └── benchmark_encoder.py    # batch-size benchmark (article Table 1)
├── figures/                    # QC images (add your own tiles_check.png)
├── requirements.txt
└── LICENSE
```

## Roadmap

- [x] CPU/MPS deployment + measured benchmarks
- [x] WSI extraction pipeline with MPP matching and HDF5 caching
- [ ] Marker-ink masking for pen-annotated slides (HSV filtering)
- [ ] Slide-level aggregation: official LongNet slide encoder & ABMIL baselines
- [ ] CLAM integration (features here are already CLAM-compatible)
- [ ] Cohort-scale weakly supervised modeling with patient-level cross-validation

## Measurement notes

- Environment: Intel Mac 16 GB (CPU-only) and Apple Silicon 64 GB (MPS); torch 2.x, timm 1.x.
- All measured figures are **single-run** results on one machine; expect ±20% variation across hardware and load.
- NVIDIA figures are engineering estimates derived from compute/bandwidth specs, **not** measurements — treat them as orientation, not benchmarking.
- The extraction sample is a small biopsy WSI (~4% tissue); large-resection figures are extrapolations at the measured marginal per-tile cost.

## References

1. Xu H, et al. *A whole-slide foundation model for digital pathology from real-world data.* Nature 630, 181–188 (2024). https://doi.org/10.1038/s41586-024-07441-w
2. GigaPath code: https://github.com/prov-gigapath/prov-gigapath · weights (gated): https://huggingface.co/prov-gigapath/prov-gigapath
3. Chen RJ, et al. *Towards a general-purpose foundation model for computational pathology (UNI).* Nat Med 30, 850–862 (2024).
4. Lu MY, et al. *Data-efficient and weakly supervised computational pathology on whole-slide images (CLAM).* Nat Biomed Eng 5, 555–570 (2021). https://github.com/mahmoodlab/CLAM
5. PyTorch MPS backend: https://developer.apple.com/metal/pytorch/
6. Hugging Face Hub cache semantics: https://huggingface.co/docs/huggingface_hub/guides/manage-cache

## License & disclaimer

Code in this repository is released under the [MIT License](LICENSE). Model weights remain subject to the GigaPath license on Hugging Face. This project is a research/engineering note, **not** a medical device; no clinical use is implied or intended.
