# figures

Place QC overlays here, e.g. the output of:

```bash
python scripts/visualize_tiles.py --wsi <slide.svs> --h5 <slide.h5> --out figures/tiles_check.png
```

Recommended figures for this repo:

| file | source |
|---|---|
| `tiles_check.png` | `visualize_tiles.py` — tile placement over the tissue mask |
| `benchmark_table.md` | copy of `benchmark_encoder.py` console output |

Note: `*_check.png` is git-ignored by default (it may contain patient-derived imagery).
Un-ignore it in `.gitignore` only for de-identified demo slides.
