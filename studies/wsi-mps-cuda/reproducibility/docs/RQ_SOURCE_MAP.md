# RQ–source–contract map

This table links supplied components to the [existing results snapshot](https://github.com/sunyitcm/gigapath-on-mac/blob/0fe0b1f408c51580b61be9b2ab826ceedffa175a/studies/wsi-mps-cuda/RESULTS_EN.md). It does not add results or make worker excerpts independently executable.

| RQ | Components | Contract / limits |
| --- | --- | --- |
| RQ1: time and measured resources | Historical M5 source; C8 workers; `timing.py`; aggregation monitors | `historical_m5_e9.json`, `c8_common.json`; historical vs common-contract timing boundaries remain distinct. |
| RQ2: numerical consistency and local tails | Historical/later CPU comparators; cosine/SSE/statistical primitives | Original order, matching coordinates and stated reference norm; historical 512-row and later 256-row algorithms remain distinct. |
| RQ3: source and aggregation-backend substitution | Aggregation call-site excerpts; `vector_metrics.py`; `fixed_mean.py`; CPU/MPS blocked attention | `aggregation.json`; full LongNet/TITAN implementations external. Equal-input aliases are not additional executions. |
| RQ4: explicit half and AMP against the same FP32 reference | CUDA/Mac precision workers and observers; same-reference helper | `cuda_precision.json`, `mps_amp.json`; independent FP32 AMP construction, model/postprocess autocast boundary and original denominator retained. |

Recorded environments are in [Environment_Matrix.csv](../environment/Environment_Matrix.csv). Component availability and historical limits are in [SOURCE_STATUS](SOURCE_STATUS.md); [usage](USAGE.md) separates commands from excerpts. The [published result tables](https://github.com/sunyitcm/gigapath-on-mac/blob/0fe0b1f408c51580b61be9b2ab826ceedffa175a/studies/wsi-mps-cuda/DATA_INDEX.md) preserve numerical values, comparison directions and input/alias relationships.
