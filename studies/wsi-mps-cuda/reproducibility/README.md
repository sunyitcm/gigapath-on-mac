# Code and Methods Supplement v1.1.0

[中文](README_CN.md) · [Usage](docs/USAGE.md) · [Source components](docs/SOURCE_STATUS.md) · [Contracts](docs/CONTRACTS.md) · [Environments](environment/Environment_Matrix.csv) · [Licenses](docs/LICENSE_SCOPE.md)

Code components for the [WSI MPS–CUDA Results and Methods Snapshot v1.0.0](https://github.com/sunyitcm/gigapath-on-mac/releases/tag/wsi-mps-cuda-study-v1.0.0): preprocessing, timing/resource sampling, precision observation, CPU comparison, aggregation-call excerpts and a public-scalar report/plot helper. Configurations and recorded environments cover the M5 Max/MPS, M3 Ultra/MPS and RTX 5090/CUDA study.

- `scripts/plan.py`: configuration planner; no model dispatch.
- `scripts/compare_h5.py`: explicitly enabled CPU comparison interface; not executed for this supplement.
- `scripts/scalar_report.py`: report and plots from existing published CSV/JSON.
- `source/historical_m5/`: GigaPath-only single-task derivative, retained preprocessing and archival sweep text.
- `source/worker_excerpts/`: scientific source excerpts with omitted orchestration/dependencies; **not standalone workers**. The included blocked-attention component accepts CPU/MPS only.

See the [RQ/source map](docs/RQ_SOURCE_MAP.md), [model inputs](docs/INPUTS_MODELS.md) and [validation limits](docs/VALIDATION_AND_GAPS.md). End-to-end execution and clean-environment installation are unvalidated. Historical final-H5 producing batch/repeat and the exact historical M3 executable are not recovered by the M5 E9 batch-level source evidence.

Existing scalar results and three figures remain unchanged; no new experiment is added. Models, weights, WSIs and feature arrays are not bundled. The ZIP contains code and necessary documentation; published scalar inputs are obtained separately as described in usage. Project software and attributed third-party portions retain the [stated license scopes](docs/LICENSE_SCOPE.md).
