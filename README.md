# Pathology Foundation Models on Apple MPS and NVIDIA CUDA

[中文](README_CN.md) · **[Study snapshot](studies/wsi-mps-cuda/README.md)** · [WSI sources](docs/WSI_INPUTS.md)

The study examines numerical consistency and measured resource costs of a fixed pathology-feature workflow on **M5 Max/Apple MPS, M3 Ultra/Apple MPS and RTX 5090/NVIDIA CUDA**. Tile encoders are **GigaPath, Virchow2 and CONCH v1.5**; fixed learned aggregation routes are compatible GigaPath and TITAN.

| Question | Descriptive finding |
| --- | --- |
| RQ1: How do platform, precision and batch relate to time and resources? | Observed ordering depends on the precision policy and measurement boundary. |
| RQ2: Do global agreement and local tails tell different stories? | Close global features can coexist with concentrated local error; cross-Mac equality is model-dependent. |
| RQ3: How do source replacement and aggregation-backend replacement differ? | Upstream substitution and identical-input backend substitution are distinct comparison axes. |
| RQ4: What are the same-FP32-reference tradeoffs of explicit half and AMP? | AMP lowers global error in the tested conditions; local tails and memory observations prevent an all-metric conclusion. |

This **Results and Methods Snapshot** contains a bounded results report, methods, three figures and supporting scalar tables. It is not the complete paper or a full from-WSI reproduction package. Observations are limited to the stated WSIs, sessions and contracts; they do not establish clinical equivalence, universal determinism or hardware causes.

[Results and figures](studies/wsi-mps-cuda/RESULTS_EN.md) · [Methods](studies/wsi-mps-cuda/METHODS_AND_MEASUREMENT_CONTRACTS_EN.md) · [Scalar tables](studies/wsi-mps-cuda/DATA_INDEX.md) · [Primary references](studies/wsi-mps-cuda/REFERENCES.md) · [Rights](studies/wsi-mps-cuda/RIGHTS_AND_SCOPE.md) · [Third-party notices](studies/wsi-mps-cuda/THIRD_PARTY_NOTICES.md)

The retained scripts and [v1.0 source](https://github.com/sunyitcm/gigapath-on-mac/tree/v1.0) belong to the early GigaPath-on-Mac exploration. Their synthetic/101-tile contracts and NVIDIA estimates differ from this study; the old publication date applies only to the content then released.

[Code and Methods Supplement v1.1.0](studies/wsi-mps-cuda/reproducibility/README.md): scripts, source components/excerpts, configurations, recorded environments, usage and licenses.
