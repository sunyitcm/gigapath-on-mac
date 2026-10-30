# WSI Results and Methods Snapshot

[中文](README_CN.md) · [Results](RESULTS_EN.md) · [中文结果](RESULTS_CN.md) · [Methods](METHODS_AND_MEASUREMENT_CONTRACTS_EN.md) · [Data and figures](DATA_INDEX.md) · [WSI sources](../../docs/WSI_INPUTS.md) · [References](REFERENCES.md) · [Rights](RIGHTS_AND_SCOPE.md) · [Third-party notices](THIRD_PARTY_NOTICES.md)

The study examines numerical consistency and measured resource costs of a fixed pathology-feature workflow on **M5 Max/Apple MPS, M3 Ultra/Apple MPS and RTX 5090/NVIDIA CUDA**. Tile encoders are **GigaPath, Virchow2 and CONCH v1.5**; fixed learned aggregation routes are compatible GigaPath and TITAN.

| Question | Descriptive finding |
| --- | --- |
| RQ1: How do platform, precision and batch relate to time and resources? | Observed ordering depends on the precision policy and measurement boundary. |
| RQ2: Do global agreement and local tails tell different stories? | Close global features can coexist with concentrated local error; cross-Mac equality is model-dependent. |
| RQ3: How do source replacement and aggregation-backend replacement differ? | Upstream substitution and identical-input backend substitution are distinct comparison axes. |
| RQ4: What are the same-FP32-reference tradeoffs of explicit half and AMP? | AMP lowers global error in the tested conditions; local tails and memory observations prevent an all-metric conclusion. |

This **Results and Methods Snapshot** contains a bounded results report, methods, three figures and supporting scalar tables. It is not the complete paper or a full from-WSI reproduction package. Observations are limited to the stated WSIs, sessions and contracts; they do not establish clinical equivalence, universal determinism or hardware causes.

The earlier [v1.0 exploration](https://github.com/sunyitcm/gigapath-on-mac/releases/tag/v1.0) uses different measurement contracts. Source data, model weights and third-party implementations are not distributed here.
