# Three-Platform Engineering Evaluation of Pathology Foundation Models: Apple MPS and NVIDIA CUDA — Results and Methods Snapshot

Tag: `wsi-mps-cuda-study-v1.0.0`

This **Results and Methods Snapshot** examines a fixed pathology-feature workflow on M5 Max/Apple MPS, M3 Ultra/Apple MPS and RTX 5090/NVIDIA CUDA. Tile models are GigaPath, Virchow2 and CONCH v1.5; learned aggregation routes are compatible GigaPath and TITAN.

Four descriptive questions concern time/resources; global agreement versus local tails; source replacement versus identical-input aggregation backends; and explicit half versus AMP against the same FP32 reference. Observations show precision-dependent timing order, model-dependent cross-Mac equality and concentrated local error. AMP lowers global numerical error in the tested conditions without an all-metric or clinical-equivalence claim.

Included are concise reports, methods, three figures, supporting scalar tables, WSI/model/literature sources and integrity lists. Source data, feature arrays, weights and third-party implementations are not included. Sessions, reference directions, dependent units, r1 limits and event-based resource sampling are defined in the methods. This is not the complete paper or a full end-to-end reproduction package.

The earlier **v1.0 Release was published on 16 August 2026** for GigaPath-on-Mac exploration. Its measurement contracts and estimated NVIDIA observations differ from this study. That date applies only to the content then published, not to these additional results. No first-comparison claim is made.

[Results](https://github.com/sunyitcm/gigapath-on-mac/blob/wsi-mps-cuda-study-v1.0.0/studies/wsi-mps-cuda/RESULTS_EN.md) · [Methods](https://github.com/sunyitcm/gigapath-on-mac/blob/wsi-mps-cuda-study-v1.0.0/studies/wsi-mps-cuda/METHODS_AND_MEASUREMENT_CONTRACTS_EN.md) · [Tables](https://github.com/sunyitcm/gigapath-on-mac/blob/wsi-mps-cuda-study-v1.0.0/studies/wsi-mps-cuda/DATA_INDEX.md) · [Rights](https://github.com/sunyitcm/gigapath-on-mac/blob/wsi-mps-cuda-study-v1.0.0/studies/wsi-mps-cuda/RIGHTS_AND_SCOPE.md) · [Third-party notices](https://github.com/sunyitcm/gigapath-on-mac/blob/wsi-mps-cuda-study-v1.0.0/studies/wsi-mps-cuda/THIRD_PARTY_NOTICES.md)

The manifest identifies the source commit and file hashes; SHA256SUMS identifies the attachment bytes.
