# Methods and measurement contracts

## Inputs and observation units

The five engineering WSIs and recorded TCGA source filenames are listed in [WSI sources](../../docs/WSI_INPUTS.md). C8 means TCGA-C8-A26W; 23-1028 means TCGA-23-1028. The five families below are separate observations, not a controlled factorial experiment.

| Family | Scope and repetition | Endpoint |
| --- | --- | --- |
| Historical Mac sweep | Two Macs × five WSIs × three models × two precision strategies × six batch sizes × three technical repeats: 1,080 timing records | Encoding and complete-pass endpoints are retained separately. |
| Common-contract C8 | Three platforms × three models × three independent workers; FP32/B8; 10 discarded warm-up batches | Complete-output wall spans the first formal tile loop through final CPU concatenation; loading, warm-up, coordinate generation and H5 close are separate. |
| CUDA precision | Five WSIs × three models × two policies; B8/r1; 10 discarded warm-up batches | Explicit half and AMP use independently restored models; saved FP32 references are cross-session. |
| Seven-source aggregation | Five WSIs; frozen compatible GigaPath or TITAN; MPS/CUDA; two calls per physical input within one model-loading lifecycle | Complete bags and original row order; source replacement and fixed-input backend replacement are separate axes. |
| Mac GigaPath AMP | Two WSIs × two Macs; B8/r1; 10 discarded warm-up batches | FP32 parameters/input with MPS FP16 autocast around model and Original postprocess. |

Warm-up batches contain eight tiles in these B8 contracts and are discarded before output collection. A technical repeat, a call in the same loading lifecycle, an equal-input alias and an independent WSI are different units. Tiles and dependent comparison rows are not independent patients. Three workers describe technical dispersion; r1 observations do not provide repeated performance interventions.

## Workflow and precision

Preprocessing fixes tissue screening, level-zero read windows, pen/low-value rules, transform, coordinate sequence and tile order. Nominal source windows for GigaPath/Virchow2/CONCH v1.5 are 256/224/512 pixels; network inputs are 224/224/448. Coordinate alignment does not establish identical RGB tiles or transformed tensors. Comparisons use entire arrays in original order, without sorting, truncation, intersections or tile removal.

GigaPath uses 1,536-dimensional CLS features, Virchow2 2,560-dimensional CLS plus mean patch features excluding registers, and CONCH v1.5 768-dimensional preprojection features. Frozen GigaPath aggregation uses the compatible blocked FP32 route for the final backend comparison, distinct from native FlashAttention/AMP and hybrid routes. TITAN receives CONCH v1.5 features through its no_proj interface. A common model name does not imply identical attention or precision implementations.

Parameter dtype, submitted input dtype, autocast execution, observed raw/postprocess dtype and storage dtype are distinct. Historical explicit half used half parameters/input plus autocast. AMP restores original FP32 checkpoint parameters independently and submits FP32 input; it is not made by floating a half model. AMP encloses model and Original postprocess consistently across MPS/CUDA. Features are stored as float32, which does not imply FP32-only internal execution. In Mac AMP warm-up observations, representative Conv2d/Linear outputs were FP16 while LayerNorm/raw/post outputs were FP32 and autocast was active; diagnostic hooks were absent during formal output collection. Kernel/accumulator precision was not exhaustively observed.

The Mac AMP sessions differ in OS, adapter packaging and observation overhead. M3 used macOS 15.3, M5 macOS 26.3.1; both recorded Python 3.11.16 and PyTorch 2.13.0. M3 copied the first warm-up input to CPU and observed four layer outputs; M5 observed metadata and six layer inputs/outputs. These differences were not isolated as interventions. Historical final-H5 producer batch/repeat and exact runtime state remain incompletely bound. The M5 formal 540-record sweep has batch-level source evidence from its synchronization fix; this does not identify every retained final H5's producing run. The same synchronization boundary is not established for historical M3. Historical t_encode_s is therefore not treated uniformly as a synchronized GPU interval.

## Reference directions, aliases and metrics

For candidate C and reference R, relative L2 = ||C−R||F / ||R||F. The reference norm is always the denominator. MAE, RMSE, max_abs and original location, global cosine, tile quantiles and the SSE contribution of top ceil(0.01N) rows describe different properties. Zero denominators remain undefined. Metrics are retained CPU FP64 blocked-array results; FP32 is a reference, not mathematical truth. Independently ranked top sets differ across pairs.

The seven source labels are F=saved CUDA FP32, M5F/M3F=Mac FP32, E=CUDA explicit half, P=CUDA AMP, M5E/M3E=historical Mac explicit half. New C8 FP32 references use common-contract sessions; the other four WSI references are earlier functional/historical outputs. [Input relationships](data/Seven_Source_Input_Relationships_105.csv) preserves producer precision/batch/repeat, anonymous session and input identities, and learned-input aliases. Anonymous IDs are references, not distributed feature files or proof of equality by themselves.

GigaPath M3F learned inputs alias M5F on all five WSIs: the four earlier inputs have accepted identical H5 content and geometry; C8 has accepted complete feature and ordered-coordinate equality. Aliases do not add forward calls. Virchow2 learned aggregation is outside this study. Historical M3-minus-M5 FP32 direction differs from later comparisons where indicated; directions in each table are authoritative. Same-input aggregation uses MPS candidate minus CUDA reference with the CUDA norm denominator.

In the Mac same-reference table, A/E/F are *local comparison roles*: AMP, historical explicit half and that Mac's FP32 reference. This F is not the seven-source CUDA F label. A−F and E−F use the same recorded reference input ID in each condition. Reference-container substitutions require prior complete-array equality, not matching names or shapes. Numerical error reduction = 1 − relativeL2(A,F)/relativeL2(E,F), undefined at zero denominator; it is not retained accuracy. A−E alone cannot rank either candidate against F.

## Timing and resource boundaries

For common-contract C8, resources are sampled immediately before traversal, at batch 1, every 50th batch, the last batch, and immediately after traversal. This is event-based sampling without a background thread, not fixed Hz. The event records contain 39, 50 and 13 samples per GigaPath, Virchow2 and CONCH v1.5 worker. The final batch can be partial. Sample maxima need not capture instantaneous or true physical peaks. This schedule is not retroactively assigned to historical or aggregation observations; CUDA precision sampling/reset scope is retained in its resource table.

Output CPU user/system deltas and derived CPU percent are available for C8; 100% denotes one logical CPU. CPU seconds are not wall seconds. Stage shares are sum(matched stage wall)/sum(matched output wall), not sums of stage medians. Host-call intervals include device work and waiting; forward/postprocess/end-sync is not pure GPU busy time. Sampled RSS maxima, lifecycle RSS HWM, MPS allocated/driver counts and CUDA allocated/reserved allocator maxima have distinct scopes. API maxima and sampled maxima are not synonyms for physical peaks. GPU busy time and independent waits were not measured. Low CPU percent does not establish GPU starvation, high forward share does not establish GPU saturation, and unified-memory capacity does not establish zero-copy.

Historical explicit-half producing times cannot be bound to the AMP output containers, and sessions/adapters/OS differ. Time/resource values can be placed alongside precision metrics descriptively, but do not establish a controlled precision speedup. High cosine is not predictive or clinical equivalence; maximum SSE is not pathological importance; equal stored FP32 values do not establish equal intermediate tensors, operator paths or rounding histories.

[Results](RESULTS_EN.md) · [Scalar dictionary](DATA_INDEX.md) · [Primary references](REFERENCES.md)
