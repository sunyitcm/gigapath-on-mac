# Scalar tables and figures

[Methods](METHODS_AND_MEASUREMENT_CONTRACTS_EN.md) defines units, precision policies, timing/resource scope, reference direction and aliases. Tables contain measurements and derived scalar comparisons, not feature arrays. Anonymous input/session/pair IDs preserve relationships without exposing asset locations. Missing or undefined values must not be treated as zero. No sorting or re-pairing of tiles is implied.

| Table | Scientific scope |
| --- | --- |
| [C8_Output_Times_27.csv](data/C8_Output_Times_27.csv) | Per-worker output and separate load/migration/warm-up/save endpoints; three workers per platform/model. |
| [C8_Output_Stage_Raw_27.csv](data/C8_Output_Stage_Raw_27.csv) | Matched worker stage walls; host-call intervals include work and waits. |
| [C8_Output_Resources_27.csv](data/C8_Output_Resources_27.csv) | Output CPU seconds/percent, sampled RSS, lifecycle HWM and device API counts; blank/nonmeasured fields are not zero. |
| [C8_Stage_Shares_9.csv](data/C8_Stage_Shares_9.csv) | Stage shares with the recorded matched-output denominator. |
| [C8_Time_Resource_Summary_9.csv](data/C8_Time_Resource_Summary_9.csv) | Actual r1/r2/r3 output seconds and descriptive time/resource summaries. |
| [Historical_Performance_Paired_Ratios_144.csv](data/Historical_Performance_Paired_Ratios_144.csv) | Historical ratio definitions, counts and descriptive spread; dependent WSI/configuration units. |
| [Historical_Performance_Strata_72.csv](data/Historical_Performance_Strata_72.csv) | Historical timing/resource strata with the original endpoint and recorded memory semantics. |
| [Historical_Technical_Repeat_IQR_360.csv](data/Historical_Technical_Repeat_IQR_360.csv) | Three-repeat dispersion per Mac/WSI/model/precision/batch. |
| [Historical_H5_Pair_Metrics.csv](data/Historical_H5_Pair_Metrics.csv) | Historical complete-feature equality and global/tile metrics with the original reference labels. |
| [C8_Comparisons_9.json](data/C8_Comparisons_9.json) | C8 cross-platform comparisons with explicit reference platform and input IDs. |
| [C8_Repeats_27.json](data/C8_Repeats_27.json) | Three dependent repeat-pair checks per C8 platform/model; unrecorded fields remain unrecorded. |
| [Seven_Source_Coverage_Overview_15.csv](data/Seven_Source_Coverage_Overview_15.csv) | WSI/model/source coverage, EXEC/ALIAS/N/A; x4 means two backends × two calls, not four independent inputs. |
| [Seven_Source_Input_Relationships_105.csv](data/Seven_Source_Input_Relationships_105.csv) | Scientific input/session IDs, producer precision/batch/repeat/warm-up and exact learned-input alias basis. |
| [Same_Input_Repeats_140.json](data/Same_Input_Repeats_140.json) | Same-lifecycle r2-minus-r1 metrics; anonymous physical-pair IDs identify duplicate alias references. |
| [Same_H5_MPS_minus_CUDA_70.json](data/Same_H5_MPS_minus_CUDA_70.json) | Same-complete-input MPS-minus-CUDA metrics; CUDA denominator and alias/physical-pair grouping. |
| [Same_Backend_Source_Pairs_180.json](data/Same_Backend_Source_Pairs_180.json) | Fixed-backend source replacement, original candidate/reference denominator and vector metrics. |
| [Propagation_135.csv](data/Propagation_135.csv) | Parallel tile/fixed-mean/learned metrics. Availability fields distinguish N/A, unrecorded and missing-direction metrics. |
| [Directional_Tile_Field_Gaps_3.json](data/Directional_Tile_Field_Gaps_3.json) | C8 M3F-minus-M5F tile fields unavailable in this direction; not a gap in learned-aggregation coverage. |
| [SSE_Selected_Cutoffs_18.json](data/SSE_Selected_Cutoffs_18.json) | Independent top-SSE cutoffs, counts, totals and contributions for the named reference pairs. |
| [CUDA_Same_Reference_Policy_15.csv](data/CUDA_Same_Reference_Policy_15.csv) | Explicit-half/AMP relative L2 and retained local-direction flags against one CUDA FP32 reference. |
| [Core5_Tile_And_Mean_75.json](data/Core5_Tile_And_Mean_75.json) | CUDA precision pair metrics, norms, max location and SSE tails; original reference direction retained. |
| [Core5_CUDA_Timing_Resources_30.csv](data/Core5_CUDA_Timing_Resources_30.csv) | CUDA r1 timing/resources and numeric stages. Worker/output CPU scopes, event rule and allocator reset boundary are separate. |
| [Mac_AMP_Same_Reference_4.json](data/Mac_AMP_Same_Reference_4.json) | A-minus-F and E-minus-the-same-F metrics, tile distributions, norms and SSE tails; F is per-Mac FP32. |
| [Mac_AMP_Direct_CrossMac_2.json](data/Mac_AMP_Direct_CrossMac_2.json) | Direct M5 AMP candidate minus M3 AMP reference, complete equality, tile quantiles and SSE tail. |
| [Mac_AMP_Time_Resources_4.csv](data/Mac_AMP_Time_Resources_4.csv) | Four r1 AMP worker observations; explicit/reference producer timing is unbound to these sessions. |

Figure S1: [output time and matched stage composition](figures/Figure_3.png), supported by the C8 time/stage tables. Figure S2: [source and backend replacement](figures/Figure_5.png), supported by Propagation_135 and the backend/source-pair tables. Figure S3: [same-reference precision/resource observations](figures/Figure_6.png), supported by CUDA_Same_Reference_Policy_15, Mac_AMP_Same_Reference_4 and Core5_CUDA_Timing_Resources_30.

PNG files retain the plotted pixels; text/EXIF/time metadata are not part of the scientific figures. Equal/zero results remain in the numerical tables and are not replaced with positive pseudocounts. API and sampled resource maxima have different definitions.
