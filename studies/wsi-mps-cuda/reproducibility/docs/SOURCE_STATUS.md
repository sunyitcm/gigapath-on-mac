# Source components and historical scope

The [source table](Source_Derivations.csv) records each included component's original small-source identity, public changes and upstream attribution. Source identity describes the retained implementation; it does not by itself identify which historical batch/repeat produced a saved final H5.

| Component | Supplied form and limit |
| --- | --- |
| Historical M5 E9 | Formal 540-record batch source at commit `c63e612245898f7bff788fec27ffc48e0b3e1a75`. Config and sweep text are retained; preprocessing definitions are preserved. Public `encoders.py` and the `run` loader-dispatch line are GigaPath-only derivatives. The public wrapper adds explicit execution, offline/no-fallback and no-overwrite safeguards. |
| C8 common contract | Ordered traversal, stage ledger, event resource sampler and scientific worker body. Platform/session wrappers are not assumed identical. |
| CUDA explicit half / AMP | Separate C8/remaining-four worker excerpts and precision/restore observers. Explicit half includes half parameters/input plus autocast; AMP starts from an independent FP32 model. |
| Mac GigaPath AMP | Separate M5/M3 worker and observer bodies. FP32 parameters/input, MPS FP16 autocast; existing warm-up observations and distinct input-observation scopes retained. |
| CPU comparisons | Historical 512-row comparator; later 256-row FP64 comparator; same-reference, cosine, SSE and summary primitives. These are distinct metric conventions, not one reconstructed historical algorithm. |
| Aggregation | Scientific MPS/CUDA call-site excerpts, monitors, vector metrics and FP64 mean derivative. The supplied blocked-attention component is the CPU/MPS variant. Full model/LongNet/backend integrations remain external. |
| Public helpers | Planner, guarded CPU-comparison CLI and scalar report/plot helper. They are distinct from historical orchestration and the original whole-file plotting script. |

Worker excerpts have no complete dispatch entry and retain names supplied by omitted orchestration/dependency context. They are source components, not standalone workers. No Virchow2/TITAN/CONCH loader is supplied. The archival sweep has a `.py.txt` suffix because its original top-level code dispatches many tasks; it is not a supported command.

M5 E9 batch-level evidence remains established. Historical final-H5 producing batch/repeat, the exact historical M3 executable/synchronization binding and some environment fields remain unavailable. Public derivatives do not fill those historical gaps. Usage and actual validation limits are described in [USAGE](USAGE.md) and [VALIDATION_AND_GAPS](VALIDATION_AND_GAPS.md).
