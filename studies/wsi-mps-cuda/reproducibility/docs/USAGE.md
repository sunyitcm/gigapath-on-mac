# Commands and execution levels

Run the following from `studies/wsi-mps-cuda/reproducibility` with an already available interpreter. No installer is supplied. [Recorded versions](../environment/Environment_Matrix.csv) are historical observations/gates, not a lockfile demonstrated on a clean system. Missing HDF5/OpenSlide C-library versions remain NOT_RECORDED. A current environment export must not be presented as a historical environment.

```sh
python scripts/plan.py --config configs/c8_common.json
python scripts/compare_h5.py --help
python scripts/compare_h5.py
python scripts/scalar_report.py --help
python scripts/scalar_report.py
```

These commands only report a plan or usage. They do not download, load a model, read arrays, or execute a forward. Research modules were not imported for verification of this supplement.

The only executed numerical validation in this supplement uses previously public **scalar** tables, not feature arrays:

```sh
MPLCONFIGDIR=<WRITABLE_CACHE> python scripts/scalar_report.py   --execute-scalars --plots --output <NEW_SCALAR_OUTPUT>
```

The helper reads an explicit list of CSV/JSON scalar tables under the study data directory. It validates matching-run stage shares, same-reference ratios and reference/physical pair counts, then regenerates three figures plus Figure_Data.json. It refuses to overwrite output. Figures reproduce the table values, not historical PNG bytes. Matplotlib is optional and imported only with `--plots`; missing dependencies are reported rather than installed.

The following describes a possible future CPU array analysis command; it **was not run** for this supplement:

```sh
python scripts/compare_h5.py --execute --reference <REFERENCE_H5>   --candidate <CANDIDATE_H5> --output <NEW_PAIR_OUTPUT>
```

Do not use `-O`: the retained comparator uses scientific assertions. `compare(reference,candidate,...)` preserves candidate−reference, the reference denominator and original row order. Full H5 datasets and coordinates would be read under this command. A wrong reference cannot be repaired by reversing a scalar relative L2 result. The same-reference helper additionally depends on retained precomputed scalar records; its archived body is not a standalone new analysis pipeline.

The M5 E9 single-task entry is supplied as a GigaPath-only public derivative:

```sh
python -m source.historical_m5.extract_mps --help
# Only under a separate execution decision, with licensed, pre-cached models:
python -m source.historical_m5.extract_mps --execute --slide <WSI>   --encoder gigapath --batch 8 --dtype fp32 --tag <UNIQUE_TAG>
```

The entry requires MPS and an empty output namespace, sets offline/no-fallback flags, and refuses overwrite; these wrapper safeguards are new. The `run` derivative changes the model-dispatch line to the retained GigaPath factory; its loop, full-pass warmup, synchronization and output conventions remain historical. It is not the newer C8 ten-batch contract. Heavy dependencies may be imported by this historical module for definitions/transforms; help does not instantiate a model. `encoders.py` is an attributed GigaPath-only loader derivative, not a download utility; only invoke it with offline caches configured. The original sweep is intentionally `.py.txt` because its top-level code dispatches a large sweep; it is not a supported execution entry.

C8/CUDA/Mac AMP and aggregation `worker_excerpts/*.py` are inspectable scientific source, **not runnable commands**. Process dispatch, machine bindings and external model dependencies are not supplied. Their complete source/derivation status is in [SOURCE_STATUS](SOURCE_STATUS.md). Component APIs (`traverse`, `ResourceSampler`, `PrecisionPath`, warmup observers, CPU/MPS `block_attention`, `fixed_mean`, vector metrics) accept the dependencies supplied by a caller. No full public orchestration or end-to-end run has been validated, and no model run is required to review this supplement.


The historical single-task derivative supports only GigaPath; no Virchow2 or CONCH v1.5 loader is supplied. The included blocked adapter rejects CUDA. Configuration entries for other study conditions describe their contracts, not complete executable workers in this package.

## Scalar inputs for the ZIP

The ZIP contains code and documentation, not copies of the previously published scalar tables or figures. In a checkout of the repository, `scalar_report.py` finds the existing study data directory. When using the ZIP alone, obtain the [v1.0.0 Results and Methods Snapshot](https://github.com/sunyitcm/gigapath-on-mac/releases/tag/wsi-mps-cuda-study-v1.0.0) and select its study directory explicitly:

```sh
python scripts/scalar_report.py --execute-scalars --plots   --study-root <V1_0_STUDY_DIRECTORY> --output <NEW_SCALAR_OUTPUT>
```

The selected study directory must contain the published `data/` CSV/JSON files. Missing inputs or libraries are reported; no dependency installer or model downloader is supplied.
