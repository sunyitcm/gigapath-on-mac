# WSI identifiers and sources

C8 means **TCGA-C8-A26W**; 23-1028 means **TCGA-23-1028**. The five engineering WSIs retain their recorded TCGA identifiers and source filenames.

| WSI identifier | Recorded project | Full source filename |
| --- | --- | --- |
| TCGA-23-1028 | OV | `TCGA-23-1028-01Z-00-DX1.117B4B7B-F796-4D33-A645-CD80E5C43E6D.svs` |
| TCGA-B5-A0K3 | UCEC | `TCGA-B5-A0K3-01Z-00-DX1.ACC15A3D-67F0-49C8-9B41-18A2D4EBF5C8.svs` |
| TCGA-C8-A26W | BRCA | `TCGA-C8-A26W-01Z-00-DX1.CFF07941-6CD1-4CF9-BE5F-387DA67B66CC.svs` |
| TCGA-CD-A4MG | STAD | `TCGA-CD-A4MG-01Z-00-DX1.CFC4618E-DFE9-4AAA-AD42-E1CC97374020.svs` |
| TCGA-DM-A1HA | COAD | `TCGA-DM-A1HA-01Z-00-DX1.E56FC26A-DDB9-4121-9E79-5009FB23CCEB.svs` |

The recorded acquisition source is TCGA through the NCI GDC using gdc-client. The engineering panel was selected by recorded tissue fraction nearest 0.5 after panel QC. Project labels are recorded metadata, not diagnoses inferred from barcodes. UUID-like filename suffixes are not asserted to be separately verified GDC API accessions. Earlier and common-contract C8 sessions use the same named WSI but are not interchangeable timing runs.

These five WSIs are not a representative clinical cohort. Complete candidate-pool history, clinical covariates and patient independence are not established by this table. No WSI data are distributed. We acknowledge TCGA participants, investigators and the NCI GDC teams. [TCGA](https://www.cancer.gov/tcga) · [GDC](https://gdc.cancer.gov/) · [Data/use notices](../studies/wsi-mps-cuda/THIRD_PARTY_NOTICES.md).

[Study](../studies/wsi-mps-cuda/README.md) · [Methods](../studies/wsi-mps-cuda/METHODS_AND_MEASUREMENT_CONTRACTS_EN.md) · [Results](../studies/wsi-mps-cuda/RESULTS_EN.md)
