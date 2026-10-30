# Third-party licenses, access and acknowledgments

The snapshot contains numerical observations. The Code and Methods Supplement adds project code and limited attributed example/interface adaptations; it contains no model weights or full third-party model implementations. Official sources below govern their own materials; this repository cannot relicense them. Current page descriptions were checked on **30 October 2026**. A current page does not retrospectively prove the terms accepted for a historical download.

| Model used | Official source and license information | Access and intended-use conditions |
| --- | --- | --- |
| Prov-GigaPath | [Repository](https://github.com/prov-gigapath/prov-gigapath), [Apache-2.0 license](https://github.com/prov-gigapath/prov-gigapath/blob/main/LICENSE), [model card](https://huggingface.co/prov-gigapath/prov-gigapath) | Current repository/model-card labels are Apache-2.0. Separate notices describe research/reproducibility intent, exclude clinical/deployed use from intended scope, and require accepting gated-access conditions. These facts should not be collapsed into a generic “research-use-only license” label. |
| Virchow2 | [Author model card](https://huggingface.co/paige-ai/Virchow2), labeled CC BY-NC-ND 4.0 | Gated individual access, registration and acceptance of terms; academic/noncommercial purpose, attribution and additional use/redistribution conditions. This is Virchow2, not another Virchow version. |
| CONCH v1.5 | [Official model card](https://huggingface.co/MahmoodLab/conchv1_5), labeled CC BY-NC-ND 4.0; [TITAN integration](https://github.com/mahmoodlab/TITAN) | Gated access and acceptance of model-specific terms; academic/noncommercial use, attribution and restrictions on redistribution. Earlier CONCH terms are not used as a substitute. |
| TITAN | [Official repository](https://github.com/mahmoodlab/TITAN), [model card](https://huggingface.co/MahmoodLab/TITAN), labeled CC BY-NC-ND 4.0 | Gated individual access and acceptance of terms; academic/noncommercial use, attribution and model redistribution restrictions. |

The official pages and the terms applicable to the obtained version must be read together; the table is a provenance summary, not a new grant. Historical acquisition-term records are not established by the current labels. Research-content copyright and software MIT scope are described in [rights and scope](RIGHTS_AND_SCOPE.md).

We acknowledge **The Cancer Genome Atlas (TCGA) Research Network**, its participants and investigators, and the **NCI Genomic Data Commons (GDC)** teams for the source data and access infrastructure. [TCGA](https://www.cancer.gov/tcga) · [GDC](https://gdc.cancer.gov/) · [GDC policies](https://gdc.cancer.gov/about-gdc/gdc-policies). The [WSI source table](../../docs/WSI_INPUTS.md) lists recorded identifiers. Source data are not redistributed; their applicable access, use and attribution terms remain separate from this repository's software license.

## Code supplement attributions

The GigaPath loading/transform example adapts the official README at revision `55431e04ff853a08f752a0ba42e6e9c48e60c776`, Apache-2.0; [complete license](reproducibility/licenses/Apache-2.0-GigaPath.txt). The legacy TorchScale interface retains **Copyright (c) 2023 Microsoft**, MIT. The [source table](reproducibility/docs/Source_Derivations.csv) and [license scope](reproducibility/docs/LICENSE_SCOPE.md) record the included files/functions and changes. Full upstream model implementations and weights remain external.
