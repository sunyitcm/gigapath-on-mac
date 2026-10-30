# Code attribution and license scope

The [repository MIT license](../../../../LICENSE) covers project-specific software and accompanying software documentation. Attributed third-party portions retain their own terms. It does not relicense models, weights, source data, research summaries, figures or scalar compilations; see [rights and scope](https://github.com/sunyitcm/gigapath-on-mac/blob/0fe0b1f408c51580b61be9b2ab826ceedffa175a/studies/wsi-mps-cuda/RIGHTS_AND_SCOPE.md).

The [source table](Source_Derivations.csv) identifies included files/functions, source identities, upstream URLs/revisions, changes and license scopes. Upstream references identify the cited example or interface; they are not necessarily the historical model checkout. Recorded model pins and environments are supplied separately.

| Included portion | Source and license | Adaptation |
| --- | --- | --- |
| GigaPath loading/transform example in `source/historical_m5/encoders.py` | Official [GigaPath README](https://github.com/prov-gigapath/prov-gigapath/blob/55431e04ff853a08f752a0ba42e6e9c48e60c776/README.md), Apache-2.0; [complete license](../licenses/Apache-2.0-GigaPath.txt) | Registry/lambda assembly, explicit `img_size=224`, shared normalization and `eval()`. GigaPath-only; no full model implementation. |
| Legacy `flash_attn_func` interface in `source/block_attention.py` | [GigaPath-vendored TorchScale interface](https://github.com/prov-gigapath/prov-gigapath/blob/55431e04ff853a08f752a0ba42e6e9c48e60c776/gigapath/torchscale/component/flash_attention.py), Copyright (c) 2023 Microsoft, MIT; notice retained in the file | Retained interface around the project FP32 blocked softmax/LSE operation; CPU/MPS only. This is not a native FlashAttention kernel or the full LongNet integration. |
| Project preprocessing, measurement, observers, comparison, configuration and helpers | Repository MIT for project-specific portions | External library APIs are called; their implementations remain dependencies. Aggregation excerpts call externally obtained model APIs. |

The blocked recurrence is related to the [FlashAttention algorithm](https://arxiv.org/abs/2205.14135); the native FlashAttention implementation is not supplied. Full model classes, LongNet integration, weights and gated model code are external dependencies. No Virchow2 or TITAN/CONCH loader implementation is included; the historical single-task entry is GigaPath-only.

Official model sources and access terms remain separate: [GigaPath](https://huggingface.co/prov-gigapath/prov-gigapath) (Apache-2.0, with model-card access/intended-use conditions), [Virchow2](https://huggingface.co/paige-ai/Virchow2), [CONCH v1.5](https://huggingface.co/MahmoodLab/conchv1_5), and [TITAN](https://huggingface.co/MahmoodLab/TITAN) (the latter three labeled CC BY-NC-ND 4.0 with gated conditions). Users obtain dependencies under their applicable terms. These notices grant no permission to redistribute external weights or restricted implementations.

TCGA/NCI GDC sources are acknowledged in the [WSI table](https://github.com/sunyitcm/gigapath-on-mac/blob/0fe0b1f408c51580b61be9b2ab826ceedffa175a/docs/WSI_INPUTS.md) and [third-party notices](https://github.com/sunyitcm/gigapath-on-mac/blob/0fe0b1f408c51580b61be9b2ab826ceedffa175a/studies/wsi-mps-cuda/THIRD_PARTY_NOTICES.md). Source slides are not distributed.
