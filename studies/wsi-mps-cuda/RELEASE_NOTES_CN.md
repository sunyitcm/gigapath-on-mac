# 三平台病理基础模型工程评估：Apple MPS与NVIDIA CUDA — Results and Methods Snapshot

Tag: `wsi-mps-cuda-study-v1.0.0`

本包为**研究结果与方法摘要包**，观察M5 Max/Apple MPS、M3 Ultra/Apple MPS、RTX 5090/NVIDIA CUDA上的固定特征工作流。tile模型为GigaPath、Virchow2、CONCH v1.5，学习型聚合为兼容GigaPath与TITAN。

四RQ讨论时间资源、整体与局部数值一致性、来源替换与同输入聚合后端替换，以及显式半精度和AMP的同FP32参考权衡。观察显示计时排序与精度策略有关，跨Mac相等具有模型依赖性，局部误差可高度集中。受测条件下AMP整体误差较低，但不能概括为所有指标改善或临床等效。

包内提供精简报告、方法、三幅图、支撑标量表、WSI及模型／文献出处和校验清单。不含原始WSI、特征数组、权重或第三方实现，不代表完整论文或端到端复现包。会话差异、参考方向、依赖观察单位、r1及事件采样限制在方法中说明。

旧 **v1.0 Release发表于2026年8月16日**，属于早期GigaPath-on-Mac探索，其计量合同及NVIDIA估计数据与本次研究不同。旧日期只适用于当时公开内容，不代表本次新增结果当时已公开；不作首次性声明。

[结果](https://github.com/sunyitcm/gigapath-on-mac/blob/wsi-mps-cuda-study-v1.0.0/studies/wsi-mps-cuda/RESULTS_CN.md) · [方法](https://github.com/sunyitcm/gigapath-on-mac/blob/wsi-mps-cuda-study-v1.0.0/studies/wsi-mps-cuda/METHODS_AND_MEASUREMENT_CONTRACTS_EN.md) · [标量表](https://github.com/sunyitcm/gigapath-on-mac/blob/wsi-mps-cuda-study-v1.0.0/studies/wsi-mps-cuda/DATA_INDEX.md) · [版权范围](https://github.com/sunyitcm/gigapath-on-mac/blob/wsi-mps-cuda-study-v1.0.0/studies/wsi-mps-cuda/RIGHTS_AND_SCOPE.md) · [第三方说明](https://github.com/sunyitcm/gigapath-on-mac/blob/wsi-mps-cuda-study-v1.0.0/studies/wsi-mps-cuda/THIRD_PARTY_NOTICES.md)

manifest标识来源commit及文件hash；SHA256SUMS标识附件字节。
