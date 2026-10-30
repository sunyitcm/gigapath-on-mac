# WSI研究结果与方法摘要包

[English](README.md) · [结果](RESULTS_CN.md) · [方法](METHODS_AND_MEASUREMENT_CONTRACTS_EN.md) · [数据与图](DATA_INDEX.md) · [WSI来源](../../docs/WSI_INPUTS.md) · [原始文献](REFERENCES.md) · [版权与范围](RIGHTS_AND_SCOPE.md) · [第三方说明](THIRD_PARTY_NOTICES.md)

本研究观察固定病理特征工作流在 **M5 Max/Apple MPS、M3 Ultra/Apple MPS、RTX 5090/NVIDIA CUDA** 上的数值一致性与已测资源代价。tile模型为 **GigaPath、Virchow2、CONCH v1.5**，固定学习型聚合路线为兼容GigaPath和TITAN。

- **RQ1 时间与资源：** 平台、精度和batch的观察性排序取决于精度策略与计时边界。
- **RQ2 全局与局部一致性：** 全局接近可与局部误差集中并存，跨Mac完全相等具有模型依赖性。
- **RQ3 差异传播：** 上游来源替换与同一输入的聚合后端替换是两条不同的比较轴。
- **RQ4 精度策略权衡：** 受测条件下AMP的同FP32参考整体误差更低，但局部尾部和内存观察不支持所有指标都改善。

本包定位为 **Results and Methods Snapshot（研究结果与方法摘要包）**，提供结果、必要方法、三幅图和支撑标量；不是完整论文或从WSI开始的端到端复现包。结论限于所述切片、会话及合同，不代表临床等效、普遍确定性或硬件因果机制。

[早期v1.0探索](https://github.com/sunyitcm/gigapath-on-mac/releases/tag/v1.0)采用不同计量合同。源数据、模型权重及完整第三方模型实现不在本包中分发。

[代码与方法补充 v1.1.0](reproducibility/README_CN.md)：脚本、源码组件与摘录、必要配置、记录环境及使用/许可说明。
