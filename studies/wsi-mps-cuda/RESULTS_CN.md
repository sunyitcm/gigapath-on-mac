# 结果摘要：四个工程问题

[完整结果与三幅图](RESULTS_EN.md) · [方法与比较方向](METHODS_AND_MEASUREMENT_CONTRACTS_EN.md) · [标量表](DATA_INDEX.md) · [五WSI来源](../../docs/WSI_INPUTS.md)

**RQ1 时间资源。** C8共同合同为FP32/B8、10 discarded warm-up batches、每平台/模型3个独立worker。GigaPath完整输出中位数M5/M3/RTX 5090为827.575/602.494/315.427秒。历史同端点下，90个FP16 M5/M3比值均<1，90个FP32比值均>1；这些是依赖配置，不合并成独立样本。已有C8输出CPU user/system及派生百分比；阶段占比不能证明GPU饱和或具体瓶颈。

**RQ2 数值一致性。** 历史FP32跨Mac完整features逐元素相等：GigaPath 5/5、Virchow2 5/5、CONCH v1.5 0/5；对应15对coords逐行一致。原23-1028和新C8的GigaPath跨CUDA/M5配对中，top约1% tiles分别贡献99.8934%和96.8823%的SSE，但两对使用不同参考会话和独立选取的top集合。历史最终H5生产batch/repeat限制保留。Virchow2官方报告FP16混合精度训练，不能以训练精度直接归因本研究FP32推理差异；最终相等不证明内部计算相同。

**RQ3 聚合传播。** 来源替换与同一完整输入的后端替换分开观察。GigaPath和TITAN同H5的MPS-minus-CUDA relative L2范围分别为1.97452e-7–3.58190e-7及8.68202e-7–1.90768e-6，分母为CUDA参考范数。七来源计数、GigaPath M3F/M5F别名与重复单位集中见[英文结果RQ3](RESULTS_EN.md#rq3--source-replacement-versus-identical-input-backend-replacement)；别名不增加forward，Virchow2学习型聚合不在研究范围。三处C8反方向tile指标不可由相反方向的relative L2替代。

**RQ4 精度策略。** CUDA AMP在15个WSI/模型条件中的整体relative L2较低，但5/15的max_abs更高、6/15的最低tile cosine更低。两张GigaPath、两台Mac的AMP同参考整体relative L2降低52.22%–54.59%，残余top 1% SSE贡献占比更高。直接Mac AMP配对两张均不完全相等；M5-minus-M3的relative L2为0.0009750596166116895和0.0013388880426934756。该降低比例是数值误差指标，不是准确率保留。r1时间/资源并列仅为描述性观察，历史生产时间、会话、系统、适配器及观测差异不支持受控加速结论。

FP32是参考而非数学真值；coords一致不证明RGB/tensor相同；高cosine不证明预测或临床等效；最高SSE不等于病理重要性。不能将tiles或依赖配对当独立患者，也不由这些观察指定硬件或算子根因。
