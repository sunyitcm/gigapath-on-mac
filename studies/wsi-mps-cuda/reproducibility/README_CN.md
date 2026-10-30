# 代码与方法补充 v1.1.0

[English](README.md) · [使用](docs/USAGE.md) · [源码组件](docs/SOURCE_STATUS.md) · [合同](docs/CONTRACTS.md) · [环境](environment/Environment_Matrix.csv) · [许可](docs/LICENSE_SCOPE.md)

本补充为[WSI MPS–CUDA结果与方法摘要包 v1.0.0](https://github.com/sunyitcm/gigapath-on-mac/releases/tag/wsi-mps-cuda-study-v1.0.0)提供预处理、计时/资源采样、精度观测、CPU比较、聚合调用源码摘录及公开标量制表/制图工具，并附M5 Max/MPS、M3 Ultra/MPS、RTX 5090/CUDA研究的配置和已记录环境。

- `scripts/plan.py`：配置计划工具，不调度模型。
- `scripts/compare_h5.py`：需显式启用的CPU比较入口；本补充未执行数组比较。
- `scripts/scalar_report.py`：从既有公开CSV/JSON生成报告和图。
- `source/historical_m5/`：仅GigaPath单任务整理版、保留的预处理及归档sweep文本。
- `source/worker_excerpts/`：省略调度及依赖上下文的科学源码摘录，**不能独立执行**。所附分块attention组件仅接受CPU/MPS。

详见[RQ—源码映射](docs/RQ_SOURCE_MAP.md)、[输入与模型](docs/INPUTS_MODELS.md)及[验证限制](docs/VALIDATION_AND_GAPS.md)。端到端执行和全新环境安装未验证。M5 E9批次级源码证据不确定每份历史最终H5的生产batch/repeat，也不恢复历史M3精确执行源码。

既有标量结果和原三幅图保持不变，未增加实验。不含模型权重、WSI或特征数组。ZIP仅含代码和必要说明；标量输入需按使用说明另行取得。项目软件和第三方片段分别遵循[所列许可范围](docs/LICENSE_SCOPE.md)。
