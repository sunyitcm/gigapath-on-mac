# Mac 本地部署 GigaPath 与 WSI 特征提取实战手册

> 环境：Mac Intel / 16GB 内存 / mambaforge / Python 3.11 / CPU-only
> 目标模型：prov-gigapath/prov-gigapath（病理 WSI 基础模型，tile encoder ViT-G/14 + LongNet slide encoder）
> 最终成果：从一张 40× 免疫组化穿刺活检 WSI（.svs）自动定位组织区域、按 0.5 MPP 提取 patch、批量编码为 1536 维特征并聚合为 slide 级向量，全链路 CPU 可复现。

---

## 一、背景与总体技术路线

GigaPath（Nature 2024，Provident/微软）是病理全切片图像（WSI）的基础模型，分两级：

1. **Tile Encoder**：ViT-G/14，将 224×224 patch（0.5 MPP，即 20× 等效放大）编码为 1536 维向量；
2. **Slide Encoder**：LongNet（12 层 dilated attention），将整张切片全部 tile 特征聚合为 slide 级表示。

官方 `environment.yaml` 硬编码 `pytorch-cuda=11.8` 等 Linux/CUDA 依赖，**在 Mac 上不可用**。正确路线是弃用官方 yaml，手工构建纯净环境：

```
brew/miniforge 已具备
  → conda 创建独立 env（python=3.11）
  → pip 安装 CPU 版 PyTorch（绝不带 CUDA）
  → 手工安装 timm / huggingface_hub / openslide-python / scikit-image / h5py
  → HF_TOKEN 鉴权下载 gated 权重（~/.cache/huggingface/hub/）
  → OpenSlide 组织检测 → patch 提取（MPP 匹配）→ timm 编码 → h5 缓存 → 聚合
```

---

## 二、模型仓库的真实文件结构（重要认知基线）

通过官方 API（`https://huggingface.co/api/models/prov-gigapath/prov-gigapath`）的 `siblings` 字段确认，该仓库**有且仅有**以下文件：

| 文件 | 大小 | 作用 |
|---|---|---|
| `config.json` | ~829 B | tile encoder 结构配置 |
| `pytorch_model.bin` | ~4.2 GB | tile encoder 权重（**不存在 model.safetensors**） |
| `slide_encoder.pth` | ~345 MB | LongNet slide encoder 权重 |
| `README.md` / `.gitattributes` | KB 级 | 元信息 |
| `sample_data/PROV-000-000001.ndpi` | GB 级 | 示例切片（可选，非必需） |

该仓库为 **gated（auto）模型**：必须先在 HF 网站同意协议，下载时携带 token。

**关键教训：报错信息中出现的文件名 ≠ 仓库中真实存在的文件。** 新版 timm/huggingface_hub 会优先探测 `.safetensors`，其报错容易误导人"补一个 safetensors 文件"。任何修复前，先查 `siblings` 列表。

---

## 三、HuggingFace 缓存机制（理解后才能正确修复）

标准缓存布局：

```
~/.cache/huggingface/hub/models--prov-gigapath--prov-gigapath/
├── blobs/                  # 真实文件，按内容SHA命名（4.2G权重在此）
├── refs/main               # 记录当前commit号
└── snapshots/<commit>/     # 软链接目录，链接指向 ../../blobs/<sha>
```

要点：

- `snapshots/` 下的软链接必须是 **`../../blobs/`（两级）**，写 `../blobs/` 即断链；
- blob 按内容哈希寻址，`snapshot_download` 会校验本地 blob，完整则跳过——**只要 blobs 没坏，修复永远不会重下大文件**；
- 缓存修复的**唯一正确姿势**：`rm -rf snapshots/<commit>` 后让 `snapshot_download()` 重建；**绝不手工创建/修改软链接**；
- 文件下齐后可用 `HF_HUB_OFFLINE=1` 完全离线运行，官方支持。

---

## 四、踩坑实录与根因分析（按时间线）

### 坑 1：手工"修复"软链接，越修越坏

**现象**：某次加载报错提及 safetensors，手工创建了 `model.safetensors`、`model.safetensors.index.json` 链接。
**三处错误**：
1. 无中生有——该仓库根本没有 safetensors 文件；
2. 链接路径少一层（`../blobs/` 应为 `../../blobs/`）→ 断链；
3. `index.json` 指向了 config.json 的 blob（829B），内容张冠李戴。

**根因**：未先核对仓库真实文件列表，望文生义地"满足"报错。
**避免**：修复前先查 API 的 `siblings`；cache 问题只删 snapshots 重建，不动手。

### 坑 2：`HF_HUB_OFFLINE=1` 幽灵变量

**现象**：`LocalEntryNotFoundError ... outgoing traffic has been disabled`。
**根因**：环境变量 `HF_HUB_OFFLINE=1` 不知何时被设置，禁止一切联网；而 snapshots 恰好被删，离线又找不到缓存，两头落空。
**避免**：持久化变量只用 `conda env config vars set` 并知晓其存在。

### 坑 3：`huggingface-cli` / `hf` 命令版本错位

**现象**：`huggingface-cli` 报错路径为 `/Library/Frameworks/Python.framework/3.11`（系统 Python 旧版）；而环境内新版 huggingface_hub 已删除 `huggingface_hub.commands` 模块，`hf` 命令又不存在。
**根因**：PATH 中系统 Python 的 CLI 与 conda 环境内的库版本不一致。
**避免**：**不依赖任何 CLI 命令**，统一直接调用库函数：

```bash
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('prov-gigapath/prov-gigapath'))"
```

### 坑 4：token 的 shell 语法

**错误写法**：`--token $hf_xxxxxxx`（`$` 会把真实 token 当变量名解析为空）。
**正确写法**：

```bash
export HF_TOKEN=hf_xxxxxxx          # 先存变量
# 之后 -H "Authorization: Bearer $HF_TOKEN" 或代码自动读取
```

### 坑 5：pip 装包错位

**现象**：`which pip` 显示环境路径正确，但包装到了 `/Library/Frameworks/...3.11`（系统 Python）。
**铁律**：**永远使用 `python -m pip install 包名`**——`python` 指向哪个解释器，包装到哪里，零歧义。同理，验证也用 `python -c "import xxx"`。

### 坑 6：timm 每次加载都联网检查

**现象**：本地文件齐全，运行脚本仍卡在 HEAD 请求超时。
**解法**：`HF_HUB_OFFLINE=1 python script.py`；或一次性持久化 `conda env config vars set HF_HUB_OFFLINE=1 -n gigapath`（需重新激活生效）。

---

## 五、WSI 特征提取 Pipeline 的技术要点

### 5.1 组织区域检测（避开空白）

低倍缩略图（1024×1024）→ 灰度化 → **Otsu 自适应阈值** → 组织掩膜。穿刺活检样本组织占比可能仅 4%，全局 60% 占比的粗判会一无所获，需要小窗口 + 较低阈值（20%）。

### 5.2 缩略图坐标映射陷阱（本次实战 bug）

缩略图 224×224 像素的窗口对应原图（24296×26654）约 5800×5800 像素的巨大区域。若"大窗口判组织 + 取窗口左上角 tile"，tile 会落在组织旁边的空白处——可视化验证时方框全部偏出组织带。

**正确做法**：缩小判定窗口（缩略图 56×56，约原图 1456 像素）、窗口重叠步进（步长 28）、**以窗口中心为 tile 中心**：

```python
cx = int((tx + WIN/2) * scale_x); cy = int((ty + WIN/2) * scale_y)
x, y = max(cx - READ//2, 0), max(cy - READ//2, 0)
```

### 5.3 MPP 匹配（极易忽略的物理尺度问题）

GigaPath 训练输入为 **0.5 MPP（20×）下的 224×224**。40× 扫描（MPP≈0.22）必须读取约 510×510 像素再 LANCZOS 缩放到 224：

```python
READ = round(224 * 0.5 / float(slide.properties['openslide.mpp-x']))  # ≈510
tile = slide.read_region((x, y), 0, (READ, READ)).convert("RGB").resize((224, 224), Image.LANCZOS)
```

不做 MPP 匹配，特征的物理视场与训练分布偏离，下游性能打折。

### 5.4 Intel CPU + 16GB 的资源控制

- BATCH=8 小批编码，峰值内存 3–4 GB；
- 逐张处理、逐张落盘，已生成的 h5 自动跳过（断点续跑）；
- 单 tile 约 1–3 s，小活检切片（~100 tile）全程 10–30 min；
- 自动化死机的根因通常是全量读入内存，务必流式 flush。

### 5.5 h5 文件结构（一切片一文件，CLAM 惯例）

```
slide.h5
├── features  (N, 1536) float32   # N个patch的GigaPath特征
└── coords    (N, 2)    int       # 每个patch在原图的坐标
```

### 5.6 Slide 级聚合

- **基线**：mean pooling → (1536,) slide 向量，零依赖，论文常用 baseline；
- **进阶**：官方 LongNet `slide_encoder.pth`（输出 768 维，需 prov-gigapath 仓库代码 + torchscale）或 ABMIL；
- slide 向量本身不是诊断，需标签数据训练分类器（LR/SVM/MLP）才有临床输出。

---

## 六、日常使用速查表

```bash
# 每次开终端
conda activate gigapath
cd /Users/<you>/projects/wsi_gigapath

# 运行任何脚本（模型已缓存，强制离线）
HF_HUB_OFFLINE=1 python run_slide_mps.py

# 查看h5
python -c "import h5py; h=h5py.File('slide.h5','r'); print(h['features'].shape, h['coords'].shape)"

# 装包铁律
python -m pip install <包名>
```

---

## 七、迁移到 Apple Silicon（M5 Max）+ MPS 的展望

当前 Intel 路线全部代码**无需修改即可在 M 系列运行**，迁移要点：

1. **环境重建**：Miniforge arm64 版，conda 建 env 后 `python -c "import platform; print(platform.machine())"` 必须输出 `arm64`；`pip install torch` 自动获得 MPS 后端；
2. **设备分流策略**：
   - tile encoding（大批量前向）→ `mps`，比 Intel CPU 快一个数量级；
   - slide encoder / MIL 聚合（小矩阵、含 index 操作）→ 留 `cpu`，规避 MPS 的 4GB 临时缓冲（`MPSTemporaryNDArray > 2^32`）与算子 fallback 风险；
3. **不要在 MPS 上使用 `torch.compile`**（CUDA 专属优化，MPS 上崩溃或静默失败）；
4. **内存红利**：M5 Max 统一内存可将 BATCH 提至 64–128，整切片的 tile 特征可全量驻留内存；全片 WSI 提取时间从几十分钟降到分钟级；
5. **OpenSlide**：ARM 上 `brew install openslide` + `pip install openslide-python`，或直接 `openslide-bin`；若报 dylib 缺失用 `conda env config vars set DYLD_LIBRARY_PATH=/opt/homebrew/lib`（注意 SIP 可能清空该变量，`openslide-bin` 更省事）；
6. **权重迁移**：直接拷贝 `~/.cache/huggingface/hub/models--prov-gigapath--prov-gigapath/` 整个目录到新机器同路径，即可离线复用，无需重新下载；
7. **xformers / flash-attn / fairscale 依旧不装**（无 Mac 官方 wheel）；LongNet 推理走纯 PyTorch 路径。

---

## 八、经验法则（带走这六条）

1. **报错里的文件名 ≠ 仓库真实文件**——动手前先查 API `siblings`；
2. **HF cache 只删 snapshots 重建，绝不手工补软链接**；blobs 在大文件就在；
3. **装包用 `python -m pip`，下载用 `python -c` 调库函数**——杜绝 CLI 版本错位；
4. **离线用 `HF_HUB_OFFLINE=1`**；
5. **WSI 取 tile 两件事必须先验证**：组织掩膜可视化 + MPP 匹配；
6. **一张切片打通全链路，再批量**——聚合几秒、提取几十分钟，顺序反了就是几小时的浪费。
