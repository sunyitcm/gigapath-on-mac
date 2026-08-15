# Mac M5 Max（128GB）GigaPath 安装与 MPS 加速指南

> 适用：Apple Silicon M5 Max / 128GB 统一内存 / macOS 15+
> 与 Intel 版的关系：代码零修改可迁移，本文给出从头安装的完整命令 + MPS 加速改造 + 已训练缓存的直接迁移方法。

---

## 第 0 步：架构确认（一次性）

```bash
uname -m    # 必须输出 arm64
arch        # 必须输出 arm64
```

若输出 `x86_64`，说明终端跑在 Rosetta 下，关掉换原生终端再继续。

---

## 第 1 步：系统层准备

```bash
# Homebrew 未装则装（已装跳过）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# OpenSlide 底层动态库（M系列路径 /opt/homebrew）
brew install openslide

# Miniforge（arm64 原生 conda）
brew install --cask miniforge
conda init zsh
```

重开终端使 conda 生效。

---

## 第 2 步：创建环境（与 Intel 版同名，便于习惯迁移）

```bash
conda create -n gigapath python=3.11 -y
conda activate gigapath
python -c "import platform; print(platform.machine())"   # 必须 arm64
```

---

## 第 3 步：PyTorch（自动获得 MPS 后端，绝不带 CUDA）

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio
```

三重验证：

```bash
python - <<'PY'
import torch, platform
print("架构:", platform.machine())                    # arm64
print("Torch:", torch.__version__)                    # 2.x
print("MPS built:", torch.backends.mps.is_built())    # True
print("MPS avail:", torch.backends.mps.is_available())# True
PY
```

四项全对才继续。

---

## 第 4 步：项目依赖（铁律：一律 python -m pip）

```bash
python -m pip install timm huggingface_hub einops pillow scikit-learn numpy pandas tqdm matplotlib h5py scikit-image openslide-python
```

验证：

```bash
python -c "import openslide, torch, timm, h5py, skimage; print('全部OK')"
```

若 openslide 报找不到 dylib：

```bash
conda env config vars set DYLD_LIBRARY_PATH=/opt/homebrew/lib -n gigapath
conda deactivate && conda activate gigapath
```

仍不行就换纯 Python 方案：`python -m pip install openslide-bin`（自带 dylib，不受 SIP 影响）。

**明确不装**（Mac 无官方 wheel / ABI 冲突）：`flash-attn`、`xformers`、`fairscale`、`pytorch-cuda`。

---

## 第 5 步：模型权重（二选一）

### 方案 A：从 Intel Mac 直接拷贝（推荐，零下载）

在 Intel 机器上打包缓存：

```bash
cd ~/.cache/huggingface/hub
tar -czf ~/Desktop/gigapath_cache.tar.gz models--prov-gigapath--prov-gigapath
```

把 `gigapath_cache.tar.gz`（约 4.6GB）用 AirDrop/移动硬盘拷到 M5 Max，解到同路径：

```bash
cd ~/.cache/huggingface/hub     # 没有则 mkdir -p 后进入
tar -xzf ~/Desktop/gigapath_cache.tar.gz
```

> 注意：tar 会保留软链接结构（snapshots → ../../blobs），这正是我们需要的；这也是"缓存迁移"比"重新下载"更稳的原因。

---

## 第 6 步：冒烟测试（离线模式，三层验证）

**A. Tile encoder（MPS 前向）**

```bash
HF_HUB_OFFLINE=1 python - <<'PY'
import torch, timm
device = "mps"
m = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True).to(device).eval()
x = torch.randn(1, 3, 224, 224).to(device)
with torch.no_grad():
    y = m(x)
print("Tile encoder OK:", tuple(y.shape), y.device)   # 期望 (1, 1536) mps:0
PY
```

**B. OpenSlide**

```bash
python -c "import openslide; print('openslide OK')"
```

**C. MPS 压测（确认 batch 上限）**

```bash
HF_HUB_OFFLINE=1 python - <<'PY'
import torch, timm, time
m = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True).to("mps").eval()
for bs in [16, 32, 64, 128]:
    x = torch.randn(bs, 3, 224, 224).to("mps")
    t0 = time.time()
    with torch.no_grad():
        m(x)
    torch.mps.synchronize()
    print(f"batch={bs}: {time.time()-t0:.2f}s")
PY
```

若某个 batch 报 `MPSTemporaryNDArray` 或 Abort，就用前一档作为生产 BATCH。128GB 内存下通常 64–128 无压力。

---

## 第 7 步：Intel 版代码迁移（只需改 2 处）

把 `run_slide_mps.py` / `batch_extract.py` 拷贝到 M5 Max 相同目录结构，修改：

**改动 1：设备与批次**

```python
device = "mps"        # Intel 版是 "cpu"
BATCH   = 64          # Intel 版是 8；以第6步压测结果为准
```

**改动 2：编码时同步（计时更准，可选）**

```python
with torch.no_grad():
    out = model(t.to(device))
torch.mps.synchronize()
feats.append(out.cpu().numpy())
```

其余逻辑（Otsu 组织检测、WIN=56/步长 28/阈值 0.2、MPP 匹配 READ=510、h5 结构）**原样保留**。

---

## 第 8 步：日常使用规则（与 Intel 版一致）

```bash
conda activate gigapath
cd /Users/<you>/projects/wsi_gigapath
HF_HUB_OFFLINE=1 python batch_extract.py      # 运行一律离线
```
- 装包永远 `python -m pip install`；
- 需要时持久化离线模式：`conda env config vars set HF_HUB_OFFLINE=1 -n gigapath`（重新激活生效）。

---

## 第 9 步：MPS 专属避坑清单

| 坑 | 现象 | 对策 |
|---|---|---|
| 4GB 临时缓冲 | `Abort trap: 6` / `MPSTemporaryNDArray > 2**32` | 降 BATCH；或该步切 CPU；或 tensor 转 float16 |
| `torch.compile` | 崩溃/静默失败 | MPS 上禁用 |
| 算子 fallback | 越用越慢 | 不开 `PYTORCH_ENABLE_MPS_FALLBACK`；slide 聚合直接放 CPU |
| 设备分流 | — | tile 编码 → MPS；LongNet/ABMIL 聚合 → CPU（数据小、含 index 操作） |
| 精度疑虑 | — | MPS 前向与 CPU 差异 ~1e-3，正常；训练才需要警惕 |

---

## 第 10 步：性能预期（相对 Intel 版）

| 环节 | Intel 16G CPU | M5 Max 128G MPS |
|---|---|---|
| 单 tile 编码 | 1–3 s | ~5–20 ms |
| 小活检 WSI（~100 tile） | 10–30 min | <1 min |
| 大 WSI（~5000 tile） | 数小时 | 5–15 min |
| BATCH | 8 | 64–128 |
| 内存策略 | 流式 flush 必须 | 可整片驻留内存 |

---

## 附：完整最小命令串（空机到跑通）

```bash
uname -m
brew install openslide
conda create -n gigapath python=3.11 -y
conda activate gigapath   # activate 需单独一行（&& 串联时 shell hook 可能不生效）
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio
python -m pip install timm huggingface_hub einops pillow scikit-learn numpy pandas tqdm matplotlib h5py scikit-image openslide-python
python -c "import torch; print(torch.backends.mps.is_available())"   # True

# 权重：从Intel机拷贝缓存目录，或下载
export HF_TOKEN=hf_你的token
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('prov-gigapath/prov-gigapath'))"

# 冒烟
HF_HUB_OFFLINE=1 python -c "
import torch, timm
m = timm.create_model('hf_hub:prov-gigapath/prov-gigapath', pretrained=True).to('mps').eval()
print('Tile encoder:', m(torch.randn(1,3,224,224).to('mps')).shape)
"
```
