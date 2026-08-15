# RIFT：Keep the Future, Drop the Rollout

论文 **[Keep the Future, Drop the Rollout: RIFT for World Action Models](https://arxiv.org/abs/2608.11521)**
的官方实现。

[![arXiv](https://img.shields.io/badge/arXiv-2608.11521-b31b1b.svg)](https://arxiv.org/abs/2608.11521)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-f7c843)](https://huggingface.co/PoopBear/RIFT)
[![LIBERO Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-LIBERO%20Dataset-f7c843)](https://huggingface.co/datasets/yuanty/LIBERO-fastwam)
[![RoboTwin Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-RoboTwin%20Dataset-f7c843)](https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-blue.svg)](./LICENSE)

[![English](https://img.shields.io/badge/README-English-111111.svg)](./README.md)
[![Chinese](https://img.shields.io/badge/README-Chinese-d14836.svg)](./README_zh.md)

## 目录

- [已发布 checkpoint](#已发布-checkpoint)
- [发布范围](#发布范围)
- [仓库结构](#仓库结构)
- [环境](#环境)
- [模型准备](#模型准备)
- [数据下载](#数据下载)
- [训练](#训练)
- [Benchmark evaluation](#benchmark-evaluation)
- [致谢](#致谢)
- [引用](#引用)

## 已发布 checkpoint

可以从 [Hugging Face](https://huggingface.co/PoopBear/RIFT/resolve/main/rift_step021700.pt?download=true)
直接下载已发布的 LIBERO checkpoint，也可以同时下载 normalization 和配置文件：

```bash
pip install -U huggingface_hub
hf download PoopBear/RIFT \
  rift_step021700.pt dataset_stats.json config.yaml \
  --local-dir ./checkpoints/rift
```

下载后的目录：

```text
checkpoints/rift/
├── rift_step021700.pt
├── dataset_stats.json
└── config.yaml
```

将 `dataset_stats.json` 与 checkpoint 放在一起，以保留发布权重使用的 normalization
metadata。下载的 `config.yaml` 与 `configs/model/rift.yaml` 一致，并使用当前的
`rift.*` namespace；运行时请与本仓库的 task 和 data configs 组合。Checkpoint
保存的是 tensor state，而不是 pickle 的模型类，因此 namespace 迁移不会改变权重。

## 发布范围

本次发布包含两个 canonical task 配置：

- [`configs/task/libero_rift_2cam224_1e-4.yaml`](./configs/task/libero_rift_2cam224_1e-4.yaml)；
- [`configs/task/robotwin_rift_3cam_384_1e-4.yaml`](./configs/task/robotwin_rift_3cam_384_1e-4.yaml)。

它们固定以下设计：

- FastWAM 原生 video co-training，然后执行 anticipation/action pass；
- full-grid anticipation tokens：LIBERO 为 196，RoboTwin 为 240；
- motion-aware render target 和后期 loss annealing；
- 仅作用于 render branch 的后期 conditioning-noise curriculum；
- flow matching 作为塑造 representation 的 objective。

## 仓库结构

```text
RIFT/
├── configs/
│   ├── data/                 # 数据配置
│   ├── model/rift.yaml       # Canonical 模型配置
│   ├── task/                 # Hydra task 配置
│   └── sim_*.yaml            # 独立 evaluation 配置
├── experiments/
│   ├── libero/               # LIBERO worker 和多 GPU manager
│   └── robotwin/             # RoboTwin adapter 和多 GPU manager
├── scripts/
│   ├── train.py
│   ├── train_zero2.sh        # DeepSpeed ZeRO-2 训练入口
│   ├── preprocess_action_dit_backbone.py
│   └── precompute_text_embeds.py
├── rift/                     # 模型、数据、runtime 和 trainer 包
└── tests/                    # 离线配置与 schema 测试
```

## 环境

已验证的环境使用 Python 3.10、PyTorch 2.7.1 和 CUDA 12.8：

```bash
conda create -n rift python=3.10 -y
conda activate rift
pip install --upgrade pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

各入口会先加载当前 checkout 根目录下的 `rift/` 包，再查询环境中安装的包。

后续命令都在仓库根目录执行。将 shell 变量 `N` 设为可用的 80 GB 级 GPU 数量。

## 模型准备

RIFT 使用 [`configs/model/rift.yaml`](./configs/model/rift.yaml) 中声明的 Wan2.2 TI2V
5B video backbone 和 Wan2.1 tokenizer/text encoder。仓库不重新分发这些外部权重。

```bash
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface

python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/rift.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

预处理会先下载并加载所需 Wan model components，再构造 ActionDiT backbone。请使用
高显存 GPU 或高内存 CPU host。`DIFFSYNTH_DOWNLOAD_SOURCE` 支持 `huggingface` 和
`modelscope`；未设置时默认使用 `modelscope`。

## 数据下载

### LIBERO

从固定 dataset revision 下载四个预处理后的 LIBERO 压缩包：

```bash
mkdir -p data/downloads/libero
hf download yuanty/LIBERO-fastwam \
  libero_10_no_noops_lerobot.tar.gz \
  libero_goal_no_noops_lerobot.tar.gz \
  libero_object_no_noops_lerobot.tar.gz \
  libero_spatial_no_noops_lerobot.tar.gz \
  --repo-type dataset \
  --revision 117413dc0ca99c7cd64036c4eaa4a316c537d692 \
  --local-dir data/downloads/libero

mkdir -p data/libero_mujoco3.3.2
for archive in data/downloads/libero/*.tar.gz; do
  tar -xzf "${archive}" -C data/libero_mujoco3.3.2
done
```

目录结构：

```text
data/libero_mujoco3.3.2/
├── libero_10_no_noops_lerobot/
├── libero_goal_no_noops_lerobot/
├── libero_object_no_noops_lerobot/
└── libero_spatial_no_noops_lerobot/
```

发布配置直接读取数据集 MP4，不需要本地 video cache。视频使用 AV1；TorchCodec 不可用
时会回退到固定的 PyAV 依赖。训练前检查一个真实文件：

```bash
python - <<'PY'
from pathlib import Path
import av

path = next(Path("data/libero_mujoco3.3.2").rglob("*.mp4"))
with av.open(str(path)) as container:
    frame = next(container.decode(video=0))
print(path, frame.width, frame.height)
PY
```

数据和对应许可证不随本仓库分发。

### RoboTwin

下载固定的 RoboTwin 预处理数据 snapshot。八个分片解压前约占 84 GB：

```bash
hf download yuanty/robotwin2.0-fastwam \
  dataset_stats.json \
  robotwin2.0.tar.gz.part-00 robotwin2.0.tar.gz.part-01 \
  robotwin2.0.tar.gz.part-02 robotwin2.0.tar.gz.part-03 \
  robotwin2.0.tar.gz.part-04 robotwin2.0.tar.gz.part-05 \
  robotwin2.0.tar.gz.part-06 robotwin2.0.tar.gz.part-07 \
  --repo-type dataset \
  --revision aac262c35d02cc71b2f6ef670bd65fd9f2bb2547 \
  --local-dir data/downloads/robotwin2.0

mkdir -p data/robotwin2.0
cp data/downloads/robotwin2.0/dataset_stats.json data/robotwin2.0/
cat data/downloads/robotwin2.0/robotwin2.0.tar.gz.part-* | \
  tar -xzf - -C data/robotwin2.0
```

解压后的目录是 `data/robotwin2.0/robotwin2.0/`，与
[`configs/data/robotwin.yaml`](./configs/data/robotwin.yaml) 一致。固定 snapshot 按配置划分后
产生 6,011,575 个 training samples。

## 训练

### 1）预计算 T5 embedding cache

为每个训练 task 运行 `scripts/precompute_text_embeds.py`：

```bash
# LIBERO
python scripts/precompute_text_embeds.py task=libero_rift_2cam224_1e-4

# RoboTwin
python scripts/precompute_text_embeds.py task=robotwin_rift_3cam_384_1e-4
```

多 GPU 预计算：

```bash
torchrun --standalone --nproc_per_node="$N" \
  scripts/precompute_text_embeds.py task=libero_rift_2cam224_1e-4
```

### 2）训练

```bash
# LIBERO
bash scripts/train_zero2.sh "$N" task=libero_rift_2cam224_1e-4

# RoboTwin
bash scripts/train_zero2.sh "$N" task=robotwin_rift_3cam_384_1e-4
```

将 `N` 设为使用的 GPU 数量。Dataset、model、batch size 和 schedule 由所选 task 配置提供。

## Benchmark evaluation

Evaluation 结构与 FastWAM 官方保持同级：每个 benchmark 都有单 task worker 和多 GPU
manager。这里替换为 RIFT model/task config、checkpoint schema 和两/三相机输入。
Simulator 作为外部依赖安装，不 vendoring 到本仓库。

### LIBERO

在 RIFT 环境中安装 [官方 LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
和 MuJoCo 3.3.2。Evaluation path 已用 LIBERO commit
`8f1084e3132a39270c3a13ebe37270a43ece2a01` 实测：

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /path/to/LIBERO
git -C /path/to/LIBERO checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
pip install -e /path/to/LIBERO
pip install mujoco==3.3.2
```

使用 `N` 张 GPU、每 task 50 trials 运行四个标准 suite：

```bash
python experiments/libero/run_libero_manager.py \
  task=libero_rift_2cam224_1e-4 \
  ckpt=./checkpoints/rift/rift_step021700.pt \
  EVALUATION.dataset_stats_path=./checkpoints/rift/dataset_stats.json \
  MULTIRUN.num_gpus="$N"
```

结果写入 `evaluate_results/libero/`，包含 per-task JSON、`summary.json`、
`summary.csv` 和 `task_success_rates.csv`。

### RoboTwin

单独安装 [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) 及其 simulator assets 和
task configs，然后指向该 checkout。Adapter 会在运行时链接到外部 checkout；
本仓库不复制 RoboTwin 源码。

```bash
git clone https://github.com/RoboTwin-Platform/RoboTwin.git /path/to/RoboTwin
git -C /path/to/RoboTwin checkout bf44be51cf5717a5595ce59447f2cf5263d2aa95
# 在该 revision 上按 RoboTwin 官方说明完成环境和 assets 安装。
export ROBOTWIN_ROOT=/path/to/RoboTwin
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_rift_3cam_384_1e-4 \
  ckpt=/path/to/your_robotwin_checkpoint.pt \
  EVALUATION.dataset_stats_path=./data/robotwin2.0/dataset_stats.json \
  MULTIRUN.num_gpus="$N"
```

Manager 依次评估 `demo_clean` 和 `demo_randomized`，并将 per-task 与聚合结果写入
`evaluate_results/robotwin/`。通过 `ckpt=...` 传入按本仓库 RoboTwin 配方训练的
checkpoint。默认评估 unseen instructions；传入
`EVALUATION.instruction_type=seen` 可评估 seen instructions。
Stock RoboTwin evaluator 按其标准设置每 task 执行 100 episodes，并在每个 control
step 渲染 observation。

## 致谢

本代码基于 [FastWAM](https://github.com/yuantianyuan01/FastWAM) 的训练和评测框架，
并包含适配后的 RoboTwin 评测代码。感谢 Wan、LIBERO、RoboTwin、LeRobot 和
DiffSynth 等开源社区。

## 引用

如果这个仓库对你的研究有帮助，请引用：

```bibtex
@article{zhang2026rift,
  title={Keep the Future, Drop the Rollout: RIFT for World Action Models},
  author={Zhang, Chushan and Tong, Jinguang and Li, Xuesong and Wang, Yikai and Li, Hongdong},
  journal={arXiv preprint arXiv:2608.11521},
  year={2026}
}
```
