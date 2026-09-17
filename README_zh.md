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
- [真实环境 Galaxea](#真实环境-galaxea)
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

将 `dataset_stats.json` 与 checkpoint 放在同一目录，用于动作归一化。

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

## 真实环境 Galaxea

本仓库包含真实环境 Galaxea 室内清洁训练配方，使用同一个过滤后的 445-episode
子集：

- [`configs/data/galaxea_indoor_cleaning.yaml`](./configs/data/galaxea_indoor_cleaning.yaml)；
- [`configs/task/galaxea_indoor_cleaning_rift_3cam224_1e-4.yaml`](./configs/task/galaxea_indoor_cleaning_rift_3cam224_1e-4.yaml)。

机器相关路径通过 `GALAXEA_DATA_ROOT`、`GALAXEA_NORM_STATS` 和
`GALAXEA_TEXT_CACHE` 提供，不写入仓库。

以下仅权重 checkpoint 均在该子集上训练 10 epochs，并发布于
[Hugging Face](https://huggingface.co/PoopBear/RIFT)：

- [FastWAM baseline](https://huggingface.co/PoopBear/RIFT/tree/main/fastwam/galaxea_indoor_cleaning_3cam224_10ep)；
- [FastWAM-Joint](https://huggingface.co/PoopBear/RIFT/tree/main/fastwam_joint/galaxea_indoor_cleaning_3cam224_10ep)；
- [RIFT](https://huggingface.co/PoopBear/RIFT/tree/main/rift/galaxea_indoor_cleaning_3cam224_10ep)。

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

使用 Python 3.10、PyTorch 2.7.1 和 CUDA 12.8：

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

下载四个预处理后的 LIBERO 压缩包：

```bash
mkdir -p data/downloads/libero
hf download yuanty/LIBERO-fastwam \
  libero_10_no_noops_lerobot.tar.gz \
  libero_goal_no_noops_lerobot.tar.gz \
  libero_object_no_noops_lerobot.tar.gz \
  libero_spatial_no_noops_lerobot.tar.gz \
  --repo-type dataset \
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

MP4 文件使用 AV1；TorchCodec 不可用时使用 PyAV 解码。

### RoboTwin

下载预处理后的 RoboTwin 数据集。八个分片解压前约占 84 GB：

```bash
hf download yuanty/robotwin2.0-fastwam \
  dataset_stats.json \
  robotwin2.0.tar.gz.part-00 robotwin2.0.tar.gz.part-01 \
  robotwin2.0.tar.gz.part-02 robotwin2.0.tar.gz.part-03 \
  robotwin2.0.tar.gz.part-04 robotwin2.0.tar.gz.part-05 \
  robotwin2.0.tar.gz.part-06 robotwin2.0.tar.gz.part-07 \
  --repo-type dataset \
  --local-dir data/downloads/robotwin2.0

mkdir -p data/robotwin2.0
cp data/downloads/robotwin2.0/dataset_stats.json data/robotwin2.0/
cat data/downloads/robotwin2.0/robotwin2.0.tar.gz.part-* | \
  tar -xzf - -C data/robotwin2.0
```

解压后的目录是 `data/robotwin2.0/robotwin2.0/`，与
[`configs/data/robotwin.yaml`](./configs/data/robotwin.yaml) 一致。

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

# 真实环境 Galaxea
bash scripts/train_zero2.sh "$N" \
  task=galaxea_indoor_cleaning_rift_3cam224_1e-4 \
  batch_size=16 gradient_accumulation_steps=2 \
  num_epochs=10 max_steps=14830 \
  log_every=10 save_every=0 resume=null wandb.enabled=false
```

将 `N` 设为使用的 GPU 数量。Dataset、model、batch size 和 schedule 由所选 task 配置提供。

## Benchmark evaluation

LIBERO 和 RoboTwin 使用独立的评估入口。

### LIBERO

安装 [官方 LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) 和 MuJoCo 3.3.2：

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /path/to/LIBERO
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

### LIBERO-Plus

安装 [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus)，然后运行：

```bash
LIBERO_PLUS_ROOT=/path/to/LIBERO-plus \
  bash scripts/run_libero_plus.sh \
  ./checkpoints/rift/rift_step021700.pt \
  ./checkpoints/rift/dataset_stats.json \
  ./evaluate_results/libero_plus
```

脚本对每个 task 执行一次 rollout，并从已完成的 task receipt 继续运行。
通过 `GPU_IDS` 和 `WORKERS_PER_GPU` 设置并行度。

### RoboTwin

单独安装 [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) 及其 simulator assets 和
task configs，然后指向该 checkout。Adapter 会在运行时链接到外部 checkout；
本仓库不复制 RoboTwin 源码。

```bash
git clone https://github.com/RoboTwin-Platform/RoboTwin.git /path/to/RoboTwin
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
