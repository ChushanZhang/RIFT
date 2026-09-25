# RIFT：Keep the Future, Drop the Rollout

论文 **[Keep the Future, Drop the Rollout: RIFT for World Action Models](https://arxiv.org/abs/2608.11521)**
的官方实现。

RIFT 通过 flow matching 和视频联合训练学习融合未来信息的动作表征。

[![arXiv](https://img.shields.io/badge/arXiv-2608.11521-b31b1b.svg)](https://arxiv.org/abs/2608.11521)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-f7c843)](https://huggingface.co/PoopBear/RIFT)
[![LIBERO Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-LIBERO%20Dataset-f7c843)](https://huggingface.co/datasets/yuanty/LIBERO-fastwam)
[![RoboTwin Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-RoboTwin%20Dataset-f7c843)](https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-blue.svg)](./LICENSE)

[![English](https://img.shields.io/badge/README-English-111111.svg)](./README.md)
[![Chinese](https://img.shields.io/badge/README-Chinese-d14836.svg)](./README_zh.md)

## 真机演示

<table>
  <tr><th>篮筐堆叠</th><th>试管混合</th><th>物体分类</th></tr>
  <tr>
    <td width="33%"><video src="https://github.com/user-attachments/assets/c9e1d67f-2696-454b-a3b0-08aaa960e630" controls></video></td>
    <td width="33%"><video src="https://github.com/user-attachments/assets/714e534f-84ca-4d8b-9c06-18ba2a5e38d6" controls></video></td>
    <td width="33%"><video src="https://github.com/user-attachments/assets/b8d897fe-e1c6-40ea-b40b-3bcf5ea90a8e" controls></video></td>
  </tr>
</table>

播放倍率：篮筐堆叠 1.5×；试管混合与物体分类 2×。

## 已发布 checkpoint

Checkpoint 及配套的配置和归一化文件可从 [Hugging Face](https://huggingface.co/PoopBear/RIFT) 下载。

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

### 2）仿真训练

```bash
# LIBERO
bash scripts/train_zero2.sh "$N" task=libero_rift_2cam224_1e-4

# RoboTwin
bash scripts/train_zero2.sh "$N" task=robotwin_rift_3cam_384_1e-4
```

### 3）真实环境训练（Galaxea / RoboCOIN）

Galaxea 使用[过滤后的 445-episode 室内清洁子集](./configs/data/galaxea_indoor_cleaning.yaml)，
设置 `GALAXEA_DATA_ROOT`、`GALAXEA_NORM_STATS` 和 `GALAXEA_TEXT_CACHE`。
RoboCOIN 按[数据准备指南](docs/robocoin_precompute.md)准备数据，并设置
`ROBOCOIN_TRAIN_ROOT`、`ROBOCOIN_NORM_STATS` 和 `ROBOCOIN_TEXT_CACHE`。
以下两条命令均使用 8 张 GPU、全局 batch size 256，训练 10 epochs；
数据量或全局 batch size 改变时需相应调整 `max_steps`。

```bash
# Galaxea 室内清洁
bash scripts/train_zero2.sh 8 \
  task=galaxea_indoor_cleaning_rift_3cam224_1e-4 \
  batch_size=16 gradient_accumulation_steps=2 \
  num_epochs=10 max_steps=14830 \
  log_every=10 save_every=0 resume=null wandb.enabled=false

# RoboCOIN 多任务
bash scripts/train_zero2.sh 8 \
  task=galaxea_indoor_cleaning_rift_3cam224_1e-4 data=robocoin_multitask \
  batch_size=16 gradient_accumulation_steps=2 \
  num_epochs=10 max_steps=9340 \
  log_every=10 save_every=0 resume=null wandb.enabled=false
```

### ⚡ 训练加速（可选）

启用本节全部加速选项后，训练速度预计可提升约 30%，具体取决于训练配置。

使用 ZeRO-1，开启 MoT 编译、训练路径融合，以及 VAE 在线批量编码与 CUDA Graph。
ZeRO-1 比 ZeRO-2 需要更多显存；这些选项可用于上述任一训练 task：

```bash
bash scripts/train_zero1.sh "$N" task=libero_rift_2cam224_1e-4 \
  model.compile_training_denoise=true \
  model.fuse_training_paths=true \
  model.vae_encode_mode=cudagraphs
```

[预计算 VAE latents（RoboCOIN 指南）](docs/robocoin_precompute.md#generate-and-verify-vae-latents)
后，训练时可直接读取缓存的 latents，跳过在线 VAE 编码。

## Benchmark evaluation

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

结果写入 `evaluate_results/libero/`。

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

安装 [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) 及其仿真资源和任务配置，
将 `ROBOTWIN_ROOT` 指向该 checkout：

```bash
git clone https://github.com/RoboTwin-Platform/RoboTwin.git /path/to/RoboTwin
export ROBOTWIN_ROOT=/path/to/RoboTwin
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_rift_3cam_384_1e-4 \
  ckpt=/path/to/your_robotwin_checkpoint.pt \
  EVALUATION.dataset_stats_path=./data/robotwin2.0/dataset_stats.json \
  MULTIRUN.num_gpus="$N"
```

`demo_clean` 和 `demo_randomized` 各运行每 task 100 episodes，结果写入
`evaluate_results/robotwin/`。默认使用 unseen instructions；设置
`EVALUATION.instruction_type=seen` 可评估 seen instructions。

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
