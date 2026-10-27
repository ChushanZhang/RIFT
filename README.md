# RIFT: Keep the Future, Drop the Rollout

Official implementation of
**[Keep the Future, Drop the Rollout: RIFT for World Action Models](https://arxiv.org/abs/2608.11521)**.

RIFT learns future-aware action representations with flow matching and video co-training.

[![arXiv](https://img.shields.io/badge/arXiv-2608.11521-b31b1b.svg)](https://arxiv.org/abs/2608.11521)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-f7c843)](https://huggingface.co/PoopBear/RIFT)
[![LIBERO Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-LIBERO%20Dataset-f7c843)](https://huggingface.co/datasets/yuanty/LIBERO-fastwam)
[![RoboTwin Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-RoboTwin%20Dataset-f7c843)](https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-blue.svg)](./LICENSE)

[![English](https://img.shields.io/badge/README-English-111111.svg)](./README.md)
[![Chinese](https://img.shields.io/badge/README-Chinese-d14836.svg)](./README_zh.md)

## Real-world demos

<table>
  <tr><th>Basket Stacking</th><th>Test-Tube Mixing</th><th>Object Sorting</th></tr>
  <tr>
    <td width="33%"><video src="https://github.com/user-attachments/assets/2928212a-961a-4a30-8672-97daee608d52" controls></video></td>
    <td width="33%"><video src="https://github.com/user-attachments/assets/bf179b18-5ce8-4bd8-b85e-606d90bf202f" controls></video></td>
    <td width="33%"><video src="https://github.com/user-attachments/assets/93fdfa27-0d9f-4733-b7e2-8f1edd0c853d" controls></video></td>
  </tr>
</table>

## Released checkpoints

Download checkpoints and their matching configuration and normalization files from
[Hugging Face](https://huggingface.co/PoopBear/RIFT).

## Environment

Use Python 3.10, PyTorch 2.7.1, and CUDA 12.8:

```bash
conda create -n rift python=3.10 -y
conda activate rift
pip install --upgrade pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

Run subsequent commands from the repository root. Set shell variable `N` to the number of
available 80 GB-class GPUs.

## Model preparation

RIFT uses the Wan2.2 TI2V 5B video backbone and Wan2.1 tokenizer/text encoder
declared in [`configs/model/rift.yaml`](./configs/model/rift.yaml). These
external weights are not redistributed here.

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

The preprocessing command downloads and loads the required Wan model
components before constructing the ActionDiT backbone. Use a high-memory GPU
or a high-RAM CPU host. `DIFFSYNTH_DOWNLOAD_SOURCE` accepts `huggingface` or
`modelscope`; the code defaults to `modelscope` when it is unset.

## Dataset download

### LIBERO

Download the four preprocessed LIBERO archives:

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

Download the preprocessed RoboTwin dataset. The eight archive parts use about
84 GB before extraction:

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

The resulting dataset directory is `data/robotwin2.0/robotwin2.0/`, matching
[`configs/data/robotwin.yaml`](./configs/data/robotwin.yaml).

## Training

### 1) Precompute the T5 embedding cache

Use `scripts/precompute_text_embeds.py` for each training task:

```bash
# LIBERO
python scripts/precompute_text_embeds.py task=libero_rift_2cam224_1e-4

# RoboTwin
python scripts/precompute_text_embeds.py task=robotwin_rift_3cam_384_1e-4
```

For multi-GPU preprocessing:

```bash
torchrun --standalone --nproc_per_node="$N" \
  scripts/precompute_text_embeds.py task=libero_rift_2cam224_1e-4
```

### 2) Simulation training

```bash
# LIBERO
bash scripts/train_zero2.sh "$N" task=libero_rift_2cam224_1e-4

# RoboTwin
bash scripts/train_zero2.sh "$N" task=robotwin_rift_3cam_384_1e-4
```

For LIBERO, [packed episode VAE caching](docs/libero_episode_cache.md) skips RGB
decoding and online VAE encoding. Set `vae_cache_dir=/path/to/cache` after
precomputing and validating the cache; leave it `null` for online encoding.

### 3) Real-world training (Galaxea / RoboCOIN)

Galaxea uses the [filtered 445-episode indoor-cleaning subset](./configs/data/galaxea_indoor_cleaning.yaml).
Set `GALAXEA_DATA_ROOT`, `GALAXEA_NORM_STATS`, and `GALAXEA_TEXT_CACHE`.
For RoboCOIN, follow the
[data preparation and packed episode cache guide](docs/robocoin_precompute.md) and set
`ROBOCOIN_TRAIN_ROOT`, `ROBOCOIN_NORM_STATS`, and `ROBOCOIN_TEXT_CACHE`.
Both examples use 8 GPUs, a global batch size of 256, and 10 epochs; adjust
`max_steps` if the dataset size or global batch size changes.

```bash
# Galaxea indoor cleaning
bash scripts/train_zero2.sh 8 \
  task=galaxea_indoor_cleaning_rift_3cam224_1e-4 \
  gradient_accumulation_steps=2 \
  num_epochs=10 max_steps=14830 \
  save_every=0

# RoboCOIN multitask
bash scripts/train_zero2.sh 8 \
  task=galaxea_indoor_cleaning_rift_3cam224_1e-4 data=robocoin_multitask \
  gradient_accumulation_steps=2 \
  num_epochs=10 max_steps=9340 \
  save_every=0
```

### ⚡ Speed up training (optional)

Enabling all acceleration options is expected to improve training speed by approximately 35%, depending on the training configuration.

The following settings can be passed as Hydra overrides to any training task
above. Their defaults preserve the original training execution.

| Setting | Default | Effect when enabled |
| --- | --- | --- |
| `model.compile_training_denoise` | `false` | Compile MoT training layers while retaining gradient checkpointing. |
| `model.fuse_training_paths` | `false` | Batch the two training paths together when their video grids match. |
| `fused_adamw` | `false` | Fuse AdamW updates on CUDA; CPU keeps the standard optimizer. |
| `model.video_future_head_only` | `false` | Apply the native video output head only to supervised future frames; unsupported or custom heads retain the full-head path. |
| `model.compile_future_head` | `false` | Compile the future flow-matching decoder during gradient-enabled training. |
| `model.vae_encode_mode` | `sequential` | Use `batched` for batched online VAE encoding, or `cudagraphs` to also enable CUDA Graphs. |
| `model.vae_encode_batch_size` | `null` | In `batched` / `cudagraphs` mode, limit the online VAE batch to a positive number to reduce peak memory; `null` uses the full batch. Ignored in `sequential` mode. |

Enable the complete training combination with the command below. ZeRO-1 requires
more VRAM than ZeRO-2. Compilation adds startup or recompilation time, and CUDA
Graphs require a CUDA device.

```bash
bash scripts/train_zero1.sh "$N" task=libero_rift_2cam224_1e-4 \
  model.compile_training_denoise=true model.fuse_training_paths=true \
  model.video_future_head_only=true model.compile_future_head=true \
  fused_adamw=true model.vae_encode_mode=cudagraphs
```

[Precompute VAE latents (RoboCOIN guide)](docs/robocoin_precompute.md#generate-and-verify-vae-latents)
to read cached latents during training and bypass online VAE encoding. The two
VAE execution settings above apply only when encoding online.

## Benchmark evaluation

### LIBERO

Install the [official LIBERO environment](https://github.com/Lifelong-Robot-Learning/LIBERO)
and MuJoCo 3.3.2:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /path/to/LIBERO
pip install -e /path/to/LIBERO
pip install mujoco==3.3.2
```

Run all four standard suites on `N` GPUs with 50 trials per task:

```bash
python experiments/libero/run_libero_manager.py \
  task=libero_rift_2cam224_1e-4 \
  ckpt=./checkpoints/rift/rift_step021700.pt \
  EVALUATION.dataset_stats_path=./checkpoints/rift/dataset_stats.json \
  MULTIRUN.num_gpus="$N"
```

Results are saved to `evaluate_results/libero/`.

### LIBERO-Plus

Install [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus), then run:

```bash
LIBERO_PLUS_ROOT=/path/to/LIBERO-plus \
  bash scripts/run_libero_plus.sh \
  ./checkpoints/rift/rift_step021700.pt \
  ./checkpoints/rift/dataset_stats.json \
  ./evaluate_results/libero_plus
```

The launcher evaluates one rollout per task and resumes from completed task
receipts. Set `GPU_IDS` and `WORKERS_PER_GPU` to control parallelism.

### RoboTwin

Install [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin), including its
simulator assets and task configs, and set `ROBOTWIN_ROOT` to that checkout:

```bash
git clone https://github.com/RoboTwin-Platform/RoboTwin.git /path/to/RoboTwin
export ROBOTWIN_ROOT=/path/to/RoboTwin
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_rift_3cam_384_1e-4 \
  ckpt=/path/to/your_robotwin_checkpoint.pt \
  EVALUATION.dataset_stats_path=./data/robotwin2.0/dataset_stats.json \
  MULTIRUN.num_gpus="$N"
```

Both `demo_clean` and `demo_randomized` run 100 episodes per task, with results
saved to `evaluate_results/robotwin/`. Evaluation uses unseen instructions;
set `EVALUATION.instruction_type=seen` for seen instructions.

## Acknowledgements

This codebase builds on the [FastWAM](https://github.com/yuantianyuan01/FastWAM)
training and evaluation stack and includes adapted RoboTwin evaluation code.
We thank the Wan, LIBERO, RoboTwin, LeRobot, and DiffSynth communities for
their open-source infrastructure.

## Citation

If you find this repository useful, please cite:

```bibtex
@article{zhang2026rift,
  title={Keep the Future, Drop the Rollout: RIFT for World Action Models},
  author={Zhang, Chushan and Tong, Jinguang and Li, Xuesong and Wang, Yikai and Li, Hongdong},
  journal={arXiv preprint arXiv:2608.11521},
  year={2026}
}
```

## Archived experiment jobs

Historical Isaac job scripts, logs, results, weights and caches are available in the public [research-vla/isaac-jobs archive](https://huggingface.co/datasets/research-vla/isaac-jobs). See the [verified snapshot, directory index and restore instructions](docs/archives/isaac-jobs-20261027.md) for records removed from local scratch storage.
