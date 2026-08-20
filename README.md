# RIFT: Keep the Future, Drop the Rollout

Official implementation of
**[Keep the Future, Drop the Rollout: RIFT for World Action Models](https://arxiv.org/abs/2608.11521)**.

[![arXiv](https://img.shields.io/badge/arXiv-2608.11521-b31b1b.svg)](https://arxiv.org/abs/2608.11521)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-f7c843)](https://huggingface.co/PoopBear/RIFT)
[![LIBERO Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-LIBERO%20Dataset-f7c843)](https://huggingface.co/datasets/yuanty/LIBERO-fastwam)
[![RoboTwin Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-RoboTwin%20Dataset-f7c843)](https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-blue.svg)](./LICENSE)

[![English](https://img.shields.io/badge/README-English-111111.svg)](./README.md)
[![Chinese](https://img.shields.io/badge/README-Chinese-d14836.svg)](./README_zh.md)

## Contents

- [Released checkpoint](#released-checkpoint)
- [Released scope](#released-scope)
- [Real-world Galaxea](#real-world-galaxea)
- [Repository layout](#repository-layout)
- [Environment](#environment)
- [Model preparation](#model-preparation)
- [Dataset download](#dataset-download)
- [Training](#training)
- [Benchmark evaluation](#benchmark-evaluation)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

## Released checkpoint

Download the released LIBERO checkpoint directly from
[Hugging Face](https://huggingface.co/PoopBear/RIFT/resolve/main/rift_step021700.pt?download=true),
or download it with its normalization and configuration files:

```bash
pip install -U huggingface_hub
hf download PoopBear/RIFT \
  rift_step021700.pt dataset_stats.json config.yaml \
  --local-dir ./checkpoints/rift
```

Expected layout:

```text
checkpoints/rift/
├── rift_step021700.pt
├── dataset_stats.json
└── config.yaml
```

Keep `dataset_stats.json` with the checkpoint to preserve the normalization
metadata used by the released weights. The downloaded `config.yaml` mirrors
`configs/model/rift.yaml` and uses the current `rift.*` namespace. Compose it
with this repository's task and data configs. The checkpoint stores tensor
state rather than pickled model classes, so the namespace migration does not
change its weights.

## Released scope

The release contains two canonical task configurations:

- [`configs/task/libero_rift_2cam224_1e-4.yaml`](./configs/task/libero_rift_2cam224_1e-4.yaml);
- [`configs/task/robotwin_rift_3cam_384_1e-4.yaml`](./configs/task/robotwin_rift_3cam_384_1e-4.yaml).

They fix the following design:

- native FastWAM video co-training followed by the anticipation/action pass;
- full-grid anticipation tokens: 196 for LIBERO and 240 for RoboTwin;
- a motion-aware render target and late loss annealing;
- a late conditioning-noise curriculum limited to the render branch;
- flow matching as the representation-shaping objective.

## Real-world Galaxea

This repository includes a real-world Galaxea indoor-cleaning recipe built around
the same filtered 445-episode subset:

- [`configs/data/galaxea_indoor_cleaning.yaml`](./configs/data/galaxea_indoor_cleaning.yaml);
- [`configs/task/galaxea_indoor_cleaning_rift_3cam224_1e-4.yaml`](./configs/task/galaxea_indoor_cleaning_rift_3cam224_1e-4.yaml).

Provide machine-specific paths through `GALAXEA_DATA_ROOT`,
`GALAXEA_NORM_STATS`, and `GALAXEA_TEXT_CACHE` rather than committing them to the
repository.

The matching FastWAM baseline was trained for 10 epochs (14,830 optimizer steps)
on this subset. Its released files are available on
[Hugging Face](https://huggingface.co/PoopBear/RIFT/tree/main/fastwam/galaxea_indoor_cleaning_3cam224_10ep).

RIFT training on the same subset is currently in progress; a real-world RIFT
checkpoint has not yet been released. No real-world evaluation result is claimed
here.

## Repository layout

```text
RIFT/
├── configs/
│   ├── data/                 # Dataset configuration
│   ├── model/rift.yaml       # Canonical model configuration
│   ├── task/                 # Hydra task configuration
│   └── sim_*.yaml            # Standalone evaluation configuration
├── experiments/
│   ├── libero/               # LIBERO workers and multi-GPU manager
│   └── robotwin/             # RoboTwin adapter and multi-GPU manager
├── scripts/
│   ├── train.py
│   ├── train_zero2.sh        # DeepSpeed ZeRO-2 training entrypoint
│   ├── preprocess_action_dit_backbone.py
│   └── precompute_text_embeds.py
├── rift/                     # Model, data, runtime, and trainer package
└── tests/                    # Offline configuration and schema tests
```

## Environment

The known-good environment uses Python 3.10, PyTorch 2.7.1, and CUDA 12.8:

```bash
conda create -n rift python=3.10 -y
conda activate rift
pip install --upgrade pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

The entrypoints resolve the root-level `rift/` package from this checkout before
installed packages.

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

Download the four preprocessed LIBERO archives at the pinned dataset revision:

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

Expected layout:

```text
data/libero_mujoco3.3.2/
├── libero_10_no_noops_lerobot/
├── libero_goal_no_noops_lerobot/
├── libero_object_no_noops_lerobot/
└── libero_spatial_no_noops_lerobot/
```

The release reads the dataset MP4 files directly and does not require a local
video cache. The files use AV1 video; the pinned PyAV package is the fallback
decoder when TorchCodec is unavailable. Verify one real file before training:

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

Dataset files and their license are not redistributed here.

### RoboTwin

Download the fixed preprocessed RoboTwin snapshot. The eight archive parts use
about 84 GB before extraction:

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

The resulting dataset directory is `data/robotwin2.0/robotwin2.0/`, matching
[`configs/data/robotwin.yaml`](./configs/data/robotwin.yaml). The pinned snapshot
produced 6,011,575 training samples after the configured split.

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

### 2) Training

```bash
# LIBERO
bash scripts/train_zero2.sh "$N" task=libero_rift_2cam224_1e-4

# RoboTwin
bash scripts/train_zero2.sh "$N" task=robotwin_rift_3cam_384_1e-4

# Real-world Galaxea
bash scripts/train_zero2.sh "$N" task=galaxea_indoor_cleaning_rift_3cam224_1e-4
```

Set `N` to the number of GPUs to use. Dataset, model, batch size, and
schedule settings come from the selected task configuration.

## Benchmark evaluation

The evaluation layout follows FastWAM's independent benchmark entrypoints: one
single-task worker plus a multi-GPU manager for each benchmark. The RIFT model,
task configs, checkpoint schema, and two-camera/three-camera inputs replace the
upstream policy defaults. The simulators remain external dependencies; this
repository does not vendor either benchmark.

### LIBERO

Install the [official LIBERO environment](https://github.com/Lifelong-Robot-Learning/LIBERO)
in the RIFT environment and use MuJoCo 3.3.2. The evaluation path was tested
with LIBERO commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /path/to/LIBERO
git -C /path/to/LIBERO checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
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

Results are written below `evaluate_results/libero/` as per-task JSON files,
`summary.json`, `summary.csv`, and `task_success_rates.csv`.

### RoboTwin

Install [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) separately,
including its simulator assets and task configs, then point RIFT to that
checkout. The adapter is linked into the external checkout at runtime; no
RoboTwin source is copied into this repository.

```bash
git clone https://github.com/RoboTwin-Platform/RoboTwin.git /path/to/RoboTwin
git -C /path/to/RoboTwin checkout bf44be51cf5717a5595ce59447f2cf5263d2aa95
# Complete RoboTwin's environment and asset installation at this revision.
export ROBOTWIN_ROOT=/path/to/RoboTwin
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_rift_3cam_384_1e-4 \
  ckpt=/path/to/your_robotwin_checkpoint.pt \
  EVALUATION.dataset_stats_path=./data/robotwin2.0/dataset_stats.json \
  MULTIRUN.num_gpus="$N"
```

The manager evaluates both `demo_clean` and `demo_randomized`, then writes
per-task and aggregate results below `evaluate_results/robotwin/`. Pass a
checkpoint trained with the included RoboTwin recipe through `ckpt=...`.
Evaluation defaults to
unseen instructions. Set `EVALUATION.instruction_type=seen` to evaluate seen
instructions. The stock RoboTwin evaluator runs its standard 100 episodes per
task and renders observations at every control step.

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
