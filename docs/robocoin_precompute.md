# RoboCOIN training precomputations

These commands cover the three datasets under `/cfsdata/chushan/RoboCOIN-downloads`:
`Galaxea_R1_Lite_classify_object_four`,
`Galaxea_R1_Lite_mix_red_yellow_large_test_tube`, and `R1_Lite_stack_baskets`.
The audited snapshot has 342 episodes and 238,864 frames/windows. All three
datasets now contain the audited flat 14D conversion.

Run from this repository with the existing `future-libero` environment. Keep the
PyTorch/CUDA stack and VAE numerical settings fixed throughout a precompute run.
Indexed training can read the completed latents in another execution environment.
All assets are loaded locally.

```bash
export PYTHONNOUSERSITE=1 PYTHONPATH="$PWD"
export DIFFSYNTH_MODEL_BASE_PATH=/data/chushan/rift-assets/checkpoints
export DIFFSYNTH_SKIP_DOWNLOAD=true HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
PY=/data/chushan/miniconda3/envs/future-libero/bin/python
DATA=/cfsdata/chushan/RoboCOIN-downloads
```

## Multitask training view and normalization statistics

All three tasks train together, using the Galaxea dataset/processor contract.
The converted source vectors are `[left arm 6, right arm 6, left gripper, right
gripper]`. The training view exposes the existing named fields in policy order
`left_arm`, `left_gripper`, `right_arm`, `right_gripper` (6+1+6+1), and aliases
three video streams to `head_rgb`, `left_wrist_rgb`, `right_wrist_rgb`.

Grippers use the selected **millimetre stroke** convention, matching Galaxea's
0–100 mm interface:

| Source dataset | View action/state gripper |
| --- | --- |
| `Galaxea_R1_Lite_classify_object_four` | `raw * (180 / pi)` |
| `Galaxea_R1_Lite_mix_red_yellow_large_test_tube` | `raw * (180 / pi)` |
| `R1_Lite_stack_baskets` | unchanged |

The published [conversion configuration](https://github.com/RogersPyke/robocoin-dataset/blob/ec951eb78c34aca04d160543a4b6d03408f0cad6/scripts/format_converters/tolerobot/configs/converter_config_galaxea_r1_lite_mcap.yaml#L75)
applies `degree2rad` to left/right gripper feedback and targets; its
[implementation](https://github.com/RogersPyke/robocoin-dataset/blob/ec951eb78c34aca04d160543a4b6d03408f0cad6/src/robocoin_dataset/format_converter/utils/spatial_data_convertor.py#L65)
uses `np.deg2rad`. Multiplying by `180 / pi` reverses that numerical scaling,
rather than converting an angle into a physical length. This is the selected
interpretation for the first two datasets, supported by the code and observed
values; no per-subset publication log establishes that conversion history.

Only raw gripper columns 12 and 13 are scaled, for both action and state.
Arm joints remain unchanged, and no values are clipped. The source annotation
denominators 1.75, 1.8 and 100 only validate `gripper_open_scale_state`; they are
not conversion factors. Classification's `gripper_open_scale_action` differs
from its raw action and is **not** substituted.

Build a separate training view; source files and videos remain intact. Videos
are linked, while numerical Parquet records and their metadata are written and
verified independently. The view keeps equivalent packed and named vectors;
normalization generation verifies they agree exactly. The builder records
source hashes, mappings and counts, and refuses to overwrite an existing view.

```bash
VIEW="$DATA/training-views/galaxea_mm_v1"
"$PY" scripts/prepare_robocoin_training_view.py \
  --source-root "$DATA" --output-root "$VIEW"

CUDA_VISIBLE_DEVICES= "$PY" scripts/precompute_robocoin_norm_stats.py \
  --dataset-dir \
    "$VIEW/Galaxea_R1_Lite_classify_object_four" \
    "$VIEW/Galaxea_R1_Lite_mix_red_yellow_large_test_tube" \
    "$VIEW/R1_Lite_stack_baskets" \
  --output "$DATA/stat-norm/robocoin_multitask_mm/dataset_stats.json"

CUDA_VISIBLE_DEVICES= "$PY" scripts/verify_robocoin_norm_stats.py \
  --dataset-dir \
    "$VIEW/Galaxea_R1_Lite_classify_object_four" \
    "$VIEW/Galaxea_R1_Lite_mix_red_yellow_large_test_tube" \
    "$VIEW/R1_Lite_stack_baskets" \
  --stats "$DATA/stat-norm/robocoin_multitask_mm/dataset_stats.json" \
  --rift-repo "$PWD" --fastwam-repo /data/chushan/FastWAM \
  --output "$DATA/stat-norm/robocoin_multitask_mm/validation.json"
```

The CPU statistics computation uses the existing Galaxea processor and
`BaseLerobotDataset.get_dataset_stats` unchanged. It includes all 342 episodes /
238,864 frames, with no validation holdout. Action windows contain 32 steps
beginning at the current frame and repeat the last frame at episode boundaries.
Only arm actions are relative to the first observed joint state. The original
episode weighting, variance correction and quantile aggregation are preserved.
The verifier independently checks means, standard deviations and extrema with
NumPy, plus RIFT/FastWAM normalizer equality on real windows. FastWAM-Joint uses
the same FastWAM data processor.

Use `data=robocoin_multitask` in RIFT. The config defaults to `galaxea_mm_v1`
and `stat-norm/robocoin_multitask_mm/dataset_stats.json`, pins the three episode
counts, loads the existing T5 cache, and preserves Galaxea's
image resizing, composition, arm transforms, stepwise action normalization and
state z-score normalization. Absolute gripper actions use the existing `0/100`
mode. Environment overrides are `ROBOCOIN_TRAIN_ROOT`, `ROBOCOIN_NORM_STATS` and
`ROBOCOIN_TEXT_CACHE`.

Verify real RIFT and FastWAM dataset outputs, including the original VAE input
keys, without loading a policy or VAE model:

```bash
CUDA_VISIBLE_DEVICES= "$PY" scripts/verify_robocoin_training_view.py \
  --rift-repo "$PWD" --fastwam-repo /data/chushan/FastWAM \
  --vae-run-dir "$DATA/vae-cache/_precompute/20260921T111155Z" \
  --vae-cache-dir "$DATA/vae-cache" \
  --output "$DATA/stat-norm/robocoin_multitask_mm/training_validation.json"
```

This checks first/middle/final windows in each task's first episode. The script
translates RIFT's processor targets for the FastWAM package and removes
unsupported RIFT-only performance/cache options. Tensor and text
outputs must remain equal. This verifies the shared dataset contract; it does
not add VAE cache support to FastWAM.

All windows are shuffled together as in the previous Galaxea run; this gives
approximately 64.2% classification, 11.8% test tubes and 23.9% baskets, without
task balancing. With eight GPUs, global batch 256 and ten epochs, the current
trainer computes 9,340 optimizer steps. Override the old task's `max_steps` and
`model.anticip_anneal_total_steps` consistently; its old 14,830-step schedule
does not describe this dataset. Keep `resume=null` and load the pretrained base.
No policy training is launched by these precompute scripts.

Existing per-task files under `stat-norm/<source dataset>/dataset_stats.json`
retain raw source gripper units. The earlier `training-views/galaxea_percent_v1`
and `stat-norm/robocoin_multitask` outputs also remain for audit. Do not use those
stats with the millimetre training view. The stats script still supports a
single raw dataset, but refuses to mix raw datasets.

## T5 embeddings

The existing text precompute script reads the exact `task` strings from each
dataset's `meta/tasks.jsonl`. The three tasks produce three unique prompts,
using the same prompt template and UMT5-XXL assets as Galaxea training.

```bash
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/precompute_text_embeds.py \
  task=galaxea_indoor_cleaning_rift_3cam224_1e-4 \
  "data.train.dataset_dirs=[$DATA/Galaxea_R1_Lite_classify_object_four,$DATA/Galaxea_R1_Lite_mix_red_yellow_large_test_tube,$DATA/R1_Lite_stack_baskets]" \
  "data.train.text_embedding_cache_dir=$DATA/t5-cache" \
  +overwrite=false
```

Each file contains a bf16 `context` tensor of shape `[128,4096]` and a bool
`mask` of shape `[128]`. RIFT, FastWAM and FastWAM-Joint share this text cache
format. Training must use the same bare task text (`drop_high_level_prob=1.0`).

## VAE pixels and windows

The precompute script uses the original `RobotVideoDataset._get` composition
and normalization code. Before encoding, each camera goes through the original
float/uint8 round trip, `ToTensor`, and `Resize([224,224])`. The head view then
becomes a 256×320 top tile; the two wrist views become 128×160 bottom tiles.
The assembled 384×320 image goes through the original final resize, crop and
normalization to `[-1,1]`.

Camera order is head, left wrist, right wrist. The first two datasets use
`cam_head_left_rgb`; baskets use `cam_high_left_rgb`. Both wrist keys are
`cam_left_wrist_rgb` and `cam_right_wrist_rgb`, under `observation.images`.

Every frame is a window start. Video offsets are `[0,4,8,12,16,20,24,28,32]`,
with offsets past the episode end clamped to its last frame. This preserves
the existing 33-frame/32-action training convention on the new 30 Hz data.
Each nine-frame window is independently VAE encoded because causal VAE state
resets at each training window; whole-episode latents cannot be sliced instead.

The input module streams raw video once per episode and retains resized CPU
canvases. CPU preprocessing of the next episode overlaps GPU encoding of the
current episode. It does not change the VAE's per-sample encode operation.

## Generate and verify VAE latents

Choose idle GPUs before running. The following is a Bash example with seven
workers. Use the same source checkout, cache root and run directory throughout
a run. Keep logs and manifests in a persistent run directory.

```bash
CACHE="$DATA/vae-cache"
RUN="$CACHE/_precompute/$(date -u +%Y%m%dT%H%M%SZ)"
VAE="$DIFFSYNTH_MODEL_BASE_PATH/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"
ARGS=(--root "$DATA" --cache-dir "$CACHE" --run-dir "$RUN" --vae-weights "$VAE")
GPUS=(0 2 3 4 5 6 7)
mkdir -p "$RUN"

CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$PY" scripts/precompute_robocoin_vae.py \
  "${ARGS[@]}" --mode smoke > "$RUN/smoke.log" 2>&1 || exit 1

pids=()
for rank in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPUS[$rank]}" "$PY" scripts/precompute_robocoin_vae.py \
    "${ARGS[@]}" --mode worker --rank "$rank" --world-size "${#GPUS[@]}" \
    --batch-size 4 --cpu-threads 4 > "$RUN/worker_rank$rank.log" 2>&1 &
  pids+=("$!")
done
worker_failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || worker_failed=1
done
(( worker_failed == 0 )) || exit 1

"$PY" scripts/verify_robocoin_vae_cache.py --run-dir "$RUN" --cache-dir "$CACHE"
```

The smoke check compares the first, middle and final window of one episode
per dataset against the original video decoder and full 33-frame image
processor, byte for byte. It also compares original VAE outputs with cache
outputs. Each worker requires cache hits on six shared examples and checks
fresh encodes against them on its assigned GPU.

The cache stores bf16 `[48,3,24,20]` tensors. Keys hash the original CPU float32
`[3,9,384,320]` input, before bf16 conversion. The namespace binds the VAE
weights, normalization scale, source, class, software stack and numerical
settings. The validated DeepSpeed training context uses bf16 weights/inputs
with CUDA autocast disabled; do not enable bf16 autocast for preprocessing.

Each write checks the stored latent checksum and equality; workers also check
all encoded outputs for finite values. Final verification checks all episode
indices, every referenced file, worker completion and cross-GPU parity, then
loads representative files for independent tensor/checksum validation. It
writes `validation.json` under the run directory and `precompute_manifest.json`
under the cache root. This verifier deliberately pins the current three-task
snapshot; update its audited expected counts for a different dataset snapshot.

Re-run the same worker command to resume: completed episodes are checked
against their index checksums and cache payloads; partial episodes reuse valid
content-addressed entries. Changes to metadata, source scripts, cache root,
video/Parquet size or timestamps are rejected. Keep the same worker count for
the same run directory so completion reports remain unambiguous.

Enable direct indexed loading in RIFT with these overrides:

```text
++vae_cache_dir=/cfsdata/chushan/RoboCOIN-downloads/vae-cache
data.train.vae_cache_index_dir=/cfsdata/chushan/RoboCOIN-downloads/vae-cache/_precompute/20260921T111155Z
```

Pass the cache root, not the namespace subdirectory, and the precompute run
directory, not its `indices` subdirectory. The reader routes the actual task,
episode and start frame through the saved index, then reads the latent on CPU.
Training bypasses video decoding, resize, camera composition, pixel transfer,
pixel hashing and VAE encoding. Action/state normalization, T5 inputs and padding
are unchanged. Startup checks every index checksum, original video fingerprint,
camera order, timestamp mapping and image settings. Each latent read checks its
checksum, shape, dtype and finite values; missing or stale entries fail without
substituting a different sample. The loaded VAE weights, normalization scale and
input dtype are checked after model restore. Indexed training reads the existing
namespace directly, so it can use a different encoder package or execution
environment with the same frozen VAE. Online encoding retains the full numerical
identity checks.

Leave `data.train.vae_cache_index_dir=null` to use the pixel-hash path. No latent
files need regeneration for the indexed reader.

For five epochs from the pretrained base on eight A800s (BS16, accumulation 2,
global batch 256), the training command is:

```bash
bash scripts/train_zero2.sh 8 \
  task=galaxea_indoor_cleaning_rift_3cam224_1e-4 data=robocoin_multitask \
  num_epochs=5 max_steps=4670 gradient_accumulation_steps=2 resume=null \
  "++vae_cache_dir=$DATA/vae-cache" \
  "data.train.vae_cache_index_dir=$DATA/vae-cache/_precompute/20260921T111155Z" \
  wandb.name=robocoin_multitask_rift_indexed_5ep
```

Use the existing `future-libero` environment and local model asset variables
above, including `ACTION_DIT_PRETRAINED_PATH` for the pretrained action model.
For ten epochs use `num_epochs=10 max_steps=9340`. The annealing schedule follows
`max_steps` through the existing Galaxea task configuration. The original
FastWAM and FastWAM-Joint packages can reuse the same cache through the
[baseline input bridge](fastwam_indexed_latents.md).
