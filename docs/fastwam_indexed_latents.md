# Indexed RoboCOIN inputs for FastWAM baselines

`rift.baselines.fastwam` connects the original FastWAM and FastWAM-Joint models
to the existing RoboCOIN latent-window index. Model construction, attention,
loss, optimizer, sampler and checkpoint handling remain in the original
FastWAM checkout at `/data/chushan/FastWAM`.

Both baselines read the cache namespace recorded in the VAE index directly.
The loaded VAE weights, normalization scale and input dtype must match the
precomputed latents. Package names, encoder source and runtime differences do
not block indexed reads. Indexed training only reads existing latents; online
encoding uses its own cache namespace. Payload checksums, shapes and dtypes
remain validated.

The same three mm training views, joint normalization stats, T5 embeddings and
238,864 indexed windows are used. Production settings remain unchanged:
8×A800, bf16, six workers/rank and ZeRO-1. FastWAM uses BS32/GAS1;
FastWAM-Joint uses BS16/GAS2. Both use global batch 256 and 934 optimizer steps
per epoch.

## Prepare and launch

```bash
BASELINE=/data/chushan/FastWAM
RIFT_REPO=/data/chushan/RIFT/code/rift-libero-plus-main
PY=/data/chushan/miniconda3/envs/fastwam-hra/bin/python
export PYTHONPATH="$RIFT_REPO:$BASELINE/src" PYTHONNOUSERSITE=1
export CUDA_HOME=/data/chushan/miniconda3/envs/omega-flow
export DIFFSYNTH_MODEL_BASE_PATH=/data/chushan/rift-assets/checkpoints
export ACTION_DIT_PRETRAINED_PATH="$DIFFSYNTH_MODEL_BASE_PATH/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
export DIFFSYNTH_SKIP_DOWNLOAD=true HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG=/data/chushan/fastwam-runs/configs/robocoin_fastwam_5ep.yaml
"$PY" "$RIFT_REPO/scripts/prepare_fastwam_indexed_config.py" \
  --fastwam-repo "$BASELINE" --baseline fastwam --epochs 5 --output "$CONFIG"

cd "$BASELINE"
/data/chushan/miniconda3/envs/fastwam-hra/bin/accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_processes 8 \
  "$RIFT_REPO/scripts/train_fastwam_indexed.py" --config "$CONFIG"
```

For Joint use `--baseline fastwam_joint`; for ten epochs use `--epochs 10`.
Generate separate config and output paths for each run. The helper sets
`resume=null`, disables evaluation/WandB and computes the step budget from the
dataset metadata. Indexed evaluation is not supported.
