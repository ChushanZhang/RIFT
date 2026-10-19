# Packed LIBERO VAE cache

LIBERO reads one BF16 safetensors file per episode. Every original start frame
selects a `[48, 3, 14, 28]` latent for the original nine-frame, two-camera
`224x448` clip. Final frames are repeated at episode boundaries. Action/state
loading, normalization and sample ordering stay unchanged.

Set `vae_cache_dir=/path/to/cache` to use a completed packed cache. The LIBERO
configuration resolves its `ready.json` automatically. Missing manifests,
incompatible preprocessing, and incomplete indices fail rather than falling
back to old per-window `.pt` files. `vae_cache_dir=null` uses online encoding.
An existing DEFT packed cache is compatible when it contains the same dataset
sources and frozen VAE weights, scale and BF16 dtype. Encoder package names and
runtime versions need not match.

```bash
bash scripts/train_zero2.sh "$N" task=libero_rift_2cam224_1e-4 \
  vae_cache_dir=/path/to/cache
```

## Build directly from the frozen VAE

Precomputation is still necessary to create a new cache. The following tools
write episode files directly, without intermediate per-window `.pt` latents.
Text embedding precomputation remains a separate required step.

```bash
cache_root=/path/to/cache
index_directory="$cache_root/_precompute/libero_run"
dataset_directories=(
  /path/to/libero_spatial_no_noops_lerobot
  /path/to/libero_object_no_noops_lerobot
  /path/to/libero_goal_no_noops_lerobot
  /path/to/libero_10_no_noops_lerobot
)
CUDA_VISIBLE_DEVICES=0 python scripts/precompute_libero_vae.py encode \
  --vae-weights /path/to/Wan2.2_VAE.safetensors \
  --cache-dir "$cache_root" --index-dir "$index_directory" \
  --dataset-dirs "${dataset_directories[@]}" --worker-index 0 --worker-count 1
```

For multiple GPUs, launch one process per GPU with distinct worker indices
`0..N-1` and the same `--worker-count N`. Use a new index directory. After all
workers finish, use their reported namespace:

```bash
cache_namespace=REPLACE_WITH_REPORTED_NAMESPACE
python scripts/precompute_libero_vae.py finalize \
  --cache-dir "$cache_root" --index-dir "$index_directory" \
  --namespace "$cache_namespace" --dataset-dirs "${dataset_directories[@]}"
# Export a resolved training configuration with your real dataset/text paths.
python scripts/train.py task=libero_rift_2cam224_1e-4 --cfg job --resolve > /tmp/rift-config.yaml
python scripts/verify_rift_training_cache.py \
  --config /tmp/rift-config.yaml --cache-dir "$cache_root" \
  --index-dir "$index_directory" --report "$index_directory/training_validation.json"
python scripts/precompute_libero_vae.py publish-ready \
  --cache-dir "$cache_root" --index-dir "$index_directory" --namespace "$cache_namespace"
python scripts/verify_libero_safetensors.py --cache-dir "$cache_root" \
  --report "$index_directory/storage_validation.json"
```

Keep the episode files, manifest, namespace `identity.json`, precompute indices,
and `ready.json`. Finalization checks all saved payload checksums. Training
validates metadata, keys, shapes, BF16 dtype and finite values, without hashing
each payload on every window read. Episode headers are validated lazily in each
DataLoader worker. The training-loader audit compares original pixel keys and
nonvisual outputs on first/middle/final starts of the first episode per suite;
it does not re-encode the VAE or claim a full-dataset policy-training test.
