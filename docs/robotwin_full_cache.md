# Batched RoboTwin VAE and text precompute

`scripts/precompute_robotwin_cache.py` prepares the original RoboTwin release:
27,500 episodes and 6,075,103 window starts. Each window uses video offsets
0,4,...,32, repeats the last frame at episode boundaries, and composes the
three cameras into a 384x320 canvas using the real training transforms.
Outputs are BF16 `[48,3,24,20]` latents, packed per episode into safetensors.

The encoder uses `OnlineVaeEncoder`, which sends a whole batch to
`vae.model.encode`. The default is native BF16, batch 16, without autocast.
`--encode-mode cudagraphs` enables the existing CUDA Graph wrapper. Tail
batches repeat their final window to the full batch size; padding is discarded
before writing. This keeps one GPU shape and one graph per worker.

Batching can change BF16 convolution rounding. The cache namespace includes
the encoding mode, batch size and tail policy, along with the frozen weights,
scales and input dtype. Sequential caches remain intact in their old namespace.
Changing encoding settings requires a separate run directory. Existing complete
episodes in the same run are validated and reused on restart. Per-worker file
locks prevent duplicate writers.

Run `init`, then `smoke`, then the worker ranks, and finally `publish`:

```bash
ARGS=(--root "$ROBOTWIN_DATA" --cache-dir "$ROBOTWIN_CACHE_DIR"
      --run-dir "$ROBOTWIN_RUN_DIR" --vae-weights "$ROBOTWIN_VAE"
      --world-size 10 --batch-size 16 --encode-mode batched)
python scripts/precompute_robotwin_cache.py init "${ARGS[@]}"
CUDA_VISIBLE_DEVICES=0 python scripts/precompute_robotwin_cache.py smoke "${ARGS[@]}"
# Launch exactly one process per rank, assigning GPUs explicitly.
CUDA_VISIBLE_DEVICES=0 python scripts/precompute_robotwin_cache.py worker "${ARGS[@]}" --rank 0
# After every rank reports complete:
python scripts/precompute_robotwin_cache.py publish "${ARGS[@]}"
```

Smoke validates first/middle/final window pixels against the training decoder,
compares batched outputs with the original sequential VAE, and checks the
safetensors round trip. The BF16 comparison requires normalized RMSE <= 1%
and maximum absolute error <= 0.125. Every worker must reproduce the optimized
smoke samples bit for bit on its GPU before processing episodes. Final
publication verifies every index, payload hash and source fingerprint.

Status files report committed windows and episodes, GPU peak memory, throughput,
and cumulative time spent waiting for input, collating, encoding and writing.

`scripts/precompute_robotwin_text.py` retains native English UMT5 embeddings:
921,032 prompts, context length 128, BF16 `[128,4096]` context and Boolean mask.
An already completed text run can be verified and reused without re-encoding.

After VAE publication, text verification and the real training-reader check,
use `task=robotwin_rift_3cam_384_cached` or
`task=robotwin_clean2random_rift_3cam_384_cached`. Set `ROBOTWIN_CACHE_DIR`,
`ROBOTWIN_RUN_DIR` and `ROBOTWIN_TEXT_DIR` to the validated cache locations.
Both presets share the text cache; Clean2Random selects its 2,500 Clean
episodes and uses its existing Clean-only normalization statistics.
