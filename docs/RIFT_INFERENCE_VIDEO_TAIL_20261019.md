# Action inference: retain final video K/V, omit unused visual outputs

Status: complete. The native action-inference shortcut is enabled by default.
On RTX 3090 with the full 6.03B architecture, prefill time fell 2.82%; complete
10/20-step action-call time fell 0.82%/0.44% in paired measurements. Cached tensors,
final actions, and checked RNG states were identical.

Reference: [deft-wam's final-layer prefill shortcut](https://github.com/ChushanZhang/deft-wam/blob/51fe00398bad853d73e409ccda4ef4236100fcf5/rift/models/wan22/mot.py#L434).
The fresh reference checkout is at revision `51fe0039`; the original RIFT control
comes from `cc5fce8`.

## Dependency and scope

RIFT action inference computes video K/V once, then reuses that cache for every
action denoising step. At video layer L, its K/V are projected from the **input to
layer L**. The action branch reads those K/V, not the updated visual tokens that
would leave layer L. No later video layer exists to consume that final output.

The final video layer therefore retains:

- Input normalization and time-dependent shift/scale.
- Key projection, key normalization, and rotary position encoding.
- Value projection and the final cache entry.

It omits the visual query projection/normalization/rotary encoding, visual
self-attention, its output projection, text cross-attention, and feed-forward
network. Earlier video layers remain complete because they produce the next
layer's input. The action expert still executes every layer on every denoising
step.

`MoT.optimize_video_prefill` defaults to `True`. The shortcut requires evaluation
mode and disabled gradients. Training and gradient-enabled prefill keep their
original execution. Joint video generation is unchanged; it needs the final
visual states. Video and future flow-matching training losses also still require
those states. Parameters and checkpoint keys are unchanged.

The supported equivalence claim concerns the native deterministic RIFT modules.
Instrumentation on skipped modules observes fewer calls; arbitrary custom
stateful modules or hooks are not part of the equivalence claim. Set
`model.mot.optimize_video_prefill = False` to run the complete original prefill.
This is a faithful original control: in deft-wam, disabling its optimization flag
still retains a final-layer early break, so that flag alone would not reproduce
original RIFT's redundant execution.

## Benchmark scope and reproduction

The runner toggles the flag on one model with identical weights. It compares
both all cached tensors and final actions with **zero tolerance**, plus CPU/CUDA
RNG states, before timing. Timings use alternating paired order, 12 warmup pairs,
and 60 measured pairs for each scope:

1. Video prefill alone, from already prepared video tokens/context.
2. Native `infer_action`, including video preparation/prefill, 10 action denoising
   steps, scheduling, and CPU action transfer.
3. The same native call with 20 denoising steps.

Actual VAE image encoding and text encoding are excluded: a resident first-frame
latent fixture and precomputed text context are used. No pretrained weights or
policy-success evaluation are involved. The benchmark keeps the upstream eager
inference execution; it adds only this terminal video-layer shortcut. The earlier
training acceleration combination does not compile this inference path.

The full architecture uses the native dimensions from `configs/model/rift.yaml`:

| Component | Parameters / dimensions |
| --- | --- |
| Video expert | 4,999,787,712 parameters; width 3072, FFN 14336 |
| Action expert | 1,020,886,023 parameters; width 1024, FFN 4096 |
| Anticipation, diagnostic probe and future FM head | 9,599,360 parameters |
| Proprioception projection | 36,864 parameters |
| Total | **6,030,309,959 parameters** |
| Both experts | 30 layers; 24 attention heads × 128 dimensions |
| Input | B1, 294 video tokens, 196 future tokens, 32 × 7 actions |
| Text context | 128 tokens, input width 4096 |

BF16 parameter storage is approximately 11.232 GiB. This is the full 5B video
backbone plus its roughly 1B action expert, not a 5B-total model. Random video
weights and upstream interpolation/alpha-scaled action-backbone initialization
are used. There is no optimizer or training state in this inference test.

```bash
conda run -n libero --no-capture-output python scripts/benchmark_inference_video_tail.py \
  --video-hidden 3072 --action-hidden 1024 \
  --video-ffn 14336 --action-ffn 4096 \
  --layers 30 --heads 24 --head-dim 128 --text-dim 4096 \
  --steps 10 20 --warmup 12 --repeats 60 \
  --output docs/benchmarks/rift-inference-tail-20261019/full_architecture3090.json
```

The JSON records all raw pairs, wall/CUDA-event medians, paired-ratio changes,
ordinary 2,000-resample bootstrap intervals, execution-order strata, memory peaks,
and GPU activity snapshots. These bootstrap intervals do not model serial
correlation. Initialization, correctness checks, and warmup are excluded.

## Results and validation

Full-size measurements on RTX 3090, PyTorch 2.7.1+cu128, BF16, B1:

| Scope | Original median | Optimized median | Paired time change | 95% bootstrap interval |
| --- | ---: | ---: | ---: | ---: |
| Video prefill | 70.80 ms | 68.81 ms | −2.82% | [−2.84%, −2.81%] |
| Native action call, 10 steps | 253.80 ms | 252.17 ms | −0.82% | [−0.95%, −0.69%] |
| Native action call, 20 steps | 439.04 ms | 436.68 ms | −0.44% | [−0.77%, −0.28%] |

Wall-time medians are calculated separately. Percentage changes are medians of
paired ratios, so they need not equal the ratio of those two independent medians.
Both execution-order subgroups favored the shortcut in all three scopes.
Prefill was faster in 60/60 pairs, the 10-step call in 51/60, and the 20-step call
in 44/60. The small whole-call gain does not imply every individual request is
faster; runtime noise exceeds the saving in some pairs.

Full-size CUDA prechecks found exact equality for every layer's cached K/V and
the final action tensors at both denoising lengths. CPU/CUDA RNG states matched.
The largest measured allocated/reserved GPU memory was **11.446/11.555 GiB**;
these are process peaks across both flag settings, not per-variant memory savings.
GPU activity snapshots show only the benchmark compute process on the 3090.

[Raw full-size measurements](benchmarks/rift-inference-tail-20261019/full_architecture3090.json)
contain the exact-output checks and all 180 pairs. The fixture metadata describes
configured model components; the explicit execution metadata clarifies that
checkpoint recomputation, the diagnostic probe, future FM head, and optimizer
are inactive in this inference test.

Six new CPU tests pass, covering FP32/BF16, one/two layers, exact cache and action
velocity agreement, complete native multi-step action agreement, RNG, checkpoint
keys, actual skipped/retained calls, gradient/training fallback, gradient parity,
and invalid masks. The combined inference/training regression set passed
**51 tests and 24 subtests**; two existing CUDA tests skipped in the CPU environment.

The reduced 475M/9-layer pilot is retained in
`benchmarks/rift-inference-tail-20261019/reduced_pilot.json`. Its 12-pair short run
found about 9.14% lower prefill time, but its 10/20-step whole-action-call confidence
intervals crossed zero. It is diagnostic evidence, not the full-size result.

This optimization saves work once per action request. It cannot be multiplied by
the number of denoising steps. For a 30-layer video branch, it saves less than one
full video layer; any end-to-end gain is diluted by the action loop and excluded
image/text encoding.
