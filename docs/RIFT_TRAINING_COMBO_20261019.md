# Native RIFT training acceleration: combined benchmark

Status: completed on RTX 3090. The final combination reduced warmed training-step
time by **22.68–22.76% against original default execution**, and by
**6.43–6.50% against upstream compilation plus path fusion**, on the reduced native
RIFT fixture below. These are compute-step measurements, not full-5B or policy
success-rate results.

Upstream control: `cc5fce8d6f89210765e02c73ba692f01bc5b4b41` from
[ChushanZhang/RIFT](https://github.com/ChushanZhang/RIFT).

## What is being compared

This experiment measures a complete combination, with two reference executions
of the same native RIFT architecture and initialized weights. It does not measure
each optimization separately.

| Execution option | Original defaults | Upstream compile + fusion | Complete combination |
| --- | --- | --- | --- |
| Compile MoT training layers | Off | On | On |
| Fuse the two compatible training paths along the batch axis | Off | On | On |
| CUDA fused AdamW | Off | Off | On |
| Apply the video output head only to future-frame tokens | Off | Off | On |
| Compile the conditional future flow-matching decoder | Off | Off | On |

The first two options already exist upstream. The last three transfer the
corresponding LAL optimizations into native RIFT. Both reference modes retain
the same single-rank ZeRO-1 runtime as the candidate.

Repository configuration defaults remain `false` for these optional execution
settings so the original control remains available. The benchmark explicitly
enables the whole candidate combination. Its relevant training configuration
overrides are:

```yaml
fused_adamw: true
model:
  compile_training_denoise: true
  fuse_training_paths: true
  video_future_head_only: true
  compile_future_head: true
```

The native-head optimization retains the full-head fallback for custom or
unsupported output heads. It omits only the first-frame output that the native
training objective already discards. It does not remove first-frame backbone
tokens or change their attention connections.

LAL's shared first-frame backbone execution has not been transferred. RIFT uses
separate video and action experts with mixed attention, so that change needs a
separate implementation and validation. Action-row skipping, predetermined
perturbation masks, and removal of the diagnostic probe are also outside this
experiment. The probe remains trainable because its gradients contribute to
global gradient clipping even though its input is detached. Online VAE batching
and VAE CUDA graphs are not exercised by the resident-latent benchmark.

## Model and data fixture

The default fixture contains **474,971,975 trainable parameters** and uses the
native `WanVideoDiT`, `ActionDiT`, `MoT`, and `RIFTModel.training_loss`.

| Component | Benchmark configuration |
| --- | --- |
| Video expert | Hidden width 1536; feed-forward width 7168 |
| Action expert | Hidden width 512; feed-forward width 2048 |
| Both experts | 9 layers; 12 attention heads of width 128 |
| Text / timestep embeddings | Input dimensions 256 / 256 |
| Future flow-matching decoder | Width 512; 2 blocks |
| Proprioception | 8 dimensions, projected to the text input width |
| Video latents | Batch × 48 × 3 × 14 × 28 |
| Video token grid | 294 total tokens; 196 future tokens |
| Actions | Batch × 32 × 7 |
| Text context | Batch × 128 × 256 |
| Original video metadata | 9 frames, 224 × 448 pixels |

The fixture preserves the native 3:1 ratio of video to action hidden widths,
128-dimensional attention heads, seven-dimensional action flow matching, and
full-grid anticipation. It reduces depth and widths to fit an RTX 3090. This is
not the full pretrained 5B configuration.

Weights are synthetic. The video expert uses its native random initialization.
The action backbone is initialized from that reduced video expert with the
upstream sequential linear interpolation and alpha-scaling rule; the action
input and output layers retain their native random initialization. This matches
the initialization procedure, not the pretrained Wan weights or their statistics.

Synthetic data are resident on the selected GPU and enter through the native
indexed-latent path. Original video-shape metadata stay on CPU. No VAE or dataset
assets are downloaded. The VAE fixture raises if encoding is attempted, making
accidental inclusion of an online-encoding path visible.

For batch size four, valid action counts are 32, 16, 1, and 0. Image and context
masks cover partial and fully padded cases. Padding values are finite. A private
CPU generator creates data without changing the training RNG stream.

## Native training behavior retained

- Both experts, anticipation tokens, future decoder, diagnostic probe, and
  proprioception encoder remain trainable. VAE and text-encoding components
  remain frozen. The native mixed train/eval modes are preserved.
- Gradient checkpointing stays enabled in both experts, mixed attention, and the
  future decoder. The compiled implementation retains this setting.
- Video and action timestep shifts remain 5, with the native timestep loss
  weighting, padding reduction, and seven-dimensional action noise and target.
- The curriculum begins at step 21,700. Cases use condition-perturbation
  probabilities 0 and 0.3; the late curriculum has sigma 0.06 and final future-loss
  weights 0.2. Perturbed samples retain native action-loss exclusion.
- Native random draws and the native metric synchronization remain in the
  measured training path. No batch-mask planning replaces these draws.
- AdamW uses learning rate `1e-4`, weight decay `0.01`, betas `(0.9, 0.95)`, and
  global gradient clipping at `1.0`. Both variants use BF16 model parameters and
  single-rank ZeRO-1 with FP32 master parameters. No optimizer offloading is used.

The benchmark holds learning rate fixed. It excludes the production learning-rate
scheduler, dataset loading, online VAE encoding, text encoding, checkpoint writes,
and experiment logging.

## Reproduction and timing

Run from the repository root in the CUDA-enabled environment. Confirm the printed
CUDA device name is the RTX 3090; CUDA and `nvidia-smi` enumeration can differ.

```bash
conda run -n libero --no-capture-output python scripts/benchmark_training_combo.py \
  --device cuda:0 \
  --video-hidden 1536 --action-hidden 512 \
  --video-ffn 7168 --action-ffn 2048 \
  --layers 9 --heads 12 --head-dim 128 \
  --text-dim 256 --context-length 128 --batch-size 4 --seed 2731 \
  --references original upstream_accelerated \
  --probabilities 0 0.3 \
  --parity-steps 4 --warmup 12 --repeats 60 \
  --output docs/benchmarks/rift-training-combo-20261019/large3090.json
```

The completed run contains four cases: two references, each with both perturbation
probabilities. Each case compares identical initial model states and identically
seeded training steps. Numerical prechecks compare forward losses, gradients,
CPU/CUDA RNG consumption, and four actual optimizer updates including master
parameters and optimizer moments. Model comparisons report backbone and head
groups separately. AdamW step counters are compared on CPU because native and
fused implementations store them on different devices.

Compilation and numerical checks are outside the timed region. Each variant
receives forced perturbation/no-perturbation warmup, then 12 regular warmup steps,
followed by 60 paired measurements. The first-running variant alternates between
pairs. Timed work includes forward, backward, gradient clipping, the optimizer
update, and gradient clearing. Native forward metric synchronization remains
included.

The report records wall-clock and CUDA-event times, every raw pair, paired median
percentage changes, bootstrap confidence intervals, order-separated medians, and
GPU activity. Negative percentage change means lower training-step time. Memory
peaks describe the timed phase with both reference and candidate engines resident;
they are not a one-engine training-memory measurement.

## Compilation and ZeRO compatibility

The first GPU pilots found a real compatibility issue: ZeRO flattens parameter
storage after the native compile warmup, and the seven-element action bias leaves
later future-decoder parameters unaligned. PyTorch 2.7 could reuse a kernel
compiled for aligned weights. Invalidating Dynamo alone was insufficient because
the AOTAutograd cache did not retain the storage offsets in this path.

The future decoder now has a private compiler entry point and checks parameter
alignment using CPU metadata. A layout change triggers guarded recompilation;
only that compilation bypasses the problematic AOTAutograd cache. Inductor's
cache stays enabled. Compiler initialization preserves the training RNG stream.
Input shape, stride, dtype, gradient requirements, and autocast transitions use
the same guarded path, including the native loader's smaller final batch.

Actual CUDA regression tests cover FP32 and BF16, gradient checkpointing, aligned
weights followed by an offset-seven flat allocation, and batch sizes 4 → 1 → 4.
Both passed, including RNG and cache-setting restoration. FP32 compares individual
outputs and gradients with tight tolerances. BF16 uses the benchmark's 2% relative
L2 bound and loss tolerances because fusion changes intermediate rounding; it
does not promise elementwise or bitwise identity near zero.

## Results and validation

Final-code results, 60 paired measurements per row:

| Reference execution | Perturbation probability | Reference ms | Combined ms | Paired time change | 95% bootstrap interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original defaults | 0 | 304.43 | 235.46 | −22.68% | [−22.75%, −22.58%] |
| Original defaults | 0.3 | 303.92 | 234.98 | −22.76% | [−22.83%, −22.62%] |
| Upstream compile + fusion | 0 | 251.15 | 234.66 | −6.50% | [−6.56%, −6.42%] |
| Upstream compile + fusion | 0.3 | 252.11 | 235.62 | −6.43% | [−6.56%, −6.36%] |

Times are separate wall-clock medians; percentage changes are medians of paired
ratios. All 240 measured pairs favored the combination. Both alternating-order
subgroups also favored it in each case. Confidence intervals use 2,000 ordinary
paired bootstrap resamples and do not model serial correlation.

The final run uses PyTorch **2.7.1+cu128** and DeepSpeed **0.18.5**. The repository's
dependency file specifies DeepSpeed 0.19.1; this experiment validates the available
0.18.5 runtime. The RTX 3090 compute-process snapshots taken before and after each
timed case contain only the benchmark process; this is endpoint evidence, not
continuous process monitoring. The largest timed-phase allocation was
**15.978 GiB allocated / 16.930 GiB reserved**, with both engines resident.

Numerical validation passed using the tolerances declared before timing:

| Comparison | Largest observed difference |
| --- | ---: |
| Same-state forward loss, relative | 0.02845% |
| Same-state component gradient, relative L2 | 1.50573% |
| Forward loss during four checked updates, relative | 0.06613% |
| FP32 master parameters during checked updates, relative L2 | 0.03007% |
| First / second optimizer moments during checked updates, relative L2 | 1.02055% / 1.94629% |
| BF16 model component during checked updates, relative L2 | 0.08775% |
| Last measured forward loss, relative | 0.04403% |
| Final model component, relative L2 | 1.92229% diagnostic probe |
| Final component excluding diagnostic probe, relative L2 | 0.86740% anticipation tokens |

CPU and CUDA RNG states matched in the same-state and all four update prechecks.
Every variant completed 78 optimizer updates: four checked updates, two forced
warmups, 12 regular warmups, and 60 timed updates. The last reported forward loss
is the loss **used by update 78, after 77 prior updates**; final model differences
are measured after all 78 updates. The initial same-state precheck also advances
the native curriculum once without updating weights. All cases remain past the
curriculum's annealing endpoint.

Validation commands completed with **89 CPU tests and 37 subtests passed**, plus
**2 real-CUDA regression tests passed**. CPU FP32 checks compare losses, gradients,
RNG, fallback behavior, parameter keys, and compiler transitions; CUDA tests cover
both FP32 and BF16 relocation and final-batch behavior. `git diff --check` passed.

```bash
/home/isaac/anaconda3/envs/libero/bin/python -m pytest -q tests
conda run -n libero --no-capture-output python -m pytest -q \
  tests/test_training_acceleration.py::test_cuda_future_compiler_survives_flat_parameter_relocation_and_batch_changes
```

Artifacts:

- [Final raw measurements and prechecks](benchmarks/rift-training-combo-20261019/large3090.json).
- [Compact numerical and performance summary](benchmarks/rift-training-combo-20261019/summary.json).
- [Initial fixed-shape run](benchmarks/rift-training-combo-20261019/fixed_shape_initial.json),
  before adding the last-batch specialization guard: 22.64–22.73% less time against
  original execution and 6.48–6.58% less against upstream acceleration. The final
  table uses the repeated measurement of the final code, not this earlier run.
- Pilot failures and the short successful diagnostic are retained in
  `benchmarks/rift-training-combo-20261019/diagnostics/`; they are not the formal
  performance results.

BF16 tolerances establish bounded numerical agreement for the checked execution;
they do not establish bitwise equivalence or identical long-run training. This
benchmark cannot establish pretrained policy success rates, full-5B speedups, or
end-to-end throughput including data loading and VAE encoding. Those require
separate measurements in the corresponding training and evaluation setup.
