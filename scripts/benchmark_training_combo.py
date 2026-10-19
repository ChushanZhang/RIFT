#!/usr/bin/env python3
"""Paired CUDA training benchmark for native RIFT and an opt-in acceleration bundle.

Uses resident synthetic indexed latents and a reduced native two-expert model.
Measures complete warmed forward/backward/ZeRO-1 optimizer steps, not data
loading, online VAE encoding, compilation, or pretrained task quality.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import tempfile
import time

import torch
from torch import nn

from training_combo_fixture import DEFAULTS, make_model, make_batch, fixture_metadata
from rift.models.wan22.helpers.training_acceleration import stock_video_head
from rift.trainer import make_training_optimizer

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("reference", "combined")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in DEFAULTS.items():
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--references", nargs="+", choices=("original", "upstream_accelerated"),
                        default=["original", "upstream_accelerated"])
    parser.add_argument("--probabilities", nargs="+", type=float, default=[0., .3])
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=60)
    parser.add_argument("--parity-steps", type=int, default=4)
    parser.add_argument("--debug-cuda", action="store_true", help="Synchronize launches for failure diagnosis; never use reported timing.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.warmup, args.repeats, args.parity_steps, args.batch_size) < 1:
        parser.error("warmup, repeats, parity-steps and batch-size must be positive")
    if any(not 0 <= p <= 1 for p in args.probabilities):
        parser.error("probabilities must be in [0, 1]")
    return args


def configure(model, *, accelerated, combined, probability):
    model.compile_training_denoise = accelerated
    model.mot.compile_training_layers = accelerated
    model.fuse_training_paths = accelerated
    model.video_future_head_only = combined
    model.compile_future_head = combined
    model.anticip_cond_noise_p = probability


def rng_state(device):
    return torch.get_rng_state(), torch.cuda.get_rng_state(device)


def check_rng(states):
    if not all(torch.equal(a, b) for a, b in zip(*states)):
        equal = {name: torch.equal(a, b) for name, (a, b) in zip(("cpu", "cuda"), zip(*states))}
        raise RuntimeError(f"CPU/CUDA random-number consumption differs: {equal}")


def compare_tensors(reference, candidate, *, relative_limit=.02):
    """Stream full-tensor statistics in bounded GPU chunks outside timed steps."""
    if reference.keys() != candidate.keys():
        raise RuntimeError("Compared tensor names differ")
    totals = {}
    with torch.no_grad():
        for name, expected in reference.items():
            actual = candidate[name]
            if expected is None or actual is None:
                if expected is not actual:
                    raise RuntimeError(f"Gradient presence differs: {name}")
                continue
            if expected.shape != actual.shape:
                raise RuntimeError(f"Compared tensor shape differs: {name}")
            if "/" in name:
                category = name.rsplit("/", 1)[-1]
            else:
                plain_name = name.removeprefix("policy.")
                if plain_name.startswith("mot.anticip.head_fm."):
                    category = "future_fm_head"
                elif plain_name.startswith("mot.anticip.head."):
                    category = "diagnostic_probe"
                elif plain_name.startswith("video_expert.head."):
                    category = "video_output_head"
                elif plain_name.startswith("action_expert.head."):
                    category = "action_output_head"
                elif plain_name.startswith("mot.anticip."):
                    category = "anticipation_tokens"
                else:
                    category = plain_name.split(".")[0]
            if category == "step":
                # Native AdamW stores this scalar on CPU; fused AdamW uses CUDA.
                if not torch.equal(expected.detach().cpu(), actual.detach().cpu()):
                    raise RuntimeError("Optimizer step counters differ")
                totals.setdefault(category, dict(max_abs=0., squared_error=0., squared_reference=0.))
                continue
            row = totals.setdefault(category, dict(max_abs=0., squared_error=0., squared_reference=0.))
            # Tensor parameters are contiguous here; flattening never copies a full model.
            left, right = expected.detach().reshape(-1), actual.detach().reshape(-1)
            for start in range(0, left.numel(), 2**20):
                a, b = left[start:start + 2**20].float(), right[start:start + 2**20].float()
                delta = a - b
                values = torch.stack((delta.abs().max(), delta.square().sum(),
                                      a.square().sum(), b.square().sum())).double().cpu().tolist()
                if not all(math.isfinite(value) for value in values):
                    raise RuntimeError(f"Nonfinite compared tensor: {name}")
                row["max_abs"] = max(row["max_abs"], values[0])
                row["squared_error"] += values[1]
                row["squared_reference"] += values[2]
    for category, row in totals.items():
        row["relative_l2"] = math.sqrt(row.pop("squared_error") / max(row.pop("squared_reference"), 1e-30))
        if relative_limit is not None and row["relative_l2"] > relative_limit:
            raise RuntimeError(f"{category} relative L2 exceeds {relative_limit}: {row}")
    return totals


def initial_parity(models, batch, args, device):
    # As in native compiled-training warmup, compile before testing steady-state
    # RNG use. Compiler initialization is not a training operation.
    for model in models:
        saved_step = model._anticip_step
        with torch.random.fork_rng(devices=[device]):
            torch.manual_seed(args.seed + 1)
            loss, _ = model.training_loss(batch)
            loss.backward()
        model.zero_grad(set_to_none=True)
        model._anticip_step = saved_step
    losses, states = [], []
    for model in models:
        torch.manual_seed(args.seed + 2)
        loss, _ = model.training_loss(batch)
        loss.backward()
        losses.append(float(loss.detach()))
        states.append(rng_state(device))
    check_rng(states)
    torch.testing.assert_close(torch.tensor(losses[0]), torch.tensor(losses[1]), rtol=.001, atol=.0001)
    gradients = compare_tensors(*[{name: p.grad for name, p in model.named_parameters()}
                                  for model in models])
    for model in models:
        model.zero_grad(set_to_none=True)
    return dict(losses=losses, gradients=gradients, cpu_and_cuda_rng_equal=True,
                loss_rtol=.001, loss_atol=.0001, relative_l2_limit=.02)


class TrainingWrapper(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, batch):
        return self.policy.training_loss(batch)


def zero_config(batch_size):
    config = json.loads((ROOT / "scripts/ds_configs/ds_zero1_config.json").read_text())
    config.update(train_batch_size=batch_size, train_micro_batch_size_per_gpu=batch_size,
                  gradient_accumulation_steps=1, gradient_clipping=1.,
                  steps_per_print=2**31 - 1, bf16={"enabled": True},
                  zero_allow_untested_optimizer=True)
    return config


def make_engines(models, args, device):
    import deepspeed
    engines, details = [], []
    for index, model in enumerate(models):
        optimizer = make_training_optimizer(
            [p for p in model.parameters() if p.requires_grad], learning_rate=1e-4,
            weight_decay=.01, device=device, fused_adamw=bool(index),
        )
        engine, _, _, _ = deepspeed.initialize(
            model=TrainingWrapper(model), optimizer=optimizer,
            config=zero_config(args.batch_size), dist_init_required=False,
        )
        actual = engine.optimizer.optimizer
        partitions = [p for group in actual.param_groups for p in group["params"]]
        if (engine.zero_optimization_stage() != 1 or actual is not optimizer
                or not partitions or any(p.dtype != torch.float32 for p in partitions)):
            raise RuntimeError("Expected native AdamW with FP32 ZeRO-1 master parameters")
        if index and not all(group.get("fused") is True for group in actual.param_groups):
            raise RuntimeError("Candidate optimizer did not retain fused=True")
        if model.training or not model.dit.training or not model.proprio_encoder.training:
            raise RuntimeError("DeepSpeed changed the native trainer's mixed train/eval modes")
        if index and not stock_video_head(model.video_expert):
            raise RuntimeError("DeepSpeed preparation disabled future-only video head eligibility")
        engines.append(engine)
        details.append(dict(fused=[group.get("fused") for group in actual.param_groups],
                            foreach=[group.get("foreach") for group in actual.param_groups],
                            partition_shapes=[list(p.shape) for p in partitions],
                            partition_dtypes=[str(p.dtype) for p in partitions]))
    return engines, details


def step(engine, batch):
    loss, _ = engine(batch)
    engine.backward(loss)
    engine.step()
    return loss.detach()


def optimizer_tensors(engine):
    optimizer, state = engine.optimizer.optimizer, {}
    for gi, group in enumerate(optimizer.param_groups):
        for pi, p in enumerate(group["params"]):
            prefix = f"{gi}/{pi}"
            state[prefix + "/master_parameter"] = p
            for name, value in optimizer.state[p].items():
                state[prefix + "/" + name] = torch.as_tensor(value)
    return state


def update_parity(engines, batch, args, device):
    rows = []
    for index in range(args.parity_steps):
        losses, states = [], []
        for engine in engines:
            torch.manual_seed(args.seed + 30 + index)
            losses.append(float(step(engine, batch)))
            states.append(rng_state(device))
        check_rng(states)
        torch.testing.assert_close(torch.tensor(losses[0]), torch.tensor(losses[1]), rtol=.003, atol=.0002)
        rows.append(dict(update=index + 1, losses=losses, cpu_and_cuda_rng_equal=True,
                         optimizer=compare_tensors(*[optimizer_tensors(e) for e in engines]),
                         model=compare_tensors(*[dict(e.module.named_parameters()) for e in engines])))
    return dict(steps=rows, relative_l2_limit=.02, loss_rtol=.003, loss_atol=.0002)


def timed_step(engine, batch, device):
    events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    events[0].record()
    loss = step(engine, batch)
    events[1].record()
    torch.cuda.synchronize(device)
    return (time.perf_counter() - started) * 1000, events[0].elapsed_time(events[1]), loss


def bootstrap_ci(values, seed):
    rng = random.Random(seed)
    medians = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(2000))
    return [medians[50], medians[1950]]


def activity():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw",
                                 "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        processes = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
                                    "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        return dict(gpus=result.stdout.strip(), returncode=result.returncode,
                    compute_processes=processes.stdout.strip(), benchmark_pid=os.getpid(), loadavg=os.getloadavg())
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(error=str(error), loadavg=os.getloadavg())


def one_case(args, batch, reference_kind, probability, device):
    reference = make_model(args, device, torch.bfloat16)
    candidate = deepcopy(reference)
    models = [reference, candidate]
    configure(reference, accelerated=reference_kind == "upstream_accelerated", combined=False, probability=probability)
    configure(candidate, accelerated=True, combined=True, probability=probability)
    print(json.dumps(dict(event="initial_parity", reference=reference_kind, probability=probability)), flush=True)
    checked = initial_parity(models, batch, args, device)
    engines, optimizer_details = make_engines(models, args, device)
    print(json.dumps(dict(event="update_parity")), flush=True)
    updates = update_parity(engines, batch, args, device)
    for forced in (0., 1.):
        for engine in engines:
            engine.module.policy.anticip_cond_noise_p = forced
            torch.manual_seed(args.seed + 3)
            step(engine, batch)
            engine.module.policy.anticip_cond_noise_p = probability
    for index in range(args.warmup):
        for variant in ([0, 1] if index % 2 == 0 else [1, 0]):
            torch.manual_seed(args.seed + 100 + index)
            step(engines[variant], batch)
    if candidate.mot._compiled_joint_layer is None or candidate._future_head_execution.compiled is None:
        raise RuntimeError("Candidate did not execute both compiled components")
    torch.cuda.synchronize(device)
    # DeepSpeed initialization resets allocator peaks. Deliberately measure the
    # timed phase after both engines and optimizer states have been initialized.
    torch.cuda.reset_peak_memory_stats(device)
    before, raw, last_losses = activity(), [], [None, None]
    print(json.dumps(dict(event="timing", repeats=args.repeats)), flush=True)
    for index in range(args.repeats):
        order = [0, 1] if index % 2 == 0 else [1, 0]
        row = dict(pair=index, order=order)
        for variant in order:
            torch.manual_seed(args.seed + 10000 + index)
            wall, cuda, loss = timed_step(engines[variant], batch, device)
            row[NAMES[variant] + "_wall_ms"] = wall
            row[NAMES[variant] + "_cuda_ms"] = cuda
            last_losses[variant] = loss
        raw.append(row)
    changes = [(r["combined_wall_ms"] / r["reference_wall_ms"] - 1) * 100 for r in raw]
    result = dict(reference=reference_kind, probability=probability, fixture=fixture_metadata(args, reference),
                  precheck=checked, update_precheck=updates, optimizer=optimizer_details,
                  median_wall_ms={name: statistics.median(r[name + "_wall_ms"] for r in raw) for name in NAMES},
                  median_cuda_ms={name: statistics.median(r[name + "_cuda_ms"] for r in raw) for name in NAMES},
                  paired_median_time_change_percent=statistics.median(changes),
                  paired_median_time_change_95ci=bootstrap_ci(changes, args.seed),
                  order_strata_median_percent={str(first): statistics.median(change for r, change in zip(raw, changes) if r["order"][0] == first)
                                              for first in (0, 1) if any(r["order"][0] == first for r in raw)},
                  final_forward_losses=[float(value) for value in last_losses],
                  optimizer_updates_per_variant=args.parity_steps + 2 + args.warmup + args.repeats,
                  final_curriculum_steps=[model._anticip_step for model in models],
                  timed_phase_two_engine_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                  timed_phase_two_engine_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                  activity_before=before, activity_after=activity(), raw_pairs=raw)
    if not all(math.isfinite(value) for value in result["final_forward_losses"]):
        raise RuntimeError("Nonfinite final loss")
    result["final_model_drift"] = compare_tensors(*[dict(e.module.named_parameters()) for e in engines], relative_limit=None)
    for engine in engines:
        engine.destroy()
    return result


def main():
    args = arguments()
    if args.debug_cuda:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.set_num_threads(1)
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), torch_version=torch.__version__,
                  settings={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                  upstream_commit="cc5fce8", status="pending", cases=[],
                  scope="Reduced native RIFT with random interpolated weights, BF16, single-rank ZeRO-1, resident synthetic indexed latents; no online VAE or data loading.",
                  timing="Full warmed forward/backward/clipping/optimizer/gradient clearing; native metric sync retained; alternating paired order; initialization, compilation and parity checks excluded.",
                  comparison="Original default model execution or upstream MoT compile+path fusion versus the complete combination; both references use the same fixed ZeRO-1 runtime.",
                  optimizer_settings=dict(learning_rate=1e-4, weight_decay=.01, betas=[.9, .95], clipping=1.,
                                          learning_rate_scheduler_included=False),
                  numerical_scope="BF16 rounding tolerance checks are not bitwise equivalence or a pretrained task-quality evaluation.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    if not torch.cuda.is_available():
        result.update(status="blocked", reason="CUDA is unavailable in this execution environment.")
        save()
        raise SystemExit(result["reason"])
    device = torch.device(args.device)
    if device.type != "cuda" or int(os.environ.get("WORLD_SIZE", "1")) != 1 or torch.distributed.is_initialized():
        raise SystemExit("Run on CUDA in a fresh single-rank process without torchrun.")
    device = torch.device("cuda", device.index if device.index is not None else torch.cuda.current_device())
    torch.cuda.set_device(device)
    os.environ["LOCAL_RANK"] = str(device.index)
    result["device_name"] = torch.cuda.get_device_name(device)
    with tempfile.TemporaryDirectory(prefix="rift-combo-zero1-") as directory:
        try:
            torch.distributed.init_process_group("nccl", init_method=(Path(directory) / "rendezvous").as_uri(), rank=0, world_size=1)
            import deepspeed
            result.update(deepspeed_version=deepspeed.__version__, zero_config=zero_config(args.batch_size), status="running")
            save()
            batch = make_batch(args, device, torch.bfloat16)
            for reference in args.references:
                for probability in args.probabilities:
                    case = one_case(args, batch, reference, probability, device)
                    result["cases"].append(case)
                    save()
                    print(json.dumps({key: case[key] for key in ("reference", "probability", "median_wall_ms", "paired_median_time_change_percent", "paired_median_time_change_95ci")}), flush=True)
                    gc.collect()
                    torch.cuda.empty_cache()
            result["status"] = "complete"
            save()
        except Exception as error:
            result.update(status="failed", reason=f"{type(error).__name__}: {error}")
            save()
            raise
        finally:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
