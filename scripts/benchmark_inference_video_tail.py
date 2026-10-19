#!/usr/bin/env python3
"""Compare original video prefill against terminal K/V-only action inference.

Uses one native RIFT model with identical random weights and toggles only the
prefill optimization. Full infer_action calls retain context/proprio processing,
the action scheduler, all denoising steps and CPU output transfer. A resident
latent VAE fixture excludes actual image encoding; no pretrained quality claim.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time

import torch
from torch import nn

from training_combo_fixture import DEFAULTS, make_model, make_batch, fixture_metadata
from benchmark_training_combo import activity, bootstrap_ci


class ResidentImageVAE(nn.Module):
    temporal_downsample_factor = 4

    def __init__(self, first_frame):
        super().__init__()
        self.register_buffer("first_frame", first_frame, persistent=False)

    def encode(self, video, **kwargs):
        if len(video) != 1 or tuple(video[0].shape) != (3, 1, 224, 448):
            raise ValueError("Expected one 224x448 RGB image")
        return self.first_frame


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in DEFAULTS.items():
        if name != "batch_size":
            parser.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.batch_size = 1
    if min(args.steps + [args.warmup, args.repeats]) < 1:
        parser.error("steps, warmup and repeats must be positive")
    return args


def timed(function, device):
    events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    events[0].record()
    output = function()
    events[1].record()
    torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * 1000, events[0].elapsed_time(events[1]), output


def measure(model, function, args, device, label):
    outputs, rng = [], []
    for enabled in (False, True):
        model.mot.optimize_video_prefill = enabled
        torch.manual_seed(args.seed + 3)
        outputs.append(function())
        rng.append((torch.get_rng_state(), torch.cuda.get_rng_state(device)))
    torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)
    assert all(torch.equal(a, b) for a, b in zip(*rng))
    del outputs
    for index in range(args.warmup):
        for enabled in ((False, True) if index % 2 == 0 else (True, False)):
            model.mot.optimize_video_prefill = enabled
            function()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    before, raw = activity(), []
    for index in range(args.repeats):
        order = [False, True] if index % 2 == 0 else [True, False]
        row = dict(pair=index, order=[int(value) for value in order])
        for enabled in order:
            model.mot.optimize_video_prefill = enabled
            wall, cuda, output = timed(function, device)
            prefix = "optimized" if enabled else "original"
            row[prefix + "_wall_ms"] = wall
            row[prefix + "_cuda_ms"] = cuda
            del output
        raw.append(row)
    changes = [(r["optimized_wall_ms"] / r["original_wall_ms"] - 1) * 100 for r in raw]
    return dict(scope=label, outputs_bitwise_equal=True, cpu_and_cuda_rng_equal=True,
                median_wall_ms={name: statistics.median(r[name + "_wall_ms"] for r in raw)
                                for name in ("original", "optimized")},
                median_cuda_ms={name: statistics.median(r[name + "_cuda_ms"] for r in raw)
                                for name in ("original", "optimized")},
                paired_median_time_change_percent=statistics.median(changes),
                paired_median_time_change_95ci=bootstrap_ci(changes, args.seed),
                order_strata_median_percent={str(first): statistics.median(change for r, change in zip(raw, changes)
                                                                          if r["order"][0] == first)
                                            for first in (0, 1) if any(r["order"][0] == first for r in raw)},
                faster_pairs=sum(value < 0 for value in changes), raw_pairs=raw,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                activity_before=before, activity_after=activity())


@torch.no_grad()
def main():
    args = arguments()
    torch.set_num_threads(1)
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), torch_version=torch.__version__,
                  status="pending", cases=[],
                  settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  scope="Native RIFT architecture with random interpolated weights; real BF16 CUDA inference. Resident first-frame latents and precomputed text context exclude VAE/text encoding; no task-quality evaluation.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable to this execution environment")
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        result["device_name"] = torch.cuda.get_device_name(device)
        print(json.dumps(dict(event="model_initialization", layers=args.layers, video_hidden=args.video_hidden)), flush=True)
        model = make_model(args, device, torch.bfloat16).eval()
        model.requires_grad_(False)
        batch = make_batch(args, device, torch.bfloat16)
        model.vae = ResidentImageVAE(batch["video_latents"][:, :, :1])
        result["fixture"] = fixture_metadata(args)
        result["execution"] = dict(model_eval=not model.training,
                                   gradient_enabled=torch.is_grad_enabled(),
                                   optimizer_present=False, checkpoint_recomputation=False,
                                   diagnostic_probe_used=False, future_fm_head_used=False)
        result["parameters"] = sum(p.numel() for p in model.parameters())
        result["status"] = "running"
        save()
        context, mask = model._append_proprio_to_context(
            context=batch["context"], context_mask=batch["context_mask"], proprio=batch["proprio"][:, 0])
        pre, tokens = model._anticip_video_pre(
            first_frame_latents=model.vae.first_frame, context=context, context_mask=mask)
        seq = int(tokens.shape[1])
        attention_mask = model._build_anticip_attention_mask(
            video_seq_len=seq, action_seq_len=32,
            video_tokens_per_frame=int(pre["meta"]["tokens_per_frame"]), device=device)
        def prefill():
            return model.mot.prefill_video_cache(
                video_tokens=tokens, video_freqs=pre["freqs"], video_t_mod=pre["t_mod"],
                video_context_payload={"context": pre["context"], "mask": pre["context_mask"]},
                video_attention_mask=attention_mask[:seq, :seq])
        print(json.dumps(dict(event="measure", scope="video_prefill")), flush=True)
        result["cases"].append(measure(model, prefill, args, device, "video_prefill"))
        save()
        image = torch.zeros(1, 3, 224, 448, device=device, dtype=torch.bfloat16)
        keys = tuple(model.state_dict())
        for steps in args.steps:
            def infer():
                return model.infer_action(
                    prompt=None, input_image=image, action_horizon=32, num_video_frames=9,
                    proprio=batch["proprio"][:, 0], context=batch["context"], context_mask=batch["context_mask"],
                    num_inference_steps=steps, seed=args.seed)["action"]
            print(json.dumps(dict(event="measure", scope="infer_action", steps=steps)), flush=True)
            case = measure(model, infer, args, device, f"infer_action_{steps}_steps")
            result["cases"].append(case)
            save()
            print(json.dumps({key: case[key] for key in ("scope", "median_wall_ms", "paired_median_time_change_percent", "paired_median_time_change_95ci")}), flush=True)
        assert tuple(model.state_dict()) == keys
        result["status"] = "complete"
        save()
    except Exception as error:
        result.update(status="failed", reason=f"{type(error).__name__}: {error}")
        save()
        raise


if __name__ == "__main__":
    main()
