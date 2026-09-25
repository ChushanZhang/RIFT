"""Precompute exact, window-wise RoboCOIN latents with the training VAE cache.

Run smoke first, then launch one worker per GPU with the same run directory.
The companion input module checks pixels against the original training path.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rift.models.wan22.helpers.loader import _load_registered_model
from rift.utils.vae_latent_cache import VaeCacheCollator, VaeLatentCache, tensor_digest
from robocoin_vae_inputs import (
    discover_episodes,
    make_window,
    prepare_episode,
    validate_episode_inputs,
)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def same_bytes(left, right):
    return torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


def source_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), Path(__file__).with_name("robocoin_vae_inputs.py")]}


def input_fingerprints(episodes):
    paths = set()
    for episode in episodes:
        paths.add(str(episode["parquet_path"]))
        paths.update(str(path) for path in episode["video_paths"].values())
    return {path: {"size": Path(path).stat().st_size, "mtime_ns": Path(path).stat().st_mtime_ns}
            for path in sorted(paths)}


def load_cache(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this environment")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    vae = _load_registered_model(
        str(args.vae_weights), "wan_video_vae", torch.bfloat16, str(device)
    )
    vae.eval().requires_grad_(False)
    # Production DeepSpeed uses bf16 parameters/inputs without native autocast.
    if torch.is_autocast_enabled("cuda"):
        raise RuntimeError("Expected the production non-autocast VAE context")
    cache = VaeLatentCache(args.cache_dir, vae, torch.bfloat16, device)
    return vae, cache, device


def collate_windows(windows, device):
    batch = VaeCacheCollator()([{"video": window} for window in windows])
    return batch["video"].to(device=device, dtype=torch.bfloat16), batch["vae_cache_keys"]


@torch.no_grad()
def smoke(args, episodes, vae, cache, device):
    selected = {}
    for episode in episodes:
        selected.setdefault(episode["dataset"], episode)
    reports, examples = [], []
    total_encode_seconds, total_encoded = 0.0, 0
    for episode in selected.values():
        print("SMOKE_PREPROCESS", episode["dataset"], flush=True)
        before_preprocessing = time.monotonic()
        canvas = prepare_episode(episode)
        preprocessing_seconds = time.monotonic() - before_preprocessing
        print("SMOKE_PIXELS_READY", episode["dataset"], preprocessing_seconds, flush=True)
        report = validate_episode_inputs(episode, canvas)
        starts = sorted({0, len(canvas) // 2, len(canvas) - 1})
        windows = [make_window(canvas, start) for start in starts]
        videos, keys = collate_windows(windows, device)
        torch.cuda.synchronize()
        before = time.monotonic()
        original = vae.encode(videos, device=device, tiled=False)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - before
        total_encode_seconds += elapsed
        total_encoded += len(windows)
        before_stats = dict(cache.stats)
        cold = cache.encode(videos, keys)
        warm = cache.encode(videos, keys)
        if not same_bytes(original, cold) or not same_bytes(original, warm):
            raise RuntimeError("Fresh VAE, cold cache and warm cache differ")
        if not bool(torch.isfinite(original).all()):
            raise RuntimeError("Non-finite latent in smoke check")
        report.update({"latent_bitwise_equal": True, "latent_shape": list(original.shape[1:]),
                       "preprocessing_seconds": preprocessing_seconds,
                       "initial_cache_misses": cache.stats["misses"] - before_stats["misses"],
                       "cache_hits": cache.stats["hits"] - before_stats["hits"],
                       "latent_dtype": str(original.dtype), "encode_seconds": elapsed})
        reports.append(report)
        # Two examples per dataset exercise cross-GPU cache reads and fresh encodes.
        for start in (0, len(canvas) - 1):
            output = args.run_dir / "examples" / f"{episode['dataset']}_{start}.pt"
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"video": make_window(canvas, start)}, output)
            examples.append(str(output))
        print("SMOKE_PASS", json.dumps(report), flush=True)
        del canvas, windows, videos, original, cold, warm
    report = {
        "status": "passed", "namespace": cache.namespace, "identity": cache.identity,
        "datasets": reports, "examples": examples,
        "mean_encode_seconds_per_window": total_encode_seconds / total_encoded,
        "episode_count": len(episodes), "window_count": sum(e["length"] for e in episodes),
    }
    save_json(args.run_dir / "smoke.json", report)
    print("SMOKE_COMPLETE", json.dumps(report), flush=True)


def assigned_episodes(episodes, world_size):
    assignments = [[] for _ in range(world_size)]
    loads = [0] * world_size
    for episode in sorted(episodes, key=lambda item: -item["length"]):
        rank = min(range(world_size), key=loads.__getitem__)
        assignments[rank].append(episode)
        loads[rank] += episode["length"]
    return assignments


def prefetched_episodes(assigned, run_dir):
    """Prepare the next episode on CPU while the current one is GPU encoded."""
    def done_path(episode):
        return (run_dir / "indices" / episode["dataset"] /
                f"episode_{episode['episode_index']:06d}.done.json")

    def prepare(episode):
        started = time.monotonic()
        canvas = prepare_episode(episode)
        return canvas, time.monotonic() - started

    pending = iter(episode for episode in assigned if not done_path(episode).exists())
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = next(pending, None)
        future = pool.submit(prepare, first) if first else None
        for episode in assigned:
            if done_path(episode).exists():
                yield episode, None, 0.0
                continue
            canvas, seconds = future.result()
            following = next(pending, None)
            future = pool.submit(prepare, following) if following else None
            yield episode, canvas, seconds


@torch.no_grad()
def worker(args, episodes, vae, cache, device):
    smoke_report = json.loads((args.run_dir / "smoke.json").read_text())
    if smoke_report["status"] != "passed" or smoke_report["namespace"] != cache.namespace:
        raise RuntimeError("Worker VAE identity does not match the validated smoke configuration")
    for path in smoke_report["examples"]:
        sample = torch.load(path, map_location="cpu", weights_only=True)
        videos, keys = collate_windows([sample["video"]], device)
        before_stats = dict(cache.stats)
        cached = cache.encode(videos, keys)
        if cache.stats["misses"] != before_stats["misses"] or cache.stats["hits"] != before_stats["hits"] + 1:
            raise RuntimeError(f"Expected a cross-GPU cache hit: {path}")
        fresh = vae.encode(videos, device=device, tiled=False)
        if not same_bytes(cached, fresh):
            raise RuntimeError(f"Cross-GPU VAE parity failed: {path}")
    save_json(args.run_dir / f"parity_rank{args.rank}.json", {
        "status": "passed", "namespace": cache.namespace, "examples": len(smoke_report["examples"]),
        "gpu": torch.cuda.get_device_name(device),
        "gpu_uuid": str(torch.cuda.get_device_properties(device).uuid),
    })
    assigned = assigned_episodes(episodes, args.world_size)[args.rank]
    status_path = args.run_dir / f"rank{args.rank}.json"
    state = {"state": "running", "rank": args.rank, "namespace": cache.namespace,
             "total_windows": sum(e["length"] for e in assigned), "completed_windows": 0,
             "total_episodes": len(assigned), "completed_episodes": 0}
    started = time.monotonic()
    save_json(status_path, state)
    try:
        for episode, canvas, preprocessing_seconds in prefetched_episodes(assigned, args.run_dir):
            dataset, episode_index = episode["dataset"], episode["episode_index"]
            index_path = args.run_dir / "indices" / dataset / f"episode_{episode_index:06d}.jsonl"
            done_path = index_path.with_suffix(".done.json")
            if done_path.exists():
                done = json.loads(done_path.read_text())
                if done["namespace"] != cache.namespace or done["windows"] != episode["length"]:
                    raise RuntimeError(f"Invalid completion manifest: {done_path}")
                if hashlib.sha256(index_path.read_bytes()).hexdigest() != done["index_sha256"]:
                    raise RuntimeError(f"Invalid index checksum: {index_path}")
                rows = [json.loads(line) for line in index_path.read_text().splitlines()]
                if [row["start_frame"] for row in rows] != list(range(episode["length"])):
                    raise RuntimeError(f"Incomplete window index: {index_path}")
                for row in rows:
                    key = row["key"]
                    cache._load(cache.directory / key[:2] / f"{key}.pt", key, (48, 3, 24, 20))
                state["completed_windows"] += episode["length"]
                state["completed_episodes"] += 1
                save_json(status_path, state)
                continue
            before = time.monotonic()
            state.update({"current_dataset": dataset, "current_episode": episode_index,
                          "preprocessing_seconds": preprocessing_seconds})
            index_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = index_path.with_suffix(f".rank{args.rank}.tmp")
            with temporary.open("w") as index_file:
                for start in range(0, len(canvas), args.batch_size):
                    starts = range(start, min(start + args.batch_size, len(canvas)))
                    videos, keys = collate_windows([make_window(canvas, s) for s in starts], device)
                    latents = cache.encode(videos, keys)
                    if not bool(torch.isfinite(latents).all()):
                        raise RuntimeError(f"Non-finite latent: {dataset}/{episode_index}/{start}")
                    for frame, key in zip(starts, keys):
                        index_file.write(json.dumps({"start_frame": frame, "key": key}) + "\n")
                    state["completed_windows"] += len(keys)
                    if start % (args.batch_size * 16) == 0:
                        elapsed = time.monotonic() - started
                        state.update({"elapsed_seconds": elapsed, "cache_stats": dict(cache.stats),
                                      "windows_per_second": state["completed_windows"] / elapsed})
                        save_json(status_path, state)
                    del videos, latents
                index_file.flush()
                os.fsync(index_file.fileno())
            os.replace(temporary, index_path)
            save_json(done_path, {"namespace": cache.namespace, "windows": len(canvas),
                                  "dataset": dataset, "episode_index": episode_index,
                                  "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest()})
            state["completed_episodes"] += 1
            save_json(status_path, state)
            print("EPISODE_COMPLETE", args.rank, dataset, episode_index, len(canvas),
                  f"seconds={time.monotonic() - before:.2f}", flush=True)
            del canvas
        state.update({"state": "complete", "elapsed_seconds": time.monotonic() - started,
                      "cache_stats": dict(cache.stats)})
        save_json(status_path, state)
        print("WORKER_COMPLETE", json.dumps(state), flush=True)
    except BaseException as exc:
        state.update({"state": "failed", "error": repr(exc)})
        save_json(status_path, state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--vae-weights", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "worker"), required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.world_size <= 0 or not 0 <= args.rank < args.world_size:
        parser.error("world-size must be positive and rank must be in [0, world-size)")
    if args.batch_size <= 0 or args.cpu_threads <= 0:
        parser.error("batch-size and cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    episodes = discover_episodes(args.root)
    if args.mode == "smoke":
        save_json(args.run_dir / "smoke.json", {"status": "running"})
        save_json(args.run_dir / "episodes.json", episodes)
        save_json(args.run_dir / "settings.json", {
            "root": str(args.root), "cache_dir": str(args.cache_dir),
            "vae_weights": str(args.vae_weights), "num_frames": 33,
            "video_offsets": list(range(0, 33, 4)), "per_camera_resize": [224, 224],
            "canvas": [384, 320], "normalization": {"mean": 0.5, "std": 0.5},
            "window_policy": "each frame, repeat final frame at episode boundary",
            "source_hashes": source_hashes(),
        })
        save_json(args.run_dir / "input_fingerprints.json", input_fingerprints(episodes))
    else:
        settings = json.loads((args.run_dir / "settings.json").read_text())
        if (settings["root"] != str(args.root) or settings["cache_dir"] != str(args.cache_dir)
                or settings["source_hashes"] != source_hashes()
                or json.loads((args.run_dir / "episodes.json").read_text()) != episodes
                or json.loads((args.run_dir / "input_fingerprints.json").read_text()) != input_fingerprints(episodes)):
            raise RuntimeError("Inputs, cache root or preprocessing source changed after smoke validation")
    vae, cache, device = load_cache(args)
    if args.mode == "smoke":
        smoke(args, episodes, vae, cache, device)
    else:
        worker(args, episodes, vae, cache, device)


if __name__ == "__main__":
    main()
