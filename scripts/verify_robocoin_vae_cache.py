"""Independently verify the completed three-task RoboCOIN VAE cache.

Run with the training checkout on PYTHONPATH. This script needs CPU PyTorch,
but does not load the VAE, initialize CUDA, or modify any latent or index.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from stat import S_ISREG
import sys
import time

import torch

from rift.utils.vae_latent_cache import identity_hash, tensor_digest


EXPECTED_EPISODES = 342
EXPECTED_WINDOWS = 238_864
EXPECTED_DATASETS = 3
LATENT_SHAPE = (48, 3, 24, 20)
LATENT_BYTES = math.prod(LATENT_SHAPE) * 2
HEX_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, ensure_ascii=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def edge_middle(values):
    return [values[index] for index in sorted({0, len(values) // 2, len(values) - 1})]


def rank_reports(run_dir, prefix):
    result = {}
    pattern = re.compile(re.escape(prefix) + r"(\d+)\.json\Z")
    for path in sorted(run_dir.glob(f"{prefix}*.json")):
        match = pattern.fullmatch(path.name)
        require(match is not None, f"Unexpected rank report filename: {path}")
        rank = int(match.group(1))
        require(rank not in result, f"Duplicate rank report: {path}")
        result[rank] = read_json(path)
    return result


def validate_workers(run_dir, episodes, smoke, namespace):
    workers = rank_reports(run_dir, "rank")
    parity = rank_reports(run_dir, "parity_rank")
    require(workers and set(workers) == set(range(len(workers))),
            "Worker reports must contain contiguous ranks starting at 0")
    require(set(parity) == set(workers), "Worker and cross-GPU parity ranks differ")
    assignments = [[] for _ in workers]
    loads = [0] * len(workers)
    for episode in sorted(episodes, key=lambda item: -item["length"]):
        rank = min(range(len(workers)), key=loads.__getitem__)
        assignments[rank].append(episode)
        loads[rank] += episode["length"]
    for rank, worker in sorted(workers.items()):
        require(worker.get("state") == "complete" and worker.get("rank") == rank,
                f"Worker {rank} is not complete or has the wrong rank")
        require(worker.get("namespace") == namespace, f"Worker {rank} namespace mismatch")
        for field in ("total_windows", "completed_windows"):
            require(worker.get(field) == loads[rank], f"Worker {rank}: incorrect {field}")
        for field in ("total_episodes", "completed_episodes"):
            require(worker.get(field) == len(assignments[rank]),
                    f"Worker {rank}: incorrect {field}")
        report = parity[rank]
        require(report.get("status") == "passed" and report.get("namespace") == namespace,
                f"Worker {rank} cross-GPU parity is not passed for this namespace")
        require(report.get("examples") == len(smoke["examples"]) and report.get("gpu_uuid"),
                f"Worker {rank} cross-GPU parity examples or GPU UUID are missing")
    return workers, parity


def verify(run_dir, cache_dir):
    started = time.monotonic()
    episodes = read_json(run_dir / "episodes.json")
    smoke = read_json(run_dir / "smoke.json")
    settings = read_json(run_dir / "settings.json")
    require(isinstance(episodes, list) and episodes, "Episode inventory must be a nonempty list")
    datasets = defaultdict(list)
    seen_episodes = set()
    for episode in episodes:
        dataset, index, length = (episode["dataset"], episode["episode_index"], episode["length"])
        require(isinstance(dataset, str) and dataset not in ("", ".", "..")
                and Path(dataset).name == dataset, f"Invalid dataset name: {dataset!r}")
        require(type(index) is int and index >= 0 and type(length) is int and length > 0,
                f"Invalid episode index or length: {dataset}/{index}")
        require((dataset, index) not in seen_episodes, f"Duplicate episode: {dataset}/{index}")
        seen_episodes.add((dataset, index))
        datasets[dataset].append(episode)
    window_count = sum(episode["length"] for episode in episodes)
    require((len(episodes), window_count, len(datasets))
            == (EXPECTED_EPISODES, EXPECTED_WINDOWS, EXPECTED_DATASETS),
            f"Unexpected inventory: {len(episodes)} episodes, {window_count} windows, "
            f"{len(datasets)} datasets; expected {EXPECTED_EPISODES}, "
            f"{EXPECTED_WINDOWS}, {EXPECTED_DATASETS}")
    namespace = smoke.get("namespace", "")
    require(isinstance(namespace, str) and HEX_DIGEST.fullmatch(namespace), "Invalid namespace")
    require(smoke.get("status") == "passed", "Smoke validation has not passed")
    require(smoke.get("episode_count") == len(episodes)
            and smoke.get("window_count") == window_count, "Smoke inventory counts differ")
    require(isinstance(smoke.get("examples"), list) and len(smoke["examples"]) == 2 * len(datasets),
            "Smoke must provide two cross-GPU examples per dataset")
    identity = read_json(cache_dir / namespace / "identity.json")
    require(identity == smoke.get("identity") and identity_hash(identity) == namespace,
            "Cache identity differs from smoke identity or namespace hash")
    expected_settings = {
        "num_frames": 33, "video_offsets": list(range(0, 33, 4)),
        "per_camera_resize": [224, 224], "canvas": [384, 320],
        "normalization": {"mean": 0.5, "std": 0.5},
        "window_policy": "each frame, repeat final frame at episode boundary",
    }
    for name, value in expected_settings.items():
        require(settings.get(name) == value, f"Unexpected preprocessing setting: {name}")
    require(Path(settings["cache_dir"]).resolve() == cache_dir.resolve(),
            "Settings refer to a different cache directory")
    smoke_datasets = smoke.get("datasets", [])
    require(len(smoke_datasets) == len(datasets)
            and {report["dataset"] for report in smoke_datasets} == set(datasets),
            "Smoke does not cover every dataset exactly once")
    for report in smoke_datasets:
        require(report.get("bitwise_equal") is True and report.get("latent_bitwise_equal") is True,
                f"Smoke pixel or latent parity failed: {report['dataset']}")
        require(report.get("latent_shape") == list(LATENT_SHAPE)
                and report.get("latent_dtype") == "torch.bfloat16",
                f"Smoke latent contract differs: {report['dataset']}")
        episode = next((item for item in datasets[report["dataset"]]
                        if item["episode_index"] == report.get("episode_index")), None)
        require(episode is not None and report.get("length") == episode["length"],
                f"Smoke episode is absent or has a different length: {report['dataset']}")
        windows = report.get("windows", [])
        require([window.get("start") for window in windows]
                == edge_middle(list(range(episode["length"]))),
                f"Smoke start/middle/end windows are missing: {report['dataset']}")
        require(all(window.get("different_bytes") == 0
                    and window.get("shape") == [3, 9, 384, 320]
                    and window.get("dtype") == "torch.float32" for window in windows),
                f"Smoke input contract differs: {report['dataset']}")
    workers, parity = validate_workers(run_dir, episodes, smoke, namespace)

    all_keys, samples, coverage = set(), [], {}
    expected_indices, expected_done = set(), set()
    for dataset, items in sorted(datasets.items()):
        items.sort(key=lambda episode: episode["episode_index"])
        selected_episodes = {episode["episode_index"] for episode in edge_middle(items)}
        dataset_keys, indexed_windows = set(), 0
        for episode in items:
            index = episode["episode_index"]
            index_path = run_dir / "indices" / dataset / f"episode_{index:06d}.jsonl"
            done_path = index_path.with_suffix(".done.json")
            expected_indices.add(index_path)
            expected_done.add(done_path)
            raw = index_path.read_bytes()
            done = read_json(done_path)
            require(done.get("namespace") == namespace and done.get("dataset") == dataset
                    and done.get("episode_index") == index
                    and done.get("windows") == episode["length"],
                    f"Completion manifest differs: {done_path}")
            require(done.get("index_sha256") == hashlib.sha256(raw).hexdigest(),
                    f"Index checksum differs: {index_path}")
            rows = [json.loads(line) for line in raw.splitlines()]
            require(len(rows) == episode["length"]
                    and all(type(row.get("start_frame")) is int
                            and row["start_frame"] == start for start, row in enumerate(rows)),
                    f"Window starts must cover 0..length-1 once, in order: {index_path}")
            for row in rows:
                key = row.get("key")
                require(isinstance(key, str) and HEX_DIGEST.fullmatch(key),
                        f"Invalid content key: {index_path}:{row['start_frame']}")
                dataset_keys.add(key)
            indexed_windows += len(rows)
            if index in selected_episodes:
                samples.extend({"dataset": dataset, "episode_index": index, **row}
                               for row in edge_middle(rows))
        coverage[dataset] = {"episode_count": len(items), "window_count": indexed_windows,
                             "unique_latents": len(dataset_keys), "all_starts_covered_once": True}
        all_keys.update(dataset_keys)
        print("INDEX_VERIFIED", dataset, json.dumps(coverage[dataset]), flush=True)
    require(set((run_dir / "indices").glob("*/episode_*.jsonl")) == expected_indices,
            "Unexpected or missing episode index files")
    require(set((run_dir / "indices").glob("*/episode_*.done.json")) == expected_done,
            "Unexpected or missing episode completion files")

    disk_bytes = allocated_disk_bytes = 0
    min_file_bytes, max_file_bytes = None, 0
    for count, key in enumerate(sorted(all_keys), 1):
        path = cache_dir / namespace / key[:2] / f"{key}.pt"
        stat = path.stat()
        require(S_ISREG(stat.st_mode) and LATENT_BYTES <= stat.st_size <= LATENT_BYTES + 65_536,
                f"Missing, non-file or implausibly sized latent: {path}")
        disk_bytes += stat.st_size
        allocated_disk_bytes += stat.st_blocks * 512
        min_file_bytes = stat.st_size if min_file_bytes is None else min(min_file_bytes, stat.st_size)
        max_file_bytes = max(max_file_bytes, stat.st_size)
        if count % 25_000 == 0:
            print("FILES_VERIFIED", count, "of", len(all_keys), flush=True)

    for sample in samples:
        key = sample["key"]
        path = cache_dir / namespace / key[:2] / f"{key}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        require(isinstance(payload, dict), f"Sample payload is not a dictionary: {path}")
        latent = payload.get("latent")
        require(payload.get("namespace") == namespace and payload.get("key") == key,
                f"Sample namespace or key differs: {path}")
        require(isinstance(latent, torch.Tensor) and tuple(latent.shape) == LATENT_SHAPE
                and latent.dtype == torch.bfloat16 and bool(torch.isfinite(latent).all()),
                f"Sample shape, dtype or finite check failed: {path}")
        checksum = tensor_digest(latent)
        require(payload.get("sha256") == checksum, f"Sample payload checksum differs: {path}")
        sample["latent_sha256"] = checksum

    return {
        "status": "passed", "validated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir.resolve()), "cache_dir": str(cache_dir.resolve()),
        "namespace": namespace, "identity": identity,
        "episode_count": len(episodes), "window_count": window_count,
        "dataset_count": len(datasets), "datasets": coverage,
        "unique_latents": len(all_keys), "disk_bytes": disk_bytes,
        "allocated_disk_bytes": allocated_disk_bytes,
        "min_file_bytes": min_file_bytes, "max_file_bytes": max_file_bytes,
        "worker_count": len(workers), "all_workers_complete": True,
        "cross_gpu_parity_passed": True, "cross_gpu_parity": parity,
        "all_indices_sha256_verified": True, "all_starts_covered_once": True,
        "all_referenced_files_size_checked": True,
        "sample_checksum_count": len(samples),
        "sample_unique_latents": len({sample["key"] for sample in samples}),
        "sample_policy": "first/middle/last episode per dataset; first/middle/last window per episode",
        "samples": samples, "latent_shape": list(LATENT_SHAPE), "latent_dtype": "torch.bfloat16",
        "preprocessing": {**expected_settings, "canvas_axes": "height,width",
                          "concat_multi_camera": "robotwin",
                          "camera_order": "head,left_wrist,right_wrist",
                          "training_input_bitwise_parity_passed": True},
        "source_hashes": settings.get("source_hashes"),
        "validation_scope": "All indices and file sizes; sampled payload checksums, shape, dtype and finite values. "
                            "Generation workers validate every payload checksum, dtype, shape and finite values.",
        "training": {"verified_reader": "RIFT", "override": f"++vae_cache_dir={cache_dir.resolve()}"},
        "elapsed_seconds": time.monotonic() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        report = verify(args.run_dir, args.cache_dir)
    except Exception as error:
        report = {"status": "failed", "validated_at": datetime.now(timezone.utc).isoformat(),
                  "run_dir": str(args.run_dir.resolve()), "cache_dir": str(args.cache_dir.resolve()),
                  "error": f"{type(error).__name__}: {error}"}
        save_json(args.run_dir / "validation.json", report)
        save_json(args.cache_dir / "precompute_manifest.json", report)
        print("VALIDATION_FAILED", json.dumps(report), file=sys.stderr, flush=True)
        return 1
    save_json(args.run_dir / "validation.json", report)
    save_json(args.cache_dir / "precompute_manifest.json", report)
    summary = {key: report[key] for key in (
        "status", "namespace", "dataset_count", "episode_count", "window_count", "unique_latents",
        "disk_bytes", "worker_count", "sample_checksum_count", "elapsed_seconds", "training")}
    print("VALIDATION_PASSED", json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
