"""Read-only checksum and index audit of a published LIBERO safetensors cache."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rift.utils.libero_episode_cache import validate_header, _file_sha256
from rift.utils.libero_indexed_vae_cache import resolve_episode_index


def verify(cache, workers=4):
    started = time.monotonic()
    cache = Path(cache).resolve()
    index = Path(resolve_episode_index(cache))
    ready = json.loads((cache / "ready.json").read_text())
    namespace = ready["namespace"]
    root = cache / namespace
    manifest_bytes = (root / "episodes/manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    rows = manifest["episodes"]
    if (manifest.get("namespace") != namespace or manifest.get("format") != "libero_episode_bf16_v1"
            or len(rows) != ready["episode_count"]
            or sum(row["windows"] for row in rows) != ready["window_count"]
            or len({(row["dataset"], row["episode_index"]) for row in rows}) != len(rows)):
        raise ValueError("Episode inventory differs from published readiness")

    def verify_episode(row):
        relative = f"episodes/{row['dataset']}/episode_{row['episode_index']:06d}.safetensors"
        path = root / relative
        if row["path"] != relative or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Invalid episode path: {path}")
        if _file_sha256(path) != row["file_sha256"]:
            raise ValueError(f"Episode file checksum differs: {path}")
        index_path = index / "indices" / row["dataset"] / f"episode_{row['episode_index']:06d}.jsonl"
        raw = index_path.read_bytes()
        indices = [json.loads(line) for line in raw.splitlines()]
        if hashlib.sha256(raw).hexdigest() != row["index_sha256"] or len(indices) != row["windows"]:
            raise ValueError(f"Episode index differs: {index_path}")
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys, _ = validate_header(handle, namespace=namespace, dataset=row["dataset"],
                                      episode_index=row["episode_index"],
                                      index_sha256=row["index_sha256"], window_count=row["windows"])
            if not all(value["start_frame"] == start and value["key"] == keys[start]
                       for start, value in enumerate(indices)):
                raise ValueError(f"Window keys differ: {index_path}")
        return path.stat().st_size

    sizes = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for count, size in enumerate(pool.map(verify_episode, rows), 1):
            sizes.append(size)
            if count % 200 == 0:
                print(f"Verified {count}/{len(rows)} episode files", flush=True)
    return {"status": "passed", "episode_count": len(rows), "window_count": ready["window_count"],
            "all_episode_files_sha256_verified": True, "all_window_keys_and_indices_verified": True,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "cache_directory": str(cache), "namespace": namespace,
            "safetensors_bytes": sum(sizes), "elapsed_seconds": time.monotonic() - started}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    report = verify(args.cache_dir, args.workers)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
