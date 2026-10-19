"""Encode frozen LIBERO VAE outputs directly into episode safetensors.

Finalize validates every saved window; publish-ready also requires the real
training-loader parity audit. No intermediate window cache is written.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.libero_vae_inputs import discover_episodes, make_window, prepare_episode, validate_episode_pixels
from rift.utils.libero_episode_cache import FORMAT, LATENT_SHAPE, pack_episode, _file_sha256 as file_sha256
from rift.utils.libero_indexed_vae_cache import CONTRACT, PREPROCESSING
from rift.utils.vae_latent_cache import identity_hash, tensor_digest, frozen_vae_identity


@dataclass(frozen=True)
class Episode:
    dataset: str
    index: int
    keys: tuple[str, ...]
    index_sha256: str

    @property
    def windows(self):
        return len(self.keys)

    @property
    def relative_path(self):
        return f"episodes/{self.dataset}/episode_{self.index:06d}.safetensors"


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".pending-",
                                     delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def save_once(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Existing run metadata differs: {path}")
    else:
        atomic_json(path, value)


def inventory(cache, run, namespace, datasets):
    identity = json.loads((cache / namespace / "identity.json").read_text())
    if identity_hash(identity) != namespace:
        raise ValueError("Cache namespace identity mismatch")
    episodes = discover_episodes(datasets)
    fingerprints = {}
    for episode in episodes:
        for path in [episode["parquet_path"], *episode["video_paths"].values()]:
            stat = Path(path).stat()
            fingerprints[path] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    save_once(run / "settings.json", PREPROCESSING)
    save_once(run / "episodes.json", episodes)
    save_once(run / "input_fingerprints.json", fingerprints)
    return identity, episodes


@torch.no_grad()
def encode_episode(cache, run, namespace, source, vae, device):
    """Write one episode directly; no window .pt file is ever created."""
    if vae.training or any(parameter.requires_grad for parameter in vae.parameters()):
        raise ValueError("Precompute requires a frozen VAE in eval mode")
    canvas = prepare_episode(source)
    pixel_check = validate_episode_pixels(source, canvas)
    keys = tuple(tensor_digest(make_window(canvas, start)) for start in range(len(canvas)))
    raw = b"".join((json.dumps({"start_frame": start, "key": key}, separators=(",", ":"))
                    + "\n").encode() for start, key in enumerate(keys))
    episode = Episode(source["dataset"], source["episode_index"], keys, hashlib.sha256(raw).hexdigest())
    path = cache / namespace / episode.relative_path
    if not path.exists():
        latents = []
        for start in range(len(canvas)):
            video = make_window(canvas, start).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
            latent = vae.encode(video, device=device, tiled=False)[0].detach().cpu().contiguous()
            if latent.dtype != torch.bfloat16 or tuple(latent.shape) != LATENT_SHAPE:
                raise ValueError("Frozen VAE output differs from the LIBERO BF16 contract")
            latents.append(latent)
        pack_episode(path, torch.stack(latents), namespace=namespace, dataset=episode.dataset,
                     episode_index=episode.index, index_sha256=episode.index_sha256,
                     window_sha256=[tensor_digest(latent) for latent in latents], window_keys=keys)
    # Verify resumed files too; never overwrite a partial/mismatched episode.
    from rift.utils.libero_episode_cache import load_window, EpisodeHeaderCache
    header_cache = EpisodeHeaderCache()
    for start, key in enumerate(keys):
        load_window(path, start, key=key, namespace=namespace, dataset=episode.dataset,
                    episode_index=episode.index, index_sha256=episode.index_sha256,
                    window_count=episode.windows, header_cache=header_cache)
    index = run / "indices" / episode.dataset / f"episode_{episode.index:06d}.jsonl"
    index.parent.mkdir(parents=True, exist_ok=True)
    if index.exists() and index.read_bytes() != raw:
        raise ValueError(f"Existing index differs: {index}")
    index.write_bytes(raw)
    row = {"namespace": namespace, "dataset": episode.dataset, "episode_index": episode.index,
           "windows": episode.windows, "index_sha256": episode.index_sha256,
           "path": episode.relative_path, "file_sha256": file_sha256(path),
           "mode": "direct-encode", "pixel_check": pixel_check}
    atomic_json(index.with_suffix(".done.json"), row)
    return row


def finalize(cache, run, namespace, identity, episodes):
    """Audit every saved window before publishing the validated inventory."""
    from safetensors import safe_open
    from rift.utils.libero_episode_cache import validate_header, TENSOR_NAME

    rows, unique_keys = [], set()
    for source in episodes:
        index = run / "indices" / source["dataset"] / f"episode_{source['episode_index']:06d}.jsonl"
        done = json.loads(index.with_suffix(".done.json").read_text())
        raw = index.read_bytes()
        indices = [json.loads(line) for line in raw.splitlines()]
        expected_path = f"episodes/{source['dataset']}/episode_{source['episode_index']:06d}.safetensors"
        if (len(indices) != source["length"] or done["namespace"] != namespace
                or done["dataset"] != source["dataset"] or done["episode_index"] != source["episode_index"]
                or done["windows"] != source["length"] or done["path"] != expected_path
                or done["index_sha256"] != hashlib.sha256(raw).hexdigest()
                or any(row["start_frame"] != start for start, row in enumerate(indices))):
            raise ValueError(f"Incomplete episode index: {index}")
        path = cache / namespace / expected_path
        if file_sha256(path) != done["file_sha256"]:
            raise ValueError(f"Episode file checksum changed: {path}")
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys, hashes = validate_header(handle, namespace=namespace, dataset=source["dataset"],
                                           episode_index=source["episode_index"],
                                           index_sha256=done["index_sha256"], window_count=source["length"])
            slab = handle.get_slice(TENSOR_NAME)
            for start, row in enumerate(indices):
                latent = slab[start]
                if (keys[start] != row["key"] or tensor_digest(latent) != hashes[start]
                        or not bool(torch.isfinite(latent).all())):
                    raise ValueError(f"Packed latent differs: {path}:{start}")
                unique_keys.add(row["key"])
        rows.append({name: done[name] for name in
                     ("dataset", "episode_index", "windows", "index_sha256", "path", "file_sha256")})
    atomic_json(cache / namespace / "episodes" / "manifest.json",
                {"format": FORMAT, "namespace": namespace, "episodes": rows})
    report = {"status": "passed", "namespace": namespace, "identity": identity,
              "preprocessing": PREPROCESSING, "episode_count": len(episodes),
              "window_count": sum(row["length"] for row in episodes), "unique_latents": len(unique_keys),
              "latent_shape": list(LATENT_SHAPE), "latent_dtype": "torch.bfloat16",
              "all_starts_covered_once": True, "all_indices_sha256_verified": True,
              "all_packed_windows_checksum_verified": True}
    atomic_json(run / "validation.json", report)
    return report


def publish_ready(cache, run, namespace):
    report = json.loads((run / "validation.json").read_text())
    training = json.loads((run / "training_validation.json").read_text())
    if (report["status"] != "passed" or training["status"] != "passed"
            or report["namespace"] != namespace or training["namespace"] != namespace
            or Path(training["cache_dir"]).resolve() != cache
            or Path(training["index_dir"]).resolve() != run
            or training["window_count"] != report["window_count"]
            or not training["nonvisual_bitwise_parity_passed"]
            or not training["original_pixels_match_index_keys"]):
        raise ValueError("The real training-loader parity audit is missing or differs")
    ready = {"status": "passed", "contract": CONTRACT, "format": FORMAT,
             "cache_dir": str(cache), "index_dir": str(run), "namespace": namespace,
             "episode_count": report["episode_count"], "window_count": report["window_count"],
             "training_validation": str(run / "training_validation.json")}
    atomic_json(cache / "ready.json", ready)
    return ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["encode", "finalize", "publish-ready"])
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--namespace")
    parser.add_argument("--dataset-dirs", nargs="+")
    parser.add_argument("--vae-weights", type=Path)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--maximum-episodes", type=int, help="Limit each encoding worker for a smoke run; does not publish readiness")
    args = parser.parse_args()
    cache, run = args.cache_dir.resolve(), args.index_dir.resolve()
    if args.command == "encode":
        if (not args.vae_weights or not args.dataset_dirs or not torch.cuda.is_available()
                or not 0 <= args.worker_index < args.worker_count
                or (args.maximum_episodes is not None and args.maximum_episodes < 1)):
            parser.error("encode needs CUDA, --vae-weights, --dataset-dirs and a valid worker shard")
        from rift.models.wan22.helpers.loader import _load_registered_model
        device = torch.device("cuda:0")
        torch.set_num_threads(1)
        vae = _load_registered_model(str(args.vae_weights), "wan_video_vae", torch.bfloat16, str(device))
        vae.eval().requires_grad_(False)
        if torch.is_autocast_enabled("cuda"):
            raise RuntimeError("Expected BF16 parameters/input without autocast")
        identity = frozen_vae_identity(vae, torch.bfloat16)
        namespace = identity_hash(identity)
        if args.namespace is not None and args.namespace != namespace:
            raise ValueError("Requested namespace differs from this frozen VAE")
        save_once(cache / namespace / "identity.json", identity)
        _, episodes = inventory(cache, run, namespace, args.dataset_dirs)
        assigned = episodes[args.worker_index::args.worker_count]
        if args.maximum_episodes is not None:
            assigned = assigned[:args.maximum_episodes]
        for source in assigned:
            print(json.dumps(encode_episode(cache, run, namespace, source, vae, device)), flush=True)
        print(json.dumps({"status": "worker_complete", "namespace": namespace,
                          "episodes": len(assigned), "next": "finalize, training-loader audit, publish-ready"}))
        return
    if not args.namespace:
        parser.error("--namespace is required for finalize/publish-ready")
    if args.command == "publish-ready":
        print(json.dumps(publish_ready(cache, run, args.namespace)))
        return
    if not args.dataset_dirs:
        parser.error("--dataset-dirs is required for finalize")
    identity, episodes = inventory(cache, run, args.namespace, args.dataset_dirs)
    torch.set_num_threads(1)
    print(json.dumps(finalize(cache, run, args.namespace, identity, episodes)), flush=True)


if __name__ == "__main__":
    main()
