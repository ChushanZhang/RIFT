"""Check packed clips against the original pixels and real RIFT loader.

This CPU audit loads no policy or VAE. It verifies first/middle/final windows of
the first episode of every configured suite, including repeated final frames.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rift.utils import misc
from rift.utils.vae_latent_cache import VaeCacheCollator, tensor_digest


VISUAL_KEYS = {"video", "video_latents", "video_shape", "vae_cache_keys",
               "vae_cache_namespace"}


def verify(args):
    started = time.monotonic()
    torch.set_num_threads(4)
    config = OmegaConf.load(args.config)
    data_config = OmegaConf.create(OmegaConf.to_container(config.data.train, resolve=True))
    data_config.vae_cache_index_dir = None
    data_config.vae_cache_dir = None
    if data_config.get("skip_padding_as_possible", False):
        raise ValueError("Cache parity audit requires the original padded windows without random retries.")
    reports = []
    with tempfile.TemporaryDirectory(prefix="rift-cache-audit-") as temporary:
        temporary = Path(temporary)
        original_work = temporary / "original"
        original_work.mkdir()
        with patch.object(misc, "get_work_dir", return_value=str(original_work)):
            original = instantiate(data_config)
        original.lerobot_dataset.strict_loading = True
        cached_config = OmegaConf.create(OmegaConf.to_container(data_config, resolve=True))
        cached_config.vae_cache_index_dir = str(args.index_dir)
        cached_config.vae_cache_dir = str(args.cache_dir)
        # Reuse the original loader's own summary, so normalized targets remain
        # exactly the same and the audit does not calculate it a second time.
        cached_config.pretrained_norm_stats = str(original_work / "dataset_stats.json")
        cached_work = temporary / "cached"
        cached_work.mkdir()
        with patch.object(misc, "get_work_dir", return_value=str(cached_work)):
            cached = instantiate(cached_config)
        assert len(original) == len(cached)
        children = original.lerobot_dataset.multi_dataset._datasets
        offset = 0
        for dataset_index, child in enumerate(children):
            episode_indices = list(child.episodes) if child.episodes is not None else list(child.meta.episodes)
            episode = int(episode_indices[0])
            first = int(child.episode_data_index["from"][0])
            end = int(child.episode_data_index["to"][0])
            for start in sorted({0, (end - first) // 2, end - first - 1}):
                index = offset + first + start
                expected = original._get(index)
                actual = cached._get(index)
                assert "video" not in actual
                assert expected["video"].dtype == torch.float32
                key = tensor_digest(expected["video"])
                assert key == actual["vae_cache_keys"] == cached._latent_reader.key(dataset_index, episode, start)
                assert set(expected) - VISUAL_KEYS == set(actual) - VISUAL_KEYS
                for name in set(expected) - VISUAL_KEYS:
                    if isinstance(expected[name], torch.Tensor):
                        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
                    else:
                        assert actual[name] == expected[name], name
                # Indexed metadata/keys survive collation without pixel hashing.
                with patch("rift.utils.vae_latent_cache.tensor_digest", side_effect=AssertionError("pixel hashing")):
                    collated = VaeCacheCollator()([actual])
                assert collated["vae_cache_keys"] == [key]
                assert tuple(actual["video_latents"].shape) == (48, 3, 14, 28)
                assert actual["video_latents"].dtype == torch.bfloat16
                reports.append({"dataset_index": dataset_index, "dataset": Path(data_config.dataset_dirs[dataset_index]).name,
                                "cached_source_dataset": cached._latent_reader.dataset_names[dataset_index],
                                "episode_index": episode, "start_frame": start, "training_index": index,
                                "pixel_key": key, "latent_sha256": tensor_digest(actual["video_latents"]),
                                "nonvisual_bitwise_equal": True,
                                "original_pixels_match_index_key": True})
            offset += len(child)
        report = {"status": "passed", "validated_at": datetime.now(timezone.utc).isoformat(),
                  "config": str(args.config.resolve()), "cache_dir": str(args.cache_dir.resolve()),
                  "index_dir": str(args.index_dir.resolve()), "namespace": cached.vae_cache_namespace,
                  "dataset_count": len(children), "window_count": len(cached), "sample_count": len(reports),
                  "sample_policy": "first episode per suite, first/middle/final padded window",
                  "samples": reports, "pixels_retained": False, "cache_payload_checksum_verified": False,
                  "original_pixels_match_index_keys": True, "nonvisual_bitwise_parity_passed": True,
                  "original_vae_encoding_performed": False, "elapsed_seconds": time.monotonic() - started}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    verify(parser.parse_args())
