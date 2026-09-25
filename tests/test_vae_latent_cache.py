from __future__ import annotations

import copy
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from rift.trainer import Wan22Trainer
from rift.utils.indexed_vae_cache import IndexedVaeLatentReader, LATENT_SHAPE
from rift.utils.vae_latent_cache import VaeLatentCache, encoding_identity, identity_hash, tensor_digest


class SmallVAE(nn.Module):
    z_dim = 48
    temporal_downsample_factor = 4
    upsampling_factor = 16

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.scale = [torch.zeros(48), torch.ones(48)]

    def single_encode(self, video, device):
        return video[:, :1, ::4, ::16, ::16].repeat(1, self.z_dim, 1, 1, 1) * self.weight


class VaeLatentCacheTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.vae = SmallVAE().to(torch.bfloat16).eval().requires_grad_(False)
        self.latent = torch.arange(LATENT_SHAPE[0], dtype=torch.bfloat16).view(-1, 1, 1, 1)
        self.latent = self.latent.expand(LATENT_SHAPE).contiguous()
        self.key = "a" * 64
        self.identity = encoding_identity(self.vae, torch.bfloat16, "cpu")
        self.identity.update(
            vae_class="rift.models.wan22.wan_video_vae.WanVideoVAE38",
            encode_source_sha256="b" * 64,
            autocast_enabled=True,
        )
        self.identity["environment"].update(
            device_type="cuda", gpu_model="NVIDIA A800", driver="different-driver",
            torch="different-torch-version",
        )
        self.namespace = self.write_cache(self.identity)

    def write_cache(self, identity):
        namespace = identity_hash(identity)
        directory = self.root / namespace
        directory.mkdir(exist_ok=True)
        (directory / "identity.json").write_text(json.dumps(identity))
        payload_path = directory / self.key[:2] / f"{self.key}.pt"
        payload_path.parent.mkdir(exist_ok=True)
        torch.save({"namespace": namespace, "key": self.key, "latent": self.latent,
                    "sha256": tensor_digest(self.latent)}, payload_path)
        return namespace

    def reader(self):
        # Exercise payload loading independently of the RoboCOIN dataset inventory.
        reader = IndexedVaeLatentReader.__new__(IndexedVaeLatentReader)
        reader.namespace = self.namespace
        reader.directory = self.root / self.namespace
        reader._keys = {(0, 0): bytes.fromhex(self.key)}
        return reader

    def open_cache(self, namespace=None):
        return VaeLatentCache(
            self.root, self.vae, torch.bfloat16, "cpu",
            precomputed_namespace=self.namespace if namespace is None else namespace,
        )

    def accept(self, cache, **overrides):
        args = dict(
            latents=self.reader().load(0, 0, 0).unsqueeze(0), keys=[self.key],
            namespaces=[self.namespace], video_shape=(1, 3, 9, 384, 320),
        )
        args.update(overrides)
        return cache.accept_precomputed(**args)

    def test_indexed_reads_reuse_other_package_and_runtime_without_writes(self):
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        with patch.object(self.vae, "single_encode", side_effect=AssertionError("online encoding")):
            cache = self.open_cache()
            actual = self.accept(cache)
        torch.testing.assert_close(actual[0], self.latent, rtol=0, atol=0)
        self.assertEqual(cache.namespace, self.namespace)
        self.assertEqual(cache.stats, {"hits": 1, "misses": 0})
        self.assertNotEqual(identity_hash(cache.identity), self.namespace)
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_precomputed_cache_rejects_different_weights_scale_or_dtype(self):
        for field, value in (("weights_sha256", "c" * 64), ("scale", [0, 2]),
                             ("input_dtype", "torch.float32"), ("format", 2)):
            with self.subTest(field=field):
                identity = copy.deepcopy(self.identity)
                identity[field] = value
                namespace = self.write_cache(identity)
                with self.assertRaisesRegex(RuntimeError, field):
                    self.open_cache(namespace)

    def test_missing_invalid_or_corrupt_namespace_fails_without_creating_cache(self):
        with self.assertRaisesRegex(ValueError, "Invalid precomputed"):
            self.open_cache("../outside-cache")
        with self.assertRaises(FileNotFoundError):
            self.open_cache("f" * 64)
        self.assertFalse((self.root / ("f" * 64)).exists())
        path = self.root / self.namespace / "identity.json"
        path.write_text(json.dumps({**self.identity, "weights_sha256": "c" * 64}))
        with self.assertRaisesRegex(RuntimeError, "namespace identity mismatch"):
            self.open_cache()

    def test_indexed_mode_cannot_encode_into_the_source_namespace(self):
        cache = self.open_cache()
        with patch.object(self.vae, "single_encode", side_effect=AssertionError("online encoding")):
            with self.assertRaisesRegex(RuntimeError, "precomputed latents only"):
                cache.encode(torch.zeros(1, 3, 9, 32, 32, dtype=torch.bfloat16), [self.key])

    def test_payload_integrity_and_batch_contract_are_still_checked(self):
        cache = self.open_cache()
        with self.assertRaisesRegex(RuntimeError, "namespace differs"):
            self.accept(cache, namespaces=["f" * 64])
        with self.assertRaisesRegex(ValueError, "shape"):
            self.accept(cache, latents=self.latent[None, :, :, :-1])
        path = self.root / self.namespace / self.key[:2] / f"{self.key}.pt"
        payload = torch.load(path, weights_only=True)
        payload["latent"][0, 0, 0, 0] += 1
        torch.save(payload, path)
        with self.assertRaisesRegex(RuntimeError, "corrupted indexed VAE latent"):
            self.accept(cache)
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.accept(cache)
        self.assertEqual(cache.stats, {"hits": 0, "misses": 0})

    def test_online_cache_keeps_its_own_namespace_and_cache_miss_encoding(self):
        cache = VaeLatentCache(self.root, self.vae, torch.bfloat16, "cpu")
        self.assertNotEqual(cache.namespace, self.namespace)
        video = torch.ones(1, 3, 9, 32, 32, dtype=torch.bfloat16)
        keys = [tensor_digest(video[0])]
        with patch.object(self.vae, "single_encode", wraps=self.vae.single_encode) as encode:
            first, second = cache.encode(video, keys), cache.encode(video, keys)
        encode.assert_called_once()
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertEqual(cache.stats, {"hits": 1, "misses": 1})

    def test_trainer_selects_the_dataset_namespace_and_rechecks_restored_weights(self):
        trainer = Wan22Trainer.__new__(Wan22Trainer)
        trainer.vae_cache_dir = self.root
        trainer.train_dataset = SimpleNamespace(vae_cache_namespace=self.namespace)
        trainer.accelerator = SimpleNamespace(autocast=nullcontext)
        model = SimpleNamespace(vae=self.vae, torch_dtype=torch.bfloat16, device="cpu")
        trainer._configure_vae_cache(model)
        torch.testing.assert_close(self.accept(model._vae_latent_cache)[0], self.latent, rtol=0, atol=0)
        self.vae.weight.add_(1)
        with self.assertRaisesRegex(RuntimeError, "changed after cache initialization"):
            self.accept(model._vae_latent_cache)
        with self.assertRaisesRegex(RuntimeError, "weights_sha256"):
            trainer._configure_vae_cache(model)


if __name__ == "__main__":
    unittest.main()
