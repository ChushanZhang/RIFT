from __future__ import annotations

from contextlib import nullcontext
import copy
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import test_fused_paths as fixtures
from rift.trainer import Wan22Trainer
from rift.utils.samplers import ResumableEpochSampler
from rift.utils.vae_latent_cache import VaeCacheCollator


class CompiledTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch._dynamo.reset()
        compile_fn = torch.compile
        # Exercise Dynamo and AOTAutograd on CPU without requiring a GPU compiler.
        self.compile_patch = patch(
            "torch.compile",
            side_effect=lambda fn, **kwargs: compile_fn(fn, backend="aot_eager", **kwargs),
        )
        self.compile_mock = self.compile_patch.start()
        self.addCleanup(self.compile_patch.stop)

    def _assert_run_equal(self, expected, actual):
        for key in ("loss", "metrics", "outputs", "grads", "keep_rows"):
            torch.testing.assert_close(actual[key], expected[key], rtol=3e-5, atol=3e-6)
        torch.testing.assert_close(actual["rng"], expected["rng"], rtol=0, atol=0)
        self.assertEqual(actual["step"], expected["step"])

    def _assert_parity(self, *, checkpoint, fused, latent_frames=3, noise=0.5, probe=True):
        fixture = fixtures.FusedTrainingPathsTest
        reference = fixture._model(
            checkpoint=checkpoint, noise_probability=noise, action_free=True,
            train_probe_head=probe,
        )
        reference.fuse_training_paths = fused
        compiled = fixture._model(
            checkpoint=checkpoint, noise_probability=noise, action_free=True,
            train_probe_head=probe, compile_training_denoise=True,
        )
        compiled.fuse_training_paths = fused
        sample = fixture._sample(padded=True, latent_frames=latent_frames)
        state = copy.deepcopy(reference.state_dict())
        for _ in range(2):
            reference.zero_grad(set_to_none=True)
            compiled.zero_grad(set_to_none=True)
            self._assert_run_equal(fixture._run(reference, sample), fixture._run(compiled, sample))
        self.compile_mock.assert_called_once()
        self.assertTrue(self.compile_mock.call_args.kwargs["fullgraph"])
        torch.testing.assert_close(compiled.state_dict(), state, rtol=0, atol=0)
        reference.load_state_dict(compiled.state_dict(), strict=True)

        # An already compiled training model must still use eager evaluation.
        reference.eval()
        compiled.eval()
        with patch.object(compiled.mot, "_compiled_joint_layer", side_effect=AssertionError("compiled eval")):
            self._assert_run_equal(fixture._run(reference, sample), fixture._run(compiled, sample))

    def test_training_gradients_with_fusion_and_checkpointing(self):
        for checkpoint in (False, True):
            for fused in (False, True):
                with self.subTest(checkpoint=checkpoint, fused=fused):
                    torch._dynamo.reset()
                    self.compile_mock.reset_mock()
                    self._assert_parity(checkpoint=checkpoint, fused=fused)

    def test_unequal_grids_and_disabled_probe(self):
        self._assert_parity(checkpoint=True, fused=True, latent_frames=4, probe=False)

    def test_all_action_rows_excluded(self):
        self._assert_parity(checkpoint=True, fused=True, noise=1.0)

    def test_default_training_does_not_compile(self):
        fixture = fixtures.FusedTrainingPathsTest
        model = fixture._model(checkpoint=True, noise_probability=0.5, action_free=True)
        fixture._run(model, fixture._sample(padded=True))
        self.compile_mock.assert_not_called()


class WarmupDataset(Dataset):
    vae_cache_namespace = "test-indexed-latents"

    def __init__(self, sample):
        self.sample = sample

    def __len__(self):
        return self.sample["video_latents"].shape[0]

    def __getitem__(self, index):
        # A warmup must also restore randomness consumed by data transforms.
        random.random()
        np.random.random()
        torch.rand(())
        return {key: value[index] for key, value in self.sample.items()}


class CompileWarmupTest(unittest.TestCase):
    def test_indexed_warmup_preserves_training_state_and_next_update(self):
        fixture = fixtures.FusedTrainingPathsTest
        model = fixture._model(checkpoint=True, noise_probability=0.5, action_free=True)
        model.fuse_training_paths = True
        Wan22Trainer._apply_dit_only_train_mode(model)
        reference = copy.deepcopy(model)
        model.compile_training_denoise = True
        model.mot.compile_training_layers = True
        source_sample = fixture._sample(padded=True)
        indexed = {key: value for key, value in source_sample.items() if key != "video"}
        indexed["video_latents"] = model.vae.encode(source_sample["video"])
        indexed["video_shape"] = torch.tensor([[3, 9, 16, 16]] * 4)
        indexed["vae_cache_keys"] = [f"key-{i}" for i in range(4)]
        indexed["vae_cache_namespace"] = [WarmupDataset.vae_cache_namespace] * 4
        dataset = WarmupDataset(indexed)
        sampler = ResumableEpochSampler(dataset, seed=123, batch_size=4, num_processes=1)
        loader = DataLoader(dataset, batch_size=4, sampler=sampler, collate_fn=VaeCacheCollator())
        trainer = Wan22Trainer.__new__(Wan22Trainer)
        trainer.model = model
        trainer.vae_cache_dir = "/unused/mock-cache"
        trainer.train_dataset, trainer.train_sampler, trainer.train_loader = dataset, sampler, loader
        trainer.batch_size = 4
        trainer.accelerator = SimpleNamespace(device=torch.device("cpu"), autocast=nullcontext)
        trainer.optimizer = torch.optim.AdamW(model.dit.parameters())
        weights = copy.deepcopy(model.state_dict())
        sampler_state = vars(sampler).copy()
        step = model._anticip_step
        torch_rng, python_rng, numpy_rng = torch.get_rng_state(), random.getstate(), np.random.get_state()

        cache = SimpleNamespace(namespace=dataset.vae_cache_namespace)
        cache.accept_precomputed = lambda latents, *args, **kwargs: latents
        torch._dynamo.reset()
        compile_fn = torch.compile
        with (
            patch("rift.trainer.VaeLatentCache", return_value=cache) as cache_factory,
            patch("torch.compile", side_effect=lambda fn, **kw: compile_fn(fn, backend="aot_eager", **kw)),
        ):
            trainer._warmup_compiled_training()
        cache_factory.assert_called_once()
        self.assertFalse(hasattr(model, "_vae_latent_cache"))
        self.assertEqual(model._anticip_step, step)
        self.assertEqual(vars(sampler), sampler_state)
        self.assertFalse(trainer.optimizer.state)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        torch.testing.assert_close(model.state_dict(), weights, rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), torch_rng, rtol=0, atol=0)
        self.assertEqual(random.getstate(), python_rng)
        np.testing.assert_equal(np.random.get_state(), numpy_rng)
        expected, actual = fixture._run(reference, source_sample), fixture._run(model, source_sample)
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
        torch.testing.assert_close(actual["rng"], expected["rng"], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
