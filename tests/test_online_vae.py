from __future__ import annotations

import copy
from contextlib import nullcontext
from functools import partial
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

import test_fused_paths as fixtures
from rift.models.wan22.helpers.vae_encoding import OnlineVaeEncoder
from rift.models.wan22.wan_video_vae import VideoVAE38_, WanVideoVAE38
from rift.trainer import Wan22Trainer
from rift.utils.samplers import ResumableEpochSampler
from rift.utils.vae_latent_cache import encoding_identity


def tiny_vae():
    # Exercise the real temporal encoder and scales, with small encoder/decoder widths.
    with patch(
        "rift.models.wan22.wan_video_vae.VideoVAE38_",
        side_effect=partial(VideoVAE38_, dec_dim=4),
    ):
        return WanVideoVAE38(dim=4).eval().requires_grad_(False)


class OnlineVaeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(123)
        self.vae = tiny_vae()
        self.video = torch.randn(3, 3, 9, 32, 32)

    def test_batching_and_partial_chunks_match_sequential_encoding(self):
        with torch.no_grad():
            expected = self.vae.encode(self.video, device="cpu")
        for batch_size in (None, 1, 2):
            with self.subTest(batch_size=batch_size):
                encoder = OnlineVaeEncoder(self.vae, cudagraphs=False, batch_size=batch_size)
                with patch.object(self.vae.model, "encode", wraps=self.vae.model.encode) as encode:
                    actual = encoder(self.video)
                self.assertEqual(encode.call_count, 1 if batch_size is None else (3 + batch_size - 1) // batch_size)
                self.assertFalse(actual.requires_grad)
                torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)

    def test_online_encoding_preserves_cache_identity_and_checkpoint_tensors(self):
        identity = encoding_identity(self.vae, torch.float32, "cpu")
        state = copy.deepcopy(self.vae.state_dict())
        scales = [scale.clone() for scale in self.vae.scale]
        OnlineVaeEncoder(self.vae, cudagraphs=False, batch_size=None)(self.video)
        self.assertEqual(encoding_identity(self.vae, torch.float32, "cpu"), identity)
        torch.testing.assert_close(self.vae.state_dict(), state, rtol=0, atol=0)
        torch.testing.assert_close(self.vae.scale, scales, rtol=0, atol=0)

    def test_device_scale_copy_refreshes_after_constant_changes(self):
        encoder = OnlineVaeEncoder(self.vae, cudagraphs=False, batch_size=None)
        original = encoder(self.video)
        self.vae.scale[0].add_(0.25)
        actual = encoder(self.video)
        with torch.no_grad():
            expected = self.vae.encode(self.video, device="cpu")
        self.assertFalse(torch.equal(actual, original))
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)

    def test_cudagraphs_require_cuda_and_online_encoder_requires_frozen_eval(self):
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            OnlineVaeEncoder(self.vae, cudagraphs=True, batch_size=None)(self.video)
        self.vae.train()
        with self.assertRaisesRegex(RuntimeError, "eval mode"):
            OnlineVaeEncoder(self.vae, cudagraphs=False, batch_size=None)(self.video)
        self.vae.eval().requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError, "frozen"):
            OnlineVaeEncoder(self.vae, cudagraphs=False, batch_size=None)(self.video)

    def test_training_loss_and_gradients_with_batched_online_inputs(self):
        fixture = fixtures.FusedTrainingPathsTest
        reference = fixture._model(checkpoint=True, noise_probability=0.5, action_free=True)
        reference.vae = self.vae
        reference.fuse_training_paths = True
        Wan22Trainer._apply_dit_only_train_mode(reference)
        batched = copy.deepcopy(reference)
        batched.vae_encode_mode, batched.vae_encode_batch_size = "batched", 2
        sample = fixture._sample(padded=True)
        sample["video"] = sample["video"].repeat_interleave(4, dim=-1).repeat_interleave(4, dim=-2)
        expected, actual = fixture._run(reference, sample), fixture._run(batched, sample)
        for key in ("loss", "metrics", "outputs", "grads", "keep_rows"):
            torch.testing.assert_close(actual[key], expected[key], rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(actual["rng"], expected["rng"], rtol=0, atol=0)
        self.assertEqual(actual["step"], expected["step"])

    def test_existing_cache_routes_bypass_online_encoder(self):
        fixture = fixtures.FusedTrainingPathsTest
        model = fixture._model(
            checkpoint=False, noise_probability=0.0, action_free=True,
            vae_encode_mode="cudagraphs",
        )
        sample = fixture._sample(padded=False)
        latents = model.vae.encode(sample["video"])
        cache = SimpleNamespace(
            encode=Mock(return_value=latents), accept_precomputed=Mock(return_value=latents),
        )
        model._vae_latent_cache = cache
        sample["vae_cache_keys"] = [f"key-{i}" for i in range(4)]
        with patch.object(model, "_encode_video_latents", side_effect=AssertionError("online encode")):
            torch.testing.assert_close(model.build_inputs(sample)["input_latents"], latents)
            indexed = {key: value for key, value in sample.items() if key != "video"}
            indexed.update(
                video_latents=latents,
                video_shape=torch.tensor([[3, 9, 16, 16]] * 4),
                vae_cache_namespace=["fixture"] * 4,
            )
            torch.testing.assert_close(model.build_inputs(indexed)["input_latents"], latents)
        cache.encode.assert_called_once()
        cache.accept_precomputed.assert_called_once()
        self.assertFalse(hasattr(model, "_online_vae_encoder"))

    def test_vae_only_warmup_does_not_run_denoiser_or_advance_curriculum(self):
        fixture = fixtures.FusedTrainingPathsTest
        model = fixture._model(
            checkpoint=True, noise_probability=0.5, action_free=True,
            vae_encode_mode="cudagraphs", vae_encode_batch_size=2,
        )
        model.vae = self.vae
        Wan22Trainer._apply_dit_only_train_mode(model)
        sample = fixture._sample(padded=False)
        sample["video"] = sample["video"].repeat_interleave(4, dim=-1).repeat_interleave(4, dim=-2)
        dataset = [{key: value[i] for key, value in sample.items()} for i in range(4)]
        trainer = Wan22Trainer.__new__(Wan22Trainer)
        trainer.model, trainer.vae_cache_dir, trainer.batch_size = model, None, 4
        trainer.train_dataset = dataset
        trainer.train_sampler = ResumableEpochSampler(dataset, seed=123, batch_size=4, num_processes=1)
        trainer.train_loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        trainer.accelerator = SimpleNamespace(device=torch.device("cpu"), autocast=nullcontext)
        step, rng = model._anticip_step, torch.get_rng_state()
        # CUDA replay is checked on GPU; use the real batched encoder for this
        # CPU test of the VAE-only trainer warmup and state restoration.
        encoder = OnlineVaeEncoder(self.vae, cudagraphs=False, batch_size=2)
        with (
            patch("rift.models.wan22.fastwam.OnlineVaeEncoder", return_value=encoder),
            patch.object(model, "training_loss", side_effect=AssertionError("denoiser warmup")),
            patch.object(self.vae.model, "encode", wraps=self.vae.model.encode) as encode,
        ):
            trainer._warmup_compiled_training()
        self.assertTrue(encode.called)
        self.assertEqual(model._anticip_step, step)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)

    def test_invalid_online_encoding_configuration_is_rejected(self):
        fixture = fixtures.FusedTrainingPathsTest
        for options in ({"vae_encode_mode": "unknown"}, {"vae_encode_batch_size": 0},
                        {"vae_encode_batch_size": -1}, {"vae_encode_batch_size": 1.5}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                fixture._model(checkpoint=False, noise_probability=0.0, action_free=True, **options)


if __name__ == "__main__":
    unittest.main()
