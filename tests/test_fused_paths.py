from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rift.models.wan22.action_dit import ActionDiT  # noqa: E402
from rift.models.wan22.mot import MoT  # noqa: E402
from rift.models.wan22.rift_model import RIFTModel  # noqa: E402
from rift.models.wan22.wan_video_dit import WanVideoDiT  # noqa: E402


class SamplingVAE(nn.Module):
    """Keep the real input pipeline without pretrained VAE weights."""

    temporal_downsample_factor = 4

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("channel_scale", torch.linspace(0.5, 1.5, 48))

    def encode(self, video: torch.Tensor, **kwargs) -> torch.Tensor:
        return (
            video[:, :1, ::4, ::4, ::4]
            * self.channel_scale.view(1, -1, 1, 1, 1)
        )


class FusedTrainingPathsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_num_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_num_threads)

    @staticmethod
    def _model(
        *, checkpoint: bool, noise_probability: float, action_free: bool,
        train_probe_head: bool = True,
        compile_training_denoise: bool = False,
        vae_encode_mode: str = "sequential",
        vae_encode_batch_size: int | None = None,
    ) -> RIFTModel:
        torch.manual_seed(1234)
        video = WanVideoDiT(
            hidden_dim=24,
            in_dim=48,
            ffn_dim=32,
            out_dim=48,
            text_dim=12,
            freq_dim=8,
            eps=1e-6,
            patch_size=(1, 2, 2),
            num_heads=2,
            attn_head_dim=8,
            num_layers=2,
            has_image_input=False,
            seperated_timestep=True,
            fuse_vae_embedding_in_latents=True,
            action_conditioned=False,
            video_attention_mask_mode="first_frame_causal",
            use_gradient_checkpointing=checkpoint,
        )
        action = ActionDiT(
            hidden_dim=16,
            action_dim=3,
            ffn_dim=24,
            text_dim=12,
            freq_dim=8,
            eps=1e-6,
            num_heads=2,
            attn_head_dim=8,
            num_layers=2,
            use_gradient_checkpointing=checkpoint,
        )
        model = RIFTModel(
            video_expert=video,
            action_expert=action,
            mot=MoT(
                mixtures={"video": video, "action": action},
                mot_checkpoint_mixed_attn=checkpoint,
            ),
            vae=SamplingVAE(),
            text_dim=12,
            proprio_dim=2,
            device="cpu",
            torch_dtype=torch.float32,
            loss_lambda_video=0.7,
            compile_training_denoise=compile_training_denoise,
            vae_encode_mode=vae_encode_mode,
            vae_encode_batch_size=vae_encode_batch_size,
        )
        model.build_anticip(
            future_tokens=8,
            latent_t=3,
            lambda_anticip=0.4,
            lambda_action_anticip=0.8,
            delta_mix=0.6,
            anneal_start_frac=0.5,
            anneal_total_steps=10,
            cond_noise_p=noise_probability,
            cond_noise_sigma=0.2,
            cond_noise_ramp_start_frac=None,
            cond_noise_action_free=action_free,
            fm_lambda=0.9,
            fm_width=16,
            fm_blocks=1,
            train_probe_head=train_probe_head,
        )
        model._anticip_step = 7
        model.train()
        return model

    @staticmethod
    def _sample(*, padded: bool, latent_frames: int = 3) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(5678)
        num_frames = 1 + 4 * (latent_frames - 1)
        sample = {
            "video": torch.randn(4, 3, num_frames, 16, 16, generator=generator),
            "action": torch.randn(4, num_frames - 1, 3, generator=generator),
            "proprio": torch.randn(4, num_frames - 1, 2, generator=generator),
            "context": torch.randn(4, 4, 12, generator=generator),
            "context_mask": torch.tensor(
                [[True, True, False, False], [True, False, False, False],
                 [True, True, True, False], [False, False, False, False]]
            ),
        }
        if padded:
            image_pad = torch.zeros(4, num_frames, dtype=torch.bool)
            image_pad[1, 5:] = True
            image_pad[2, 3:] = True
            image_pad[3, 1:] = True
            action_pad = torch.zeros(4, num_frames - 1, dtype=torch.bool)
            action_pad[1, 4:] = True
            action_pad[2, 2:] = True
            action_pad[3] = True
            sample.update(image_is_pad=image_pad, action_is_pad=action_pad)
        return sample

    @staticmethod
    def _run(model: RIFTModel, sample: dict) -> dict:
        outputs = []

        def capture_output(module, args, output):
            outputs.append({name: tokens.detach().clone() for name, tokens in output.items()})

        handle = model.mot.register_forward_hook(capture_output)
        try:
            with patch.object(model, "_action_loss", wraps=model._action_loss) as action_loss:
                torch.manual_seed(42)
                loss, metrics = model.training_loss(sample)
                loss.backward()
                keep_rows = action_loss.call_args.kwargs["keep_rows"]
                return {
                    "loss": loss.detach(),
                    "metrics": metrics,
                    "outputs": outputs,
                    "keep_rows": None if keep_rows is None else keep_rows.clone(),
                    "rng": torch.get_rng_state().clone(),
                    "step": model._anticip_step,
                    "grads": {
                        name: None if parameter.grad is None else parameter.grad.detach().clone()
                        for name, parameter in model.named_parameters()
                    },
                }
        finally:
            handle.remove()

    def _assert_parity(
        self,
        *,
        checkpoint: bool = False,
        padded: bool = True,
        noise_probability: float = 0.5,
        action_free: bool = True,
        latent_frames: int = 3,
        train_probe_head: bool = True,
    ) -> dict:
        separate = self._model(
            checkpoint=checkpoint,
            noise_probability=noise_probability,
            action_free=action_free,
            train_probe_head=train_probe_head,
        )
        separate.fuse_training_paths = False
        fused = copy.deepcopy(separate)
        fused.fuse_training_paths = True
        sample = self._sample(padded=padded, latent_frames=latent_frames)
        expected = self._run(separate, sample)
        actual = self._run(fused, sample)

        torch.testing.assert_close(actual["loss"], expected["loss"], rtol=3e-5, atol=3e-6)
        self.assertEqual(actual["metrics"].keys(), expected["metrics"].keys())
        for name in expected["metrics"]:
            torch.testing.assert_close(
                actual["metrics"][name], expected["metrics"][name], rtol=3e-5, atol=3e-6,
                msg=lambda message: f"Metric {name}: {message}",
            )
        self.assertEqual(expected["step"], 8)
        self.assertEqual(actual["step"], expected["step"])
        torch.testing.assert_close(actual["rng"], expected["rng"], rtol=0, atol=0)
        if expected["keep_rows"] is None:
            self.assertIsNone(actual["keep_rows"])
        else:
            torch.testing.assert_close(actual["keep_rows"], expected["keep_rows"])

        self.assertEqual(len(expected["outputs"]), 2)
        if latent_frames == 3:
            self.assertEqual(len(actual["outputs"]), 1)
            combined = actual["outputs"][0]
            torch.testing.assert_close(combined["video"][:4], expected["outputs"][0]["video"])
            for expert in ("video", "action"):
                torch.testing.assert_close(combined[expert][4:], expected["outputs"][1][expert])
        else:
            self.assertEqual(len(actual["outputs"]), 2)
            for actual_path, expected_path in zip(actual["outputs"], expected["outputs"]):
                for expert in ("video", "action"):
                    torch.testing.assert_close(actual_path[expert], expected_path[expert])

        self.assertEqual(actual["grads"].keys(), expected["grads"].keys())
        for name, expected_grad in expected["grads"].items():
            actual_grad = actual["grads"][name]
            if expected_grad is None:
                self.assertIsNone(actual_grad, name)
                continue
            self.assertIsNotNone(actual_grad, name)
            self.assertTrue(torch.isfinite(actual_grad).all(), name)
            torch.testing.assert_close(
                actual_grad, expected_grad, rtol=3e-5, atol=3e-6,
                msg=lambda message, name=name: f"Gradient {name}: {message}",
            )
        active_heads = ["mot.anticip.basis", "mot.anticip.head_fm.in_c.weight"]
        if train_probe_head:
            active_heads.append("mot.anticip.head.weight")
        for name in active_heads:
            self.assertGreater(actual["grads"][name].abs().sum().item(), 0.0, name)
        return actual

    def test_padded_training_matches_with_and_without_checkpointing(self) -> None:
        for checkpoint in (False, True):
            with self.subTest(checkpoint=checkpoint):
                actual = self._assert_parity(checkpoint=checkpoint)
                keep = actual["keep_rows"]
                self.assertTrue(keep.any())
                self.assertTrue((~keep).any())
                self.assertGreater(actual["metrics"]["loss_action_anticip"], 0.0)

    def test_unpadded_training_matches_without_condition_noise(self) -> None:
        self._assert_parity(padded=False, noise_probability=0.0)

    def test_all_action_rows_excluded_still_train_future_heads(self) -> None:
        actual = self._assert_parity(checkpoint=True, noise_probability=1.0)
        self.assertFalse(actual["keep_rows"].any())
        self.assertEqual(actual["metrics"]["loss_action_anticip"], 0.0)

    def test_shared_condition_noise_matches(self) -> None:
        actual = self._assert_parity(noise_probability=1.0, action_free=False)
        self.assertIsNone(actual["keep_rows"])

    def test_unequal_video_grids_keep_separate_forwards(self) -> None:
        self._assert_parity(checkpoint=True, latent_frames=4)

    def test_disabled_probe_preserves_initialization_and_leaves_no_optimizer_parameters(self) -> None:
        enabled = self._model(checkpoint=False, noise_probability=0.5, action_free=True)
        enabled_rng = torch.get_rng_state().clone()
        disabled = self._model(
            checkpoint=False, noise_probability=0.5, action_free=True, train_probe_head=False,
        )
        torch.testing.assert_close(torch.get_rng_state(), enabled_rng, rtol=0, atol=0)
        self.assertIsNone(disabled.mot.anticip.head)
        self.assertIsNotNone(disabled.mot.anticip.head_fm)
        self.assertTrue(disabled.mot.anticip.basis.requires_grad)

        enabled_parameters = dict(enabled.named_parameters())
        disabled_parameters = dict(disabled.named_parameters())
        removed_parameters = set(enabled_parameters) - set(disabled_parameters)
        self.assertEqual(removed_parameters, {"mot.anticip.head.weight", "mot.anticip.head.bias"})
        for name, parameter in disabled_parameters.items():
            torch.testing.assert_close(parameter, enabled_parameters[name], rtol=0, atol=0)
        self.assertFalse(any(".anticip.head." in name for name in disabled.state_dict()))

        trainable = list(disabled.dit.parameters()) + list(disabled.proprio_encoder.parameters())
        optimizer = torch.optim.AdamW(trainable)
        optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        self.assertEqual(optimizer_ids, {id(parameter) for parameter in disabled_parameters.values()})
        self.assertIn(id(disabled.mot.anticip.basis), optimizer_ids)
        self.assertTrue(all(id(parameter) in optimizer_ids for parameter in disabled.mot.anticip.head_fm.parameters()))

    def test_disabled_probe_preserves_shared_raw_gradients_across_checkpoint_and_fusion_modes(self) -> None:
        for checkpoint in (False, True):
            for fused in (False, True):
                with self.subTest(checkpoint=checkpoint, fused=fused):
                    enabled = self._model(checkpoint=checkpoint, noise_probability=0.5, action_free=True)
                    disabled = self._model(
                        checkpoint=checkpoint, noise_probability=0.5, action_free=True,
                        train_probe_head=False,
                    )
                    enabled.fuse_training_paths = disabled.fuse_training_paths = fused
                    sample = self._sample(padded=True)
                    expected = self._run(enabled, sample)
                    actual = self._run(disabled, sample)
                    self.assertEqual(actual["metrics"]["loss_anticip"], 0.0)
                    self.assertGreater(expected["metrics"]["loss_anticip"], 0.0)
                    torch.testing.assert_close(
                        actual["loss"], expected["loss"] - expected["metrics"]["loss_anticip"],
                        rtol=3e-5, atol=3e-6,
                    )
                    for name in expected["metrics"]:
                        if name != "loss_anticip":
                            self.assertEqual(actual["metrics"][name], expected["metrics"][name], name)
                    torch.testing.assert_close(actual["rng"], expected["rng"], rtol=0, atol=0)
                    torch.testing.assert_close(actual["keep_rows"], expected["keep_rows"])
                    self.assertEqual(actual["step"], expected["step"])
                    self.assertEqual(len(actual["outputs"]), 1 if fused else 2)
                    for actual_path, expected_path in zip(actual["outputs"], expected["outputs"]):
                        for expert in ("video", "action"):
                            torch.testing.assert_close(actual_path[expert], expected_path[expert], rtol=0, atol=0)
                    for name, actual_grad in actual["grads"].items():
                        expected_grad = expected["grads"][name]
                        if expected_grad is None:
                            self.assertIsNone(actual_grad, name)
                        else:
                            torch.testing.assert_close(
                                actual_grad, expected_grad, rtol=3e-5, atol=3e-6,
                                msg=lambda message, name=name: f"Gradient {name}: {message}",
                            )
                    for name in ("mot.anticip.basis", "mot.anticip.head_fm.in_c.weight"):
                        self.assertGreater(actual["grads"][name].abs().sum().item(), 0.0, name)

    def test_fused_parity_with_probe_disabled(self) -> None:
        for checkpoint in (False, True):
            with self.subTest(checkpoint=checkpoint):
                self._assert_parity(checkpoint=checkpoint, train_probe_head=False)


if __name__ == "__main__":
    unittest.main()
