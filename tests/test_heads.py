from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rift.models.wan22.rift_model import (  # noqa: E402
    AnticipFMHead,
    RIFTModel,
    ProbeRMSLinear,
)


class RIFTHeadTest(unittest.TestCase):
    @staticmethod
    def _build_lightweight_model() -> RIFTModel:
        model = RIFTModel.__new__(RIFTModel)
        nn.Module.__init__(model)

        video_expert = nn.Module()
        video_expert.video_attention_mask_mode = "first_frame_causal"
        video_expert.action_conditioned = False
        video_expert.seperated_timestep = True
        video_expert.fuse_vae_embedding_in_latents = True
        video_expert.hidden_dim = 12
        video_expert.patch_size = (1, 2, 2)
        video_expert.out_dim = 6

        model.video_expert = video_expert
        model.mot = nn.Module()
        model.device = torch.device("cpu")
        model.torch_dtype = torch.float32
        model.build_anticip(future_tokens=4, fm_width=16, fm_blocks=1)
        return model

    def test_l2_probe_detaches_input_but_trains_its_parameters(self) -> None:
        torch.manual_seed(0)
        probe = ProbeRMSLinear(8, 5)
        tokens = torch.randn(2, 3, 8, requires_grad=True)

        output = probe(tokens)
        self.assertEqual(output.shape, (2, 3, 5))
        output.square().mean().backward()

        self.assertIsNone(tokens.grad)
        self.assertIsNotNone(probe.weight.grad)
        self.assertIsNotNone(probe.bias.grad)
        self.assertGreater(probe.weight.grad.abs().sum().item(), 0.0)

    def test_fm_head_shape_and_gradient_flow(self) -> None:
        torch.manual_seed(0)
        head = AnticipFMHead(hidden_dim=12, patch_dim=6, width=16, blocks=1)
        x_t = torch.randn(2, 4, 6, requires_grad=True)
        cond = torch.randn(2, 4, 12, requires_grad=True)
        t = torch.tensor([0.25, 0.75])

        velocity = head(x_t, t, cond)
        self.assertEqual(velocity.shape, x_t.shape)
        velocity.square().mean().backward()

        self.assertIsNotNone(x_t.grad)
        self.assertIsNotNone(cond.grad)
        self.assertGreater(cond.grad.abs().sum().item(), 0.0)

    def test_state_dict_key_contract(self) -> None:
        model = self._build_lightweight_model()
        self.assertIsInstance(model.mot.anticip.head, ProbeRMSLinear)
        self.assertIsInstance(model.mot.anticip.head_fm, AnticipFMHead)
        keys = set(model.state_dict())

        self.assertIn("mot.anticip.basis", keys)
        self.assertIn("mot.anticip.head.weight", keys)
        self.assertIn("mot.anticip.head.bias", keys)
        for key in (
            "mot.anticip.head_fm.in_x.weight",
            "mot.anticip.head_fm.in_c.weight",
            "mot.anticip.head_fm.t_mlp.0.weight",
            "mot.anticip.head_fm.blocks.0.q.weight",
            "mot.anticip.head_fm.blocks.0.norm_q.weight",
            "mot.anticip.head_fm.out.1.weight",
        ):
            self.assertIn(key, keys)
        self.assertFalse(any("commit" in k for k in keys))

        # ProbeRMSLinear intentionally preserves nn.Linear checkpoint names.
        plain = nn.Linear(12, 6)
        probe = ProbeRMSLinear(12, 6)
        self.assertEqual(set(plain.state_dict()), set(probe.state_dict()))
        probe.load_state_dict(plain.state_dict(), strict=True)


if __name__ == "__main__":
    unittest.main()
