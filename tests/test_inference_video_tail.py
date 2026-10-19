from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_fused_paths as fixtures  # noqa: E402


class InferenceSamplingVAE(fixtures.SamplingVAE):
    """Use the native image-encoding call with deterministic tiny latents."""

    def encode(self, video, **kwargs):
        if isinstance(video, list):
            return [super(InferenceSamplingVAE, self).encode(item.unsqueeze(0))[0] for item in video]
        return super().encode(video, **kwargs)


class InferenceVideoTailTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_num_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_num_threads)

    @staticmethod
    def _model(dtype=torch.float32, layers=2):
        model = fixtures.FusedTrainingPathsTest._model(
            checkpoint=False, noise_probability=0.0, action_free=True,
        )
        model.vae = InferenceSamplingVAE()
        if layers != model.mot.num_layers:
            model.video_expert.blocks = model.video_expert.blocks[:layers]
            model.action_expert.blocks = model.action_expert.blocks[:layers]
            model.mot.num_layers = layers
        model.to(dtype=dtype).eval()
        model.torch_dtype = dtype
        return model

    @staticmethod
    def _inputs(model):
        generator = torch.Generator().manual_seed(7654)
        dtype = model.torch_dtype
        context = torch.randn(2, 4, 12, generator=generator, dtype=dtype)
        context_mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
        first_frame = torch.randn(2, 48, 1, 4, 4, generator=generator, dtype=dtype)
        pre, tokens = model._anticip_video_pre(
            first_frame_latents=first_frame, context=context, context_mask=context_mask,
        )
        mask = model._build_anticip_attention_mask(
            video_seq_len=tokens.shape[1], action_seq_len=8,
            video_tokens_per_frame=int(pre["meta"]["tokens_per_frame"]), device=tokens.device,
        )
        cache_kwargs = dict(
            video_tokens=tokens, video_freqs=pre["freqs"], video_t_mod=pre["t_mod"],
            video_context_payload={"context": pre["context"], "mask": pre["context_mask"]},
            video_attention_mask=mask[:tokens.shape[1], :tokens.shape[1]],
        )
        action_kwargs = dict(
            latents_action=torch.randn(2, 8, 3, generator=generator, dtype=dtype),
            timestep_action=torch.full((2,), 400.0, dtype=dtype),
            context=context, context_mask=context_mask,
            attention_mask=mask, video_seq_len=tokens.shape[1],
        )
        return cache_kwargs, action_kwargs

    @staticmethod
    def _count_video_calls(model, stack):
        counts = Counter()
        for index, block in enumerate(model.video_expert.blocks):
            for name, module in (
                ("q", block.self_attn.q), ("k", block.self_attn.k),
                ("v", block.self_attn.v), ("o", block.self_attn.o),
                ("cross", block.cross_attn), ("ffn", block.ffn),
            ):
                def count(_module, _args, _output, key=(index, name)):
                    counts[key] += 1
                handle = module.register_forward_hook(count)
                stack.callback(handle.remove)
        return counts

    def test_cache_and_action_are_bitwise_equal(self):
        for dtype in (torch.float32, torch.bfloat16):
            for layers in (1, 2):
                with self.subTest(dtype=dtype, layers=layers), torch.no_grad():
                    model = self._model(dtype, layers)
                    cache_kwargs, action_kwargs = self._inputs(model)
                    keys = tuple(model.state_dict())
                    rng = torch.get_rng_state().clone()
                    model.mot.optimize_video_prefill = False
                    reference = model.mot.prefill_video_cache(**cache_kwargs)
                    reference_action = model._predict_action_noise_with_cache(
                        video_kv_cache=reference, **action_kwargs,
                    )
                    model.mot.optimize_video_prefill = True
                    candidate = model.mot.prefill_video_cache(**cache_kwargs)
                    candidate_action = model._predict_action_noise_with_cache(
                        video_kv_cache=candidate, **action_kwargs,
                    )
                    self.assertEqual(len(candidate), layers)
                    for old, new in zip(reference, candidate):
                        for name in ("k", "v"):
                            self.assertTrue(torch.equal(old[name], new[name]))
                    self.assertTrue(torch.equal(reference_action, candidate_action))
                    self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                    self.assertEqual(keys, tuple(model.state_dict()))

    def test_only_final_video_query_and_post_block_are_skipped(self):
        model = self._model()
        with torch.no_grad():
            cache_kwargs, _ = self._inputs(model)
            with ExitStack() as stack:
                counts = self._count_video_calls(model, stack)
                attention = stack.enter_context(patch.object(
                    model.mot, "_mixed_attention", wraps=model.mot._mixed_attention,
                ))
                model.mot.prefill_video_cache(**cache_kwargs)
                self.assertEqual(attention.call_count, model.mot.num_layers - 1)
            for index in range(model.mot.num_layers):
                final = index == model.mot.num_layers - 1
                for name in ("k", "v"):
                    self.assertEqual(counts[index, name], 1)
                for name in ("q", "o", "cross", "ffn"):
                    self.assertEqual(counts[index, name], int(not final))

    def test_grad_enabled_and_training_prefill_keep_original_execution(self):
        for training, grad_enabled in ((False, True), (True, False), (True, True)):
            with self.subTest(training=training, grad_enabled=grad_enabled):
                model = self._model().train(training)
                with torch.set_grad_enabled(grad_enabled):
                    cache_kwargs, _ = self._inputs(model)
                    with ExitStack() as stack:
                        counts = self._count_video_calls(model, stack)
                        attention = stack.enter_context(patch.object(
                            model.mot, "_mixed_attention", wraps=model.mot._mixed_attention,
                        ))
                        cache = model.mot.prefill_video_cache(**cache_kwargs)
                        self.assertEqual(attention.call_count, model.mot.num_layers)
                    for index in range(model.mot.num_layers):
                        for name in ("q", "k", "v", "o", "cross", "ffn"):
                            self.assertEqual(counts[index, name], 1)
                    self.assertEqual(cache[-1]["k"].requires_grad, grad_enabled)

    def test_grad_enabled_prefill_gradients_are_unchanged(self):
        original = self._model()
        original.mot.optimize_video_prefill = False
        candidate = copy.deepcopy(original)
        candidate.mot.optimize_video_prefill = True
        for model in (original, candidate):
            cache_kwargs, _ = self._inputs(model)
            cache = model.mot.prefill_video_cache(**cache_kwargs)
            sum(value.square().mean() for layer in cache for value in layer.values()).backward()
        for (name, old), (new_name, new) in zip(original.named_parameters(), candidate.named_parameters()):
            self.assertEqual(name, new_name)
            if old.grad is None:
                self.assertIsNone(new.grad, name)
            else:
                self.assertTrue(torch.equal(old.grad, new.grad), name)

    def test_multistep_native_inference_preserves_action_and_rng(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = self._model(dtype)
                generator = torch.Generator().manual_seed(812)
                kwargs = dict(
                    prompt=None, input_image=torch.randn(3, 16, 16, generator=generator),
                    action_horizon=8, num_video_frames=9,
                    context=torch.randn(4, 12, generator=generator),
                    context_mask=torch.tensor([True, True, False, False]),
                    proprio=torch.randn(2, generator=generator),
                    num_inference_steps=4, seed=77,
                )
                rng = torch.get_rng_state().clone()
                model.mot.optimize_video_prefill = False
                reference = model.infer_action(**kwargs)["action"]
                model.mot.optimize_video_prefill = True
                with ExitStack() as stack:
                    counts = self._count_video_calls(model, stack)
                    prefill = stack.enter_context(patch.object(
                        model.mot, "prefill_video_cache", wraps=model.mot.prefill_video_cache,
                    ))
                    action_forward = stack.enter_context(patch.object(
                        model.mot, "forward_action_with_video_cache",
                        wraps=model.mot.forward_action_with_video_cache,
                    ))
                    actual = model.infer_action(**kwargs)["action"]
                    self.assertEqual(prefill.call_count, 1)
                    self.assertEqual(action_forward.call_count, 4)
                self.assertTrue(torch.equal(reference, actual))
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                for index in range(model.mot.num_layers):
                    for name in ("k", "v"):
                        self.assertEqual(counts[index, name], 1)
                    self.assertEqual(counts[index, "q"], int(index < model.mot.num_layers - 1))

    def test_invalid_masks_still_raise(self):
        model = self._model()
        with torch.no_grad():
            cache_kwargs, _ = self._inputs(model)
            for mask, message in (
                (torch.ones(12, dtype=torch.bool), "must be 2D"),
                (torch.ones(12, 11, dtype=torch.bool), "must be square"),
                (torch.ones(11, 11, dtype=torch.bool), "seq length mismatch"),
            ):
                with self.subTest(shape=tuple(mask.shape)), self.assertRaisesRegex(ValueError, message):
                    model.mot.prefill_video_cache(**{**cache_kwargs, "video_attention_mask": mask})


if __name__ == "__main__":
    unittest.main()
