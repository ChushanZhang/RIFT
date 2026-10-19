from __future__ import annotations

import copy
from contextlib import nullcontext
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import test_fused_paths as fixtures
from rift.models.wan22.helpers.training_acceleration import (
    FutureHeadExecution, aot_config, stock_video_head, video_training_prediction,
)
from rift.models.wan22.wan_video_dit import Head, WanVideoDiT
from rift.trainer import make_training_optimizer


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def model():
    return fixtures.FusedTrainingPathsTest._model(
        checkpoint=True, noise_probability=0.3, action_free=True,
    )


@pytest.mark.parametrize("shared_time", [False, True])
def test_future_only_head_preserves_values_gradients_rng_and_keys(shared_time):
    reference = model().video_expert
    candidate = copy.deepcopy(reference)
    tokens = torch.randn(2, 12, reference.hidden_dim, requires_grad=True)
    time = torch.randn(2, 1 if shared_time else 12, reference.hidden_dim, requires_grad=True)
    candidate_tokens = tokens.detach().clone().requires_grad_()
    candidate_time = time.detach().clone().requires_grad_()
    pre = {"meta": {"grid_size": (3, 2, 2)}, "t": time}
    candidate_pre = {**pre, "t": candidate_time}
    assert stock_video_head(candidate)
    state = copy.deepcopy(candidate.state_dict())
    rng = torch.get_rng_state()
    expected = reference.post_dit(tokens, pre)[:, :, 1:]
    actual = video_training_prediction(
        candidate, candidate_tokens, candidate_pre, enabled=True, training=True,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(candidate_tokens.grad, tokens.grad)
    torch.testing.assert_close(candidate_time.grad, time.grad)
    for (name, left), (_, right) in zip(reference.named_parameters(), candidate.named_parameters()):
        if left.grad is None:
            assert right.grad is None, name
        else:
            torch.testing.assert_close(left.grad, right.grad, msg=name)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    torch.testing.assert_close(candidate.state_dict(), state, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["disabled", "evaluation", "no_grad", "rank_two_time", "forward_hook"])
def test_future_only_head_retains_native_fallback(kind):
    expert = model().video_expert
    tokens = torch.randn(2, 12, expert.hidden_dim, requires_grad=True)
    time_shape = (2, 1, expert.hidden_dim) if kind == "rank_two_time" else (2, 12, expert.hidden_dim)
    time = torch.randn(*time_shape)
    if kind == "rank_two_time":
        time = time[:, 0]
    pre = {"meta": {"grid_size": (3, 2, 2)}, "t": time}
    seen = []
    # A real hook must see the native full token sequence, not the suffix.
    handle = expert.head.register_forward_pre_hook(lambda module, args: seen.append(args[0].shape[1]))
    try:
        expected = expert.post_dit(tokens, pre)[:, :, 1:]
        context = torch.no_grad() if kind == "no_grad" else torch.enable_grad()
        guard = (nullcontext() if kind == "forward_hook" else patch(
            "rift.models.wan22.helpers.training_acceleration.stock_video_head",
            side_effect=AssertionError("fallback must be selected before native-head inspection"),
        ))
        with context, guard:
            actual = video_training_prediction(
                expert, tokens, pre, enabled=kind != "disabled", training=kind != "evaluation",
            )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert seen == [12, 12]
    finally:
        handle.remove()


@pytest.mark.parametrize("owner,name", [
    (WanVideoDiT, "post_dit"), (WanVideoDiT, "unpatchify"),
    (Head, "forward"), (nn.LayerNorm, "forward"), (nn.Linear, "forward"),
])
def test_class_customization_disables_head_shortcut(owner, name):
    expert = model().video_expert
    original = getattr(owner, name)
    def custom(self, *args, **kwargs):
        return original(self, *args, **kwargs)
    with patch.object(owner, name, custom):
        assert not stock_video_head(expert)


@pytest.mark.parametrize("probability", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("fused", [False, True])
def test_native_training_combination_preserves_losses_gradients_rng(probability, fused):
    fixture = fixtures.FusedTrainingPathsTest
    reference = fixture._model(checkpoint=True, noise_probability=probability, action_free=True)
    reference.fuse_training_paths = fused
    candidate = copy.deepcopy(reference)
    candidate.video_future_head_only = True
    # This matches the trainer: the parent is eval, while trainable DiT is train.
    reference.eval()
    reference.dit.train()
    candidate.eval()
    candidate.dit.train()
    sample = fixture._sample(padded=True)
    before = copy.deepcopy(candidate.state_dict())
    expected = fixture._run(reference, sample)
    actual = fixture._run(candidate, sample)
    torch.testing.assert_close(actual["loss"], expected["loss"], rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(actual["rng"], expected["rng"], rtol=0, atol=0)
    assert actual["step"] == expected["step"]
    for name, value in expected["grads"].items():
        if value is None:
            assert actual["grads"][name] is None, name
        else:
            torch.testing.assert_close(actual["grads"][name], value, rtol=3e-4, atol=3e-5, msg=name)
    torch.testing.assert_close(candidate.state_dict(), before, rtol=0, atol=0)


def test_future_compiler_preserves_checkpointed_training_and_eval_fallback():
    fixture = fixtures.FusedTrainingPathsTest
    reference = model()
    candidate = copy.deepcopy(reference)
    candidate.compile_future_head = True
    candidate.video_future_head_only = True
    candidate.eval()
    candidate.dit.train()
    sample = fixture._sample(padded=True)
    state_keys = tuple(candidate.state_dict())
    original_compile = torch.compile
    with patch("torch.compile", side_effect=lambda fn, **kw: original_compile(fn, backend="aot_eager", **kw)) as compiler:
        expected = fixture._run(reference, sample)
        actual = fixture._run(candidate, sample)
    compiler.assert_called_once()
    assert compiler.call_args.kwargs == {"fullgraph": True, "dynamic": False}
    torch.testing.assert_close(actual["loss"], expected["loss"], rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(actual["rng"], expected["rng"], rtol=0, atol=0)
    for name, value in expected["grads"].items():
        if value is None:
            assert actual["grads"][name] is None, name
        else:
            torch.testing.assert_close(actual["grads"][name], value, rtol=3e-4, atol=3e-5, msg=name)
    assert tuple(candidate.state_dict()) == state_keys
    assert not isinstance(candidate._future_head_execution, nn.Module)
    candidate.eval()
    with patch.object(candidate._future_head_execution, "compiled", side_effect=AssertionError("compiled eval")):
        fixture._run(candidate, sample)


@pytest.mark.parametrize("enabled,training,grad", [(False, True, True), (True, False, True), (True, True, False)])
def test_future_compiler_stays_eager_outside_training(enabled, training, grad):
    execution = FutureHeadExecution()
    context = torch.enable_grad() if grad else torch.no_grad()
    with patch("torch.compile", side_effect=AssertionError("unexpected compilation")), context:
        result = execution.run(lambda x, t, c: x + t + c, 1, 2, 3, enabled=enabled, training=training)
    assert result == 6


def test_future_compiler_rebuilds_after_parameter_relocation_without_rng_changes():
    decoder = model().mot.anticip.head_fm
    execution = FutureHeadExecution()
    noisy = torch.randn(2, 8, 192, requires_grad=True)
    timestep = torch.ones(2)
    condition = torch.randn(2, 8, 24, requires_grad=True)
    compiled_functions = []
    original_cache_config = (aot_config.enable_autograd_cache, aot_config.enable_remote_autograd_cache)

    def fake_compile(function, **kwargs):
        assert not aot_config.enable_autograd_cache
        assert not aot_config.enable_remote_autograd_cache
        torch.rand(3)  # Compiler construction must also be outside training RNG.
        first = True
        compiled_functions.append(function)
        def compiled(*args):
            nonlocal first
            if first:
                assert not aot_config.enable_autograd_cache
                assert not aot_config.enable_remote_autograd_cache
                torch.rand(5)  # Simulate RNG use during cold compiler initialization.
                first = False
            return function(*args)
        return compiled

    expected = decoder(noisy, timestep, condition)
    rng = torch.get_rng_state()
    keys = tuple(decoder.state_dict())
    with patch("torch.compile", side_effect=fake_compile), patch("torch._dynamo.reset_code") as reset:
        for index in range(3):
            if index == 2:
                for parameter in decoder.parameters():
                    storage = torch.empty(parameter.numel() + 1, dtype=parameter.dtype)
                    storage[1:].copy_(parameter.detach().reshape(-1))
                    parameter.data = storage[1:].view_as(parameter)
            actual = execution.run(
                decoder, noisy, timestep, condition, enabled=True, training=True,
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
            assert (aot_config.enable_autograd_cache, aot_config.enable_remote_autograd_cache) == original_cache_config
        assert len(compiled_functions) == reset.call_count == 2
        assert all(call.args == (execution._forward_function.__code__,) for call in reset.call_args_list)
    assert tuple(decoder.state_dict()) == keys


def test_future_compiler_caches_are_isolated_when_copying_models():
    execution = FutureHeadExecution()
    execution.compiled = lambda *args: None
    copied = copy.deepcopy(execution)
    assert copied.compiled is None
    assert copied._parameter_layout is None
    assert copied._input_signature is None
    assert copied._forward_function.__code__ is not execution._forward_function.__code__


@pytest.mark.parametrize("transition", ["smaller_batch", "stride", "requires_grad", "autocast"])
def test_future_compiler_input_transitions_preserve_rng_and_restore_cache_settings(transition):
    decoder = model().mot.anticip.head_fm
    execution = FutureHeadExecution()
    noisy = torch.randn(4, 8, 192, requires_grad=True)
    timestep = torch.ones(4)
    condition = torch.randn(4, 8, 24, requires_grad=True)
    original = (noisy, timestep, condition)
    changed = original
    if transition == "smaller_batch":
        changed = tuple(value[:1] for value in original)
    elif transition == "stride":
        changed = (noisy.transpose(1, 2).contiguous().transpose(1, 2), timestep, condition)
    elif transition == "requires_grad":
        changed = tuple(value.detach() for value in original)
    original_cache_config = (aot_config.enable_autograd_cache, aot_config.enable_remote_autograd_cache)
    rng = torch.get_rng_state()

    def fake_compile(function, **kwargs):
        assert not aot_config.enable_autograd_cache
        assert not aot_config.enable_remote_autograd_cache
        torch.rand(3)
        first = True
        def compiled(*args):
            nonlocal first
            if first:
                assert not aot_config.enable_autograd_cache
                assert not aot_config.enable_remote_autograd_cache
                torch.rand(5)
                first = False
            return function(*args)
        return compiled

    with patch("torch.compile", side_effect=fake_compile) as compiler:
        # Unchanged hot calls must reuse compilation. The last batch and the
        # next epoch's original batch must both enter the protected cold path.
        for index, values in enumerate((original, original, changed, changed, original)):
            autocast = transition == "autocast" and index in (2, 3)
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
                expected = decoder(*values)
                actual = execution.run(decoder, *values, enabled=True, training=True)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
            assert (aot_config.enable_autograd_cache, aot_config.enable_remote_autograd_cache) == original_cache_config
        assert compiler.call_count == 3


@pytest.mark.parametrize("device,enabled,expected", [
    ("cpu", False, False), ("cpu", True, False), ("cuda:0", False, False), ("cuda:0", True, True),
])
def test_fused_optimizer_selection_preserves_settings(device, enabled, expected):
    parameters = [nn.Parameter(torch.ones(2))]
    with patch("torch.optim.AdamW") as adam:
        make_training_optimizer(parameters, learning_rate=1e-4, weight_decay=0.01, device=device, fused_adamw=enabled)
    assert adam.call_args.args == (parameters,)
    kwargs = {"lr": 1e-4, "weight_decay": 0.01, "betas": (0.9, 0.95)}
    if expected:
        kwargs["fused"] = True
    assert adam.call_args.kwargs == kwargs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_future_compiler_survives_flat_parameter_relocation_and_batch_changes(dtype):
    """Exercise real Inductor kernels before and after ZeRO-style relocation."""
    from rift.models.wan22.helpers.gradient import gradient_checkpoint_forward
    from rift.models.wan22.rift_model import AnticipFMHead

    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("This regression requires CUDA BF16 support")
    device = torch.device("cuda", torch.cuda.current_device())
    with torch.random.fork_rng(devices=[device]):
        torch.random.default_generator.manual_seed(2718)
        # Keep the native 512-wide decoder (64 dimensions per attention head).
        # Reducing token count and depth keeps this regression inexpensive.
        reference = AnticipFMHead(24, 192, width=512, blocks=1).to(
            device=device, dtype=dtype,
        )
        candidate = copy.deepcopy(reference)
        execution = FutureHeadExecution()
        assert all(parameter.data_ptr() % 16 == 0 for parameter in candidate.parameters())
        generator = torch.Generator().manual_seed(3141)
        noisy = torch.randn(4, 8, 192, generator=generator).to(device, dtype)
        condition = torch.randn(4, 8, 24, generator=generator).to(device, dtype)
        target = torch.randn(4, 8, 192, generator=generator).to(device, dtype)
        timestep = torch.tensor([81, 327, 613, 901], device=device, dtype=dtype)
        cache_settings = (aot_config.enable_autograd_cache, aot_config.enable_remote_autograd_cache)

        for phase, batch_size in enumerate((4, 4, 1, 4)):
            reference.zero_grad(set_to_none=True)
            candidate.zero_grad(set_to_none=True)
            if phase == 1:
                parameters = list(candidate.parameters())
                flat = torch.empty(
                    7 + sum(parameter.numel() for parameter in parameters),
                    device=device, dtype=dtype,
                )
                offset = 7
                with torch.no_grad():
                    for parameter in parameters:
                        view = flat[offset:offset + parameter.numel()].view_as(parameter)
                        view.copy_(parameter)
                        parameter.data = view
                        offset += parameter.numel()
                assert any(parameter.data_ptr() % 16 != 0 for parameter in parameters)

            results = []
            rng = (torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone())
            for decoder, compiled in ((reference, False), (candidate, True)):
                x = noisy[:batch_size].detach().clone().requires_grad_()
                c = condition[:batch_size].detach().clone().requires_grad_()
                t = timestep[:batch_size]
                function = (
                    lambda x, t, c: execution.run(candidate, x, t, c, enabled=True, training=True)
                ) if compiled else decoder
                prediction = gradient_checkpoint_forward(function, True, x, t, c)
                loss = (prediction.float() - target[:batch_size].float()).square().mean()
                loss.backward()
                gradients = {name: parameter.grad for name, parameter in decoder.named_parameters()}
                gradients.update(input_noisy=x.grad, input_condition=c.grad)
                assert all(value is not None and torch.isfinite(value).all() for value in gradients.values())
                results.append((prediction.detach(), loss.detach(), gradients))
                torch.testing.assert_close(torch.get_rng_state(), rng[0], rtol=0, atol=0)
                torch.testing.assert_close(torch.cuda.get_rng_state(device), rng[1], rtol=0, atol=0)
                assert (aot_config.enable_autograd_cache, aot_config.enable_remote_autograd_cache) == cache_settings

            expected, actual = results
            if dtype == torch.float32:
                torch.testing.assert_close(actual[0], expected[0], rtol=2e-4, atol=2e-5)
            else:
                # Fusion changes intermediate BF16 rounding. Use the full-step
                # benchmark's normwise bound, also for outputs near zero;
                # the FP32 case separately checks each output and gradient.
                output_error = ((actual[0].float() - expected[0].float()).norm()
                                / expected[0].float().norm().clamp_min(1e-12))
                assert float(output_error) <= .02, f"phase {phase} output relative L2: {output_error}"
            torch.testing.assert_close(actual[1], expected[1], rtol=.003, atol=.0002, msg=f"phase {phase} loss")
            if dtype == torch.float32:
                for name, gradient in expected[2].items():
                    torch.testing.assert_close(
                        actual[2][name], gradient, rtol=3e-4, atol=3e-5,
                        msg=f"phase {phase} gradient {name}",
                    )
            expected_flat = torch.cat([gradient.float().reshape(-1) for gradient in expected[2].values()])
            actual_flat = torch.cat([gradient.float().reshape(-1) for gradient in actual[2].values()])
            relative_error = (actual_flat - expected_flat).norm() / expected_flat.norm().clamp_min(1e-12)
            assert float(relative_error) <= .02, f"phase {phase} gradient relative L2: {relative_error}"
            assert execution.compiled is not None
