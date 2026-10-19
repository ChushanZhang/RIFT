"""Opt-in execution changes that preserve native RIFT parameters and losses."""
from __future__ import annotations

from types import FunctionType

import torch
from torch import nn
from torch._functorch import config as aot_config
from torch.nn.modules import module as module_hooks

from ..wan_video_dit import Head, WanVideoDiT


_VIDEO_POST_DIT = WanVideoDiT.post_dit
_VIDEO_UNPATCHIFY = WanVideoDiT.unpatchify
_HEAD_FORWARDS = (
    (Head, Head.forward),
    (nn.LayerNorm, nn.LayerNorm.forward),
    (nn.Linear, nn.Linear.forward),
)


def stock_video_head(expert):
    """Reject custom implementations and hooks whose observations would change."""
    if any(getattr(module_hooks, name) for name in (
        "_global_forward_pre_hooks", "_global_forward_hooks",
        "_global_backward_pre_hooks", "_global_backward_hooks",
    )):
        return False
    if (type(expert) is not WanVideoDiT
            or getattr(expert.post_dit, "__func__", None) is not _VIDEO_POST_DIT
            or getattr(expert.unpatchify, "__func__", None) is not _VIDEO_UNPATCHIFY
            or type(expert.head) is not Head):
        return False
    for module, (expected_type, expected_forward) in zip(
        (expert.head, expert.head.norm, expert.head.head), _HEAD_FORWARDS,
    ):
        if (type(module) is not expected_type
                or getattr(module.forward, "__func__", None) is not expected_forward
                or module._forward_pre_hooks or module._forward_hooks
                or module._backward_pre_hooks or module._backward_hooks
                or getattr(module, "_compiled_call_impl", None) is not None):
            return False
    return (tuple(expert.head.norm.normalized_shape) == (expert.hidden_dim,)
            and tuple(expert.head.patch_size) == tuple(expert.patch_size))


def video_training_prediction(expert, tokens, pre_state, *, enabled, training):
    """Omit the unused first frame only for the native tokenwise training head."""
    frames, height, width = pre_state["meta"]["grid_size"]
    time = pre_state["t"]
    if (not enabled or not training or not torch.is_grad_enabled()
            or expert.patch_size[0] != 1 or frames < 2
            or tokens.shape[1] != frames * height * width
            or time.ndim != 3 or time.shape[1] not in (1, tokens.shape[1])
            or not stock_video_head(expert)):
        return expert.post_dit(tokens, pre_state)[:, :, 1:]
    current = height * width
    suffix_time = time if time.shape[1] == 1 else time[:, current:]
    patches = expert.head(tokens[:, current:], suffix_time)
    return expert.unpatchify(patches, (frames - 1, height, width))


def future_head_forward(decoder, noisy, timestep, condition):
    return decoder(noisy, timestep, condition)


class FutureHeadExecution:
    """Compile the deterministic native FM head without registering a module.

    ZeRO initialization relocates Parameters into a flat allocation. Torch 2.7
    does not guard static input alignment, so a pre-ZeRO kernel may assume
    alignment that those new views no longer have. Metadata-only checks detect
    that transition without synchronizing CUDA or copying model weights. Input
    specialization changes, including a smaller final batch, follow the same
    protected compilation path rather than recompiling inside a hot call.
    """

    def __init__(self):
        self.compiled = None
        self._parameter_layout = None
        self._input_signature = None
        # Dynamo caches by code object. Isolate live models so one model's
        # aligned weights cannot supply a cached kernel for another's views.
        self._forward_function = FunctionType(
            future_head_forward.__code__.replace(), future_head_forward.__globals__,
            future_head_forward.__name__,
        )

    def __deepcopy__(self, memo):
        # Compiler state belongs to one model's storage layout, not its weights.
        copied = type(self)()
        memo[id(self)] = copied
        return copied

    @staticmethod
    def _layout(decoder):
        return tuple((parameter.device, parameter.dtype, parameter.data_ptr() % 16)
                     for parameter in decoder.parameters())

    @staticmethod
    def _inputs(noisy, timestep, condition):
        tensors = tuple(
            (tuple(tensor.shape), tuple(tensor.stride()), tensor.device,
             tensor.dtype, tensor.requires_grad)
            for tensor in (noisy, timestep, condition)
        )
        autocast = tuple(
            (device, torch.is_autocast_enabled(device), torch.get_autocast_dtype(device))
            for device in ("cpu", "cuda")
        )
        return tensors, autocast, torch.is_autocast_cache_enabled()

    def run(self, decoder, noisy, timestep, condition, *, enabled, training):
        if not enabled or not training or not torch.is_grad_enabled():
            return future_head_forward(decoder, noisy, timestep, condition)
        layout = self._layout(decoder)
        inputs = self._inputs(noisy, timestep, condition)
        if (self.compiled is None or layout != self._parameter_layout
                or inputs != self._input_signature):
            # The native head contains no stochastic operations. Preserve the
            # training RNG stream if compiler initialization consumes randomness.
            devices = [noisy.device] if noisy.device.type == "cuda" else []
            # Torch 2.7's AOTAutograd cache removes storage offsets before
            # Inductor can mark static Parameters. It can otherwise reload the
            # old aligned kernel even after Dynamo invalidation. Bypass that
            # cache only while building this graph; Inductor's cache retains
            # static-input offsets and remains enabled.
            with torch.random.fork_rng(devices=devices), aot_config.patch(
                enable_autograd_cache=False, enable_remote_autograd_cache=False,
            ):
                torch._dynamo.reset_code(self._forward_function.__code__)
                self.compiled = torch.compile(
                    self._forward_function, fullgraph=True, dynamic=False,
                )
                self._parameter_layout = layout
                self._input_signature = inputs
                return self.compiled(decoder, noisy, timestep, condition)
        return self.compiled(decoder, noisy, timestep, condition)
