"""Synthetic, native RIFT fixtures for paired training-step benchmarks.

No model or dataset assets are loaded. The video expert starts from its native
random initialization; the action backbone is interpolated from that expert
with the upstream preprocessing rule. This reproduces the initialization
structure, not the values or statistics of the pretrained Wan checkpoint.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rift.models.wan22.action_dit import ActionDiT  # noqa: E402
from rift.models.wan22.mot import MoT  # noqa: E402
from rift.models.wan22.rift_model import RIFTModel  # noqa: E402
from rift.models.wan22.wan_video_dit import WanVideoDiT  # noqa: E402
from scripts.preprocess_action_dit_backbone import _resize_tensor_to_shape  # noqa: E402


DEFAULTS = {
    "video_hidden": 1536,
    "action_hidden": 512,
    "video_ffn": 7168,
    "action_ffn": 2048,
    "layers": 9,
    "heads": 12,
    "head_dim": 128,
    "text_dim": 256,
    "context_length": 128,
    "seed": 2731,
    "batch_size": 4,
}
CACHE_NAMESPACE = "synthetic-native-rift-training-combo"


def _arg(args: Any, name: str) -> int:
    return int(getattr(args, name, DEFAULTS[name]))


class FrozenVAE(nn.Module):
    """Only the native padding calculation needs VAE metadata in this fixture."""

    temporal_downsample_factor = 4

    def encode(self, *args, **kwargs):
        raise AssertionError("The benchmark must use resident indexed latents.")


class ResidentLatentCache:
    """Supply synthetic resident latents through RIFT's indexed-input path."""

    namespace = CACHE_NAMESPACE

    def accept_precomputed(
        self, latents, keys, namespaces, video_shape, *, tiled=False
    ):
        batch_size = int(video_shape[0])
        if tuple(video_shape) != (batch_size, 3, 9, 224, 448):
            raise ValueError("The fixture requires 9 video frames at 224 by 448.")
        if tuple(latents.shape) != (batch_size, 48, 3, 14, 28):
            raise ValueError("The fixture requires native three-frame latent grids.")
        if len(keys) != batch_size or len(namespaces) != batch_size:
            raise ValueError("Indexed metadata must contain one entry per sample.")
        return latents


@torch.no_grad()
def _initialize_action_backbone(video: WanVideoDiT, action: ActionDiT) -> dict:
    """Apply exactly the upstream resize/alpha rule, retaining random I/O heads."""
    video_state = video.state_dict()
    action_state = action.state_dict()
    copied, resized = 0, 0
    for key in sorted(ActionDiT.backbone_key_set(action_state.keys())):
        source, target = video_state[key], action_state[key]
        if source.shape == target.shape:
            value = source
            copied += 1
        else:
            value = _resize_tensor_to_shape(source, tuple(target.shape))
            if source.ndim >= 2 and source.shape[-1] != target.shape[-1]:
                alpha = (float(source.shape[-1]) / float(target.shape[-1])) ** 0.5
                value = value.float() * alpha
            resized += 1
        target.copy_(value.to(dtype=target.dtype, device=target.device))
    return {"copied_backbone_tensors": copied, "resized_backbone_tensors": resized}


def make_model(args: Any, device, dtype: torch.dtype) -> RIFTModel:
    """Build a native RIFT model without consuming the caller's CPU RNG stream."""
    device = torch.device(device)
    common = {
        "text_dim": _arg(args, "text_dim"),
        "freq_dim": 256,
        "eps": 1e-6,
        "num_heads": _arg(args, "heads"),
        "attn_head_dim": _arg(args, "head_dim"),
        "num_layers": _arg(args, "layers"),
        "use_gradient_checkpointing": True,
    }
    # Initialize on CPU as native constructors do. Seed only that generator:
    # torch.manual_seed() would also alter the caller's CUDA RNG streams.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(_arg(args, "seed"))
        video = WanVideoDiT(
            hidden_dim=_arg(args, "video_hidden"),
            ffn_dim=_arg(args, "video_ffn"),
            in_dim=48,
            out_dim=48,
            patch_size=(1, 2, 2),
            has_image_input=False,
            seperated_timestep=True,
            fuse_vae_embedding_in_latents=True,
            action_conditioned=False,
            action_dim=7,
            action_group_causal_mask_mode="group_diagonal",
            video_attention_mask_mode="first_frame_causal",
            **common,
        )
        action = ActionDiT(
            hidden_dim=_arg(args, "action_hidden"),
            ffn_dim=_arg(args, "action_ffn"),
            action_dim=7,
            **common,
        )
        initialization = _initialize_action_backbone(video, action)
        model = RIFTModel(
            video_expert=video,
            action_expert=action,
            mot=MoT(
                mixtures={"video": video, "action": action},
                mot_checkpoint_mixed_attn=True,
            ),
            vae=FrozenVAE(),
            text_dim=_arg(args, "text_dim"),
            proprio_dim=8,
            device="cpu",
            torch_dtype=torch.float32,
            video_train_shift=5.0,
            video_infer_shift=5.0,
            video_num_train_timesteps=1000,
            action_train_shift=5.0,
            action_infer_shift=5.0,
            action_num_train_timesteps=1000,
            loss_lambda_video=1.0,
            loss_lambda_action=1.0,
            compile_training_denoise=False,
        )
        model.build_anticip(
            future_tokens=196,
            latent_t=3,
            lambda_anticip=1.0,
            lambda_action_anticip=1.0,
            delta_mix=0.8,
            lambda_anticip_final=0.2,
            anneal_start_frac=0.7,
            anneal_total_steps=21700,
            cond_noise_p=0.3,
            cond_noise_sigma=0.06,
            cond_noise_ramp_start_frac=0.7,
            cond_noise_action_free=True,
            fm_lambda=1.0,
            fm_width=512,
            fm_blocks=2,
            fm_lambda_final=0.2,
            fuse_training_paths=False,
            train_probe_head=True,
        )

    model.to(device=device, dtype=dtype)
    model.device, model.torch_dtype = device, dtype
    model._vae_latent_cache = ResidentLatentCache()
    model._anticip_step = 21700
    model._training_combo_initialization = initialization

    # Exact Wan22Trainer._apply_dit_only_train_mode behavior. The outer model
    # remains in eval mode while both MoT experts and anticipation heads train.
    model.eval()
    model.requires_grad_(False)
    model.dit.train()
    model.dit.requires_grad_(True)
    model.proprio_encoder.train()
    model.proprio_encoder.requires_grad_(True)
    return model


def make_batch(args: Any, device, dtype: torch.dtype) -> dict:
    """Create a resident batch using a private CPU generator and finite padding."""
    batch_size = _arg(args, "batch_size")
    context_length = _arg(args, "context_length")
    if batch_size < 1 or context_length < 1:
        raise ValueError("Batch size and context length must be positive.")
    generator = torch.Generator(device="cpu").manual_seed(_arg(args, "seed") + 1)

    def randn(*shape):
        return torch.randn(*shape, generator=generator).to(device=device, dtype=dtype)

    action_pad = torch.zeros(batch_size, 32, dtype=torch.bool)
    image_pad = torch.zeros(batch_size, 9, dtype=torch.bool)
    context_mask = torch.ones(batch_size, context_length, dtype=torch.bool)
    if batch_size > 1:
        action_pad[1, 16:] = True
        image_pad[1, 5:] = True
        context_mask[1, max(context_length // 2, 1):] = False
    if batch_size > 2:
        action_pad[2, 1:] = True
        image_pad[2, 2:] = True
        context_mask[2, 1:] = False
    if batch_size > 3:
        action_pad[3] = True
        image_pad[3, 1:] = True
        # Native RIFT appends a valid proprio token, including for this row.
        context_mask[3] = False

    return {
        "video_latents": randn(batch_size, 48, 3, 14, 28),
        "video_shape": torch.tensor([[3, 9, 224, 448]] * batch_size),
        "vae_cache_keys": [f"synthetic-row-{row}" for row in range(batch_size)],
        "vae_cache_namespace": [CACHE_NAMESPACE] * batch_size,
        "action": randn(batch_size, 32, 7),
        "proprio": randn(batch_size, 32, 8),
        "context": randn(batch_size, context_length, _arg(args, "text_dim")),
        "context_mask": context_mask.to(device=device),
        "action_is_pad": action_pad.to(device=device),
        "image_is_pad": image_pad.to(device=device),
    }


def parameter_count(model: nn.Module, *, trainable_only: bool = True) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad or not trainable_only
    )


def fixture_metadata(args: Any, model: RIFTModel | None = None) -> dict:
    metadata = {
        "synthetic": True,
        "pretrained_weights_loaded": False,
        "initialization": "native_random_video_interpolated_action_backbone_with_alpha",
        "configuration": {name: _arg(args, name) for name in DEFAULTS},
        "video_latent_shape": [_arg(args, "batch_size"), 48, 3, 14, 28],
        "action_shape": [_arg(args, "batch_size"), 32, 7],
        "future_tokens": 196,
        "fm_width": 512,
        "fm_blocks": 2,
        "gradient_checkpointing": True,
        "probe_head_enabled": True,
        "curriculum_step": 21700,
        "vae_encoding_included": False,
        "data_loading_included": False,
    }
    if model is not None:
        metadata["trainable_parameters"] = parameter_count(model)
        metadata["component_parameters"] = {
            name: parameter_count(module)
            for name, module in (
                ("video", model.video_expert),
                ("action", model.action_expert),
                ("anticipation_including_fm", model.mot.anticip),
                ("proprio", model.proprio_encoder),
            )
        }
        metadata["initialization_counts"] = dict(model._training_combo_initialization)
    return metadata
