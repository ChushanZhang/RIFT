"""Canonical RIFT flow-matching model factory.

This module keeps canonical model construction separate from the inherited
training runtime.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from omegaconf import DictConfig, OmegaConf

from .models.wan22.rift_model import RIFTModel


def _as_dict(value: Any, name: str, *, required: bool = False) -> dict:
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if value is None:
        if required:
            raise ValueError(f"`{name}` is required for RIFT.")
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"`{name}` must resolve to a dict, got {type(value)}.")
    return value


def create_rift(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: Optional[int] = None,
    action_dit_config=None,
    action_dit_pretrained_path: Optional[str] = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    anticip_future_tokens: int = 196,
    anticip_latent_t: int = 3,
    anticip_lambda: float = 1.0,
    anticip_action_lambda: float = 1.0,
    anticip_delta_mix: float = 0.8,
    anticip_lambda_final: Optional[float] = 0.2,
    anticip_anneal_start_frac: float = 0.7,
    anticip_anneal_total_steps: int = 21700,
    anticip_cond_noise_p: float = 0.3,
    anticip_cond_noise_sigma: float = 0.06,
    anticip_cond_noise_ramp_start_frac: Optional[float] = 0.7,
    anticip_cond_noise_action_free: bool = True,
    anticip_fm_lambda: float = 1.0,
    anticip_fm_width: int = 512,
    anticip_fm_blocks: int = 2,
    anticip_fm_lambda_final: Optional[float] = 0.2,
) -> RIFTModel:
    """Create canonical full-grid RIFT with its flow-matching objective."""
    video_dit_config = _as_dict(video_dit_config, "video_dit_config", required=True)
    action_dit_config = _as_dict(action_dit_config, "action_dit_config")
    video_scheduler = _as_dict(video_scheduler, "video_scheduler")
    action_scheduler = _as_dict(action_scheduler, "action_scheduler", required=True)
    loss = _as_dict(loss, "loss")

    if bool(video_dit_config.get("action_conditioned", False)):
        raise ValueError("RIFT requires `video_dit_config.action_conditioned=false`.")
    scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing = scheduler_keys - action_scheduler.keys()
    if missing:
        raise ValueError(
            f"`action_scheduler` is missing required keys: {sorted(missing)}."
        )

    model = RIFTModel.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=None if proprio_dim is None else int(proprio_dim),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
    )
    model.build_anticip(
        future_tokens=int(anticip_future_tokens),
        latent_t=int(anticip_latent_t),
        lambda_anticip=float(anticip_lambda),
        lambda_action_anticip=float(anticip_action_lambda),
        delta_mix=float(anticip_delta_mix),
        lambda_anticip_final=(
            None if anticip_lambda_final is None else float(anticip_lambda_final)
        ),
        anneal_start_frac=float(anticip_anneal_start_frac),
        anneal_total_steps=int(anticip_anneal_total_steps),
        cond_noise_p=float(anticip_cond_noise_p),
        cond_noise_sigma=float(anticip_cond_noise_sigma),
        cond_noise_ramp_start_frac=(
            None
            if anticip_cond_noise_ramp_start_frac is None
            else float(anticip_cond_noise_ramp_start_frac)
        ),
        cond_noise_action_free=bool(anticip_cond_noise_action_free),
        fm_lambda=float(anticip_fm_lambda),
        fm_width=int(anticip_fm_width),
        fm_blocks=int(anticip_fm_blocks),
        fm_lambda_final=(
            None if anticip_fm_lambda_final is None else float(anticip_fm_lambda_final)
        ),
    )
    return model


__all__ = ["create_rift"]
