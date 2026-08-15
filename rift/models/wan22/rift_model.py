"""Canonical RIFT flow-matching training with full-grid anticipation.

Pass A keeps FastWAM's native unconditional video co-training. Pass B replaces
future video rows with learned anticipation tokens and trains the action expert
to read them. The future-token representation is shaped by conditional flow
matching; a detached RMS-normalized L2 probe is retained in the checkpoint.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from rift.utils.logging_config import get_logger

from .fastwam import FastWAM
from .helpers.gradient import gradient_checkpoint_forward
from .wan_video_dit import RMSNorm, flash_attention

logger = get_logger(__name__)


class ProbeRMSLinear(nn.Linear):
    """Detached, non-learned RMS normalization followed by a trainable probe."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        detached = x.detach().float()
        normalized = detached * torch.rsqrt(
            detached.pow(2).mean(dim=-1, keepdim=True) + 1e-6
        )
        return super().forward(normalized.to(dtype=x.dtype))


class AnticipModule(nn.Module):
    """Full-grid learned tokens and the checkpoint-compatible L2 probe."""

    def __init__(self, n_tokens: int, hidden_dim: int, patch_dim: int):
        super().__init__()
        self.basis = nn.Parameter(torch.randn(n_tokens, hidden_dim) * 0.02)
        self.head = ProbeRMSLinear(hidden_dim, patch_dim)


class AnticipFMHead(nn.Module):
    """Conditional flow-matching decoder used to shape anticipation tokens."""

    def __init__(self, hidden_dim: int, patch_dim: int, width: int = 512, blocks: int = 2):
        super().__init__()
        if width % 8 != 0:
            raise ValueError(f"`fm_width` must be divisible by 8, got {width}.")
        self.in_x = nn.Linear(patch_dim, width)
        self.in_c = nn.Linear(hidden_dim, width)
        self.t_mlp = nn.Sequential(
            nn.Linear(128, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.n_heads = 8
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "ln1": nn.LayerNorm(width),
                        "q": nn.Linear(width, width),
                        "k": nn.Linear(width, width),
                        "v": nn.Linear(width, width),
                        "o": nn.Linear(width, width),
                        "norm_q": RMSNorm(width),
                        "norm_k": RMSNorm(width),
                        "ln2": nn.LayerNorm(width),
                        "mlp": nn.Sequential(
                            nn.Linear(width, width * 4),
                            nn.SiLU(),
                            nn.Linear(width * 4, width),
                        ),
                    }
                )
                for _ in range(blocks)
            ]
        )
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, patch_dim))

    @staticmethod
    def _timestep_embedding(timestep: torch.Tensor, dim: int = 128) -> torch.Tensor:
        half = dim // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=timestep.device, dtype=torch.float32)
            / half
        )
        angles = (
            timestep.float().view(-1, 1) / 1000.0
        ) * frequencies.view(1, -1) * 1000.0
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    def forward(
        self,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        hidden = (
            self.in_x(x_t)
            + self.in_c(condition)
            + self.t_mlp(self._timestep_embedding(timestep).to(x_t.dtype)).unsqueeze(1)
        )
        for block in self.blocks:
            normed = block["ln1"](hidden)
            query = block["norm_q"](block["q"](normed))
            key = block["norm_k"](block["k"](normed))
            value = block["v"](normed)
            attended = flash_attention(
                q=query,
                k=key,
                v=value,
                num_heads=self.n_heads,
            )
            hidden = hidden + block["o"](attended)
            hidden = hidden + block["mlp"](block["ln2"](hidden))
        return self.out(hidden)


class RIFTModel(FastWAM):
    """FastWAM with the canonical full-grid RIFT flow-matching objective."""

    def build_anticip(
        self,
        future_tokens: int,
        latent_t: int = 3,
        lambda_anticip: float = 1.0,
        lambda_action_anticip: float = 1.0,
        delta_mix: float = 0.8,
        lambda_anticip_final: Optional[float] = 0.2,
        anneal_start_frac: float = 0.7,
        anneal_total_steps: int = 21700,
        cond_noise_p: float = 0.3,
        cond_noise_sigma: float = 0.06,
        cond_noise_ramp_start_frac: Optional[float] = 0.7,
        cond_noise_action_free: bool = True,
        fm_lambda: float = 1.0,
        fm_width: int = 512,
        fm_blocks: int = 2,
        fm_lambda_final: Optional[float] = 0.2,
    ) -> None:
        """Attach the RIFT FM head, diagnostic probe, and training schedule."""
        if latent_t < 2:
            raise ValueError(f"`anticip_latent_t` must be at least 2, got {latent_t}.")
        if future_tokens <= 0:
            raise ValueError(f"`anticip_future_tokens` must be positive, got {future_tokens}.")
        if not 0.0 <= delta_mix <= 1.0:
            raise ValueError(f"`anticip_delta_mix` must be in [0, 1], got {delta_mix}.")
        if not 0.0 <= cond_noise_p <= 1.0:
            raise ValueError(f"`anticip_cond_noise_p` must be in [0, 1], got {cond_noise_p}.")
        if cond_noise_sigma < 0.0:
            raise ValueError(f"`anticip_cond_noise_sigma` must be non-negative, got {cond_noise_sigma}.")
        if not 0.0 <= anneal_start_frac <= 1.0:
            raise ValueError(
                f"`anticip_anneal_start_frac` must be in [0, 1], got {anneal_start_frac}."
            )
        if cond_noise_ramp_start_frac is not None and not (
            0.0 <= cond_noise_ramp_start_frac <= 1.0
        ):
            raise ValueError(
                "`anticip_cond_noise_ramp_start_frac` must be in [0, 1] or null, "
                f"got {cond_noise_ramp_start_frac}."
            )
        if anneal_total_steps <= 0:
            raise ValueError(
                f"`anticip_anneal_total_steps` must be positive, got {anneal_total_steps}."
            )
        if min(lambda_anticip, lambda_action_anticip, fm_lambda) < 0.0:
            raise ValueError("RIFT loss weights must be non-negative.")
        if fm_lambda <= 0.0:
            raise ValueError("RIFT requires `anticip_fm_lambda > 0`.")
        if lambda_anticip_final is not None and lambda_anticip_final < 0.0:
            raise ValueError("`anticip_lambda_final` must be non-negative or null.")
        if fm_lambda_final is not None and fm_lambda_final < 0.0:
            raise ValueError("`anticip_fm_lambda_final` must be non-negative or null.")

        video_expert = self.video_expert
        if getattr(video_expert, "video_attention_mask_mode", None) != "first_frame_causal":
            raise ValueError(
                "RIFT requires `video_attention_mask_mode='first_frame_causal'`."
            )
        if bool(getattr(video_expert, "action_conditioned", False)):
            raise ValueError("RIFT requires `video_dit_config.action_conditioned=false`.")
        for flag in ("seperated_timestep", "fuse_vae_embedding_in_latents"):
            if not bool(getattr(video_expert, flag, False)):
                raise ValueError(f"RIFT requires `video_dit_config.{flag}=true`.")

        hidden_dim = int(getattr(video_expert, "hidden_dim", 3072))
        patch_dim = int(math.prod(video_expert.patch_size)) * int(
            getattr(video_expert, "out_dim", 48)
        )
        video_head = getattr(getattr(video_expert, "head", None), "head", None)
        if (
            video_head is not None
            and hasattr(video_head, "out_features")
            and int(video_head.out_features) != patch_dim
        ):
            raise ValueError(
                f"RIFT patch width {patch_dim} does not match video head width "
                f"{int(video_head.out_features)}."
            )

        anticipation = AnticipModule(future_tokens, hidden_dim, patch_dim).to(
            device=self.device,
            dtype=self.torch_dtype,
        )
        anticipation.head_fm = AnticipFMHead(
            hidden_dim,
            patch_dim,
            width=fm_width,
            blocks=fm_blocks,
        ).to(device=self.device, dtype=self.torch_dtype)
        self.mot.anticip = anticipation

        self.anticip_future_tokens = int(future_tokens)
        self.anticip_latent_t = int(latent_t)
        self.lambda_anticip = float(lambda_anticip)
        self.lambda_action_anticip = float(lambda_action_anticip)
        self.anticip_delta_mix = float(delta_mix)
        self.lambda_anticip_final = (
            None if lambda_anticip_final is None else float(lambda_anticip_final)
        )
        self.anticip_anneal_start_frac = float(anneal_start_frac)
        self.anticip_anneal_total_steps = int(anneal_total_steps)
        self.anticip_cond_noise_p = float(cond_noise_p)
        self.anticip_cond_noise_sigma = float(cond_noise_sigma)
        self.anticip_cond_noise_ramp_start_frac = (
            None
            if cond_noise_ramp_start_frac is None
            else float(cond_noise_ramp_start_frac)
        )
        self.anticip_cond_noise_action_free = bool(cond_noise_action_free)
        self.anticip_fm_lambda = float(fm_lambda)
        self.anticip_fm_lambda_final = (
            None if fm_lambda_final is None else float(fm_lambda_final)
        )
        self._anticip_step = 0

        params = sum(parameter.numel() for parameter in anticipation.parameters())
        logger.info(
            "RIFT: full-grid=%d, latent_t=%d, params=%.2fM, "
            "delta_mix=%.2f, fm_lambda=%.2f",
            future_tokens,
            latent_t,
            params / 1e6,
            delta_mix,
            fm_lambda,
        )

    @torch.no_grad()
    def _build_anticip_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total = video_seq_len + action_seq_len
        mask = torch.zeros((total, total), dtype=torch.bool, device=device)
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        mask[video_seq_len:, :video_seq_len] = True
        mask[video_seq_len:, video_seq_len:] = True
        return mask

    def _anneal_lambda(self, initial: float, final: Optional[float]) -> float:
        if final is None:
            return float(initial)
        knee = int(self.anticip_anneal_start_frac * self.anticip_anneal_total_steps)
        if self._anticip_step < knee:
            return float(initial)
        progress = min(
            1.0,
            (self._anticip_step - knee)
            / max(1, self.anticip_anneal_total_steps - knee),
        )
        return float(final) + 0.5 * (float(initial) - float(final)) * (
            1.0 + math.cos(math.pi * progress)
        )

    def _anticip_video_pre(
        self,
        *,
        first_frame_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> tuple[dict, torch.Tensor]:
        if not hasattr(self.mot, "anticip"):
            raise RuntimeError("`build_anticip()` must run before RIFT training or inference.")
        batch_size = first_frame_latents.shape[0]
        latents = first_frame_latents.new_zeros(
            (
                batch_size,
                first_frame_latents.shape[1],
                self.anticip_latent_t,
                first_frame_latents.shape[3],
                first_frame_latents.shape[4],
            )
        )
        latents[:, :, :1] = first_frame_latents
        timestep = torch.zeros(
            (batch_size,),
            dtype=latents.dtype,
            device=self.device,
        )
        pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=True,
        )
        tokens_per_frame = int(pre["meta"]["tokens_per_frame"])
        future_slots = int(pre["tokens"].shape[1]) - tokens_per_frame
        if future_slots != self.anticip_future_tokens:
            raise ValueError(
                f"Configured anticipation grid has {self.anticip_future_tokens} tokens, "
                f"but the encoded future grid has {future_slots}."
            )
        anticipation = self.mot.anticip.basis.to(dtype=pre["tokens"].dtype)
        anticipation = anticipation.unsqueeze(0).expand(batch_size, -1, -1)
        tokens = torch.cat(
            [pre["tokens"][:, :tokens_per_frame], anticipation],
            dim=1,
        )
        return pre, tokens

    def _action_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        timestep: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
        keep_rows: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        token_loss = F.mse_loss(
            prediction.float(), target.float(), reduction="none"
        ).mean(dim=2)
        if action_is_pad is None:
            per_sample = token_loss.mean(dim=1)
        else:
            valid = (~action_is_pad).to(device=token_loss.device, dtype=token_loss.dtype)
            per_sample = (token_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        weight = self.train_action_scheduler.training_weight(timestep).to(
            device=per_sample.device,
            dtype=per_sample.dtype,
        )
        if keep_rows is None:
            return (per_sample * weight).mean()
        keep = keep_rows.to(device=per_sample.device, dtype=per_sample.dtype)
        return (per_sample * weight * keep).sum() / keep.sum().clamp(min=1.0)

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]
        first_frame = inputs["first_frame_latents"]
        if first_frame is None:
            raise ValueError("RIFT requires `fuse_vae_embedding_in_latents=true`.")
        if input_latents.shape[2] < self.anticip_latent_t:
            raise ValueError(
                f"Training has {input_latents.shape[2]} latent frames, but RIFT requires "
                f"at least {self.anticip_latent_t}."
            )

        # Pass A: native FastWAM unconditional video co-training.
        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        noisy_video = self.train_video_scheduler.add_noise(
            input_latents, noise_video, timestep_video
        )
        target_video = self.train_video_scheduler.training_target(
            input_latents, noise_video, timestep_video
        )
        noisy_video[:, :, :1] = first_frame

        noise_action_a = torch.randn_like(action)
        timestep_action_a = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action_a = self.train_action_scheduler.add_noise(
            action, noise_action_a, timestep_action_a
        )
        video_pre_a = self.video_expert.pre_dit(
            x=noisy_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=True,
        )
        action_pre_a = self.action_expert.pre_dit(
            action_tokens=noisy_action_a,
            timestep=timestep_action_a,
            context=context,
            context_mask=context_mask,
        )
        tokens_per_frame = int(video_pre_a["meta"]["tokens_per_frame"])
        mask_a = self._build_mot_attention_mask(
            video_seq_len=video_pre_a["tokens"].shape[1],
            action_seq_len=action_pre_a["tokens"].shape[1],
            video_tokens_per_frame=tokens_per_frame,
            device=video_pre_a["tokens"].device,
        )
        output_a = self.mot(
            embeds_all={
                "video": video_pre_a["tokens"],
                "action": action_pre_a["tokens"],
            },
            attention_mask=mask_a,
            freqs_all={"video": video_pre_a["freqs"], "action": action_pre_a["freqs"]},
            context_all={
                "video": {
                    "context": video_pre_a["context"],
                    "mask": video_pre_a["context_mask"],
                },
                "action": {
                    "context": action_pre_a["context"],
                    "mask": action_pre_a["context_mask"],
                },
            },
            t_mod_all={"video": video_pre_a["t_mod"], "action": action_pre_a["t_mod"]},
        )
        prediction_video = self.video_expert.post_dit(output_a["video"], video_pre_a)[:, :, 1:]
        target_video = target_video[:, :, 1:]
        video_per_sample = self._compute_video_loss_per_sample(
            pred_video=prediction_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=False,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            device=video_per_sample.device,
            dtype=video_per_sample.dtype,
        )
        loss_video = (video_per_sample * video_weight).mean()

        # Pass B: learned future grid and always-read action training.
        first_frame_b = first_frame
        noise_probability = self.anticip_cond_noise_p
        noise_sigma = self.anticip_cond_noise_sigma
        ramp_start = self.anticip_cond_noise_ramp_start_frac
        if ramp_start is not None:
            ramp_step = ramp_start * self.anticip_anneal_total_steps
            progress = min(
                1.0,
                max(
                    0.0,
                    (self._anticip_step - ramp_step)
                    / max(1.0, self.anticip_anneal_total_steps - ramp_step),
                ),
            )
            noise_probability *= progress
            noise_sigma *= progress

        noisy_rows = None
        if noise_probability > 0.0 and self.anticip_cond_noise_action_free:
            noisy_rows = torch.rand(batch_size, device=first_frame.device) < noise_probability
            if bool(noisy_rows.any()):
                scale = first_frame.float().std().clamp(min=1e-3).to(first_frame.dtype)
                perturbation = noise_sigma * scale * torch.randn_like(first_frame)
                first_frame_b = torch.where(
                    noisy_rows.view(-1, 1, 1, 1, 1),
                    first_frame + perturbation,
                    first_frame,
                )
        elif noise_probability > 0.0 and float(torch.rand(())) < noise_probability:
            scale = first_frame.float().std().clamp(min=1e-3).to(first_frame.dtype)
            first_frame_b = first_frame + noise_sigma * scale * torch.randn_like(first_frame)

        video_pre_b, video_tokens_b = self._anticip_video_pre(
            first_frame_latents=first_frame_b,
            context=context,
            context_mask=context_mask,
        )
        target_future = input_latents[:, :, 1 : self.anticip_latent_t]
        patch_t, patch_h, patch_w = self.video_expert.patch_size
        target_patches = rearrange(
            target_future,
            "b c (f x) (h y) (w z) -> b (f h w) (x y z c)",
            x=int(patch_t),
            y=int(patch_h),
            z=int(patch_w),
        )

        noise_action_b = torch.randn_like(action)
        timestep_action_b = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action_b = self.train_action_scheduler.add_noise(
            action, noise_action_b, timestep_action_b
        )
        target_action_b = self.train_action_scheduler.training_target(
            action, noise_action_b, timestep_action_b
        )
        action_pre_b = self.action_expert.pre_dit(
            action_tokens=noisy_action_b,
            timestep=timestep_action_b,
            context=context,
            context_mask=context_mask,
        )
        mask_b = self._build_anticip_attention_mask(
            video_seq_len=video_tokens_b.shape[1],
            action_seq_len=action_pre_b["tokens"].shape[1],
            video_tokens_per_frame=tokens_per_frame,
            device=video_tokens_b.device,
        )
        if self.dit.training:
            self._anticip_step += 1
        output_b = self.mot(
            embeds_all={"video": video_tokens_b, "action": action_pre_b["tokens"]},
            attention_mask=mask_b,
            freqs_all={"video": video_pre_b["freqs"], "action": action_pre_b["freqs"]},
            context_all={
                "video": {
                    "context": video_pre_b["context"],
                    "mask": video_pre_b["context_mask"],
                },
                "action": {
                    "context": action_pre_b["context"],
                    "mask": action_pre_b["context_mask"],
                },
            },
            t_mod_all={"video": video_pre_b["t_mod"], "action": action_pre_b["t_mod"]},
        )
        future_states = output_b["video"][
            :, tokens_per_frame : tokens_per_frame + self.anticip_future_tokens
        ]
        prediction_action_b = self.action_expert.post_dit(
            output_b["action"], action_pre_b
        )
        keep_rows = None if noisy_rows is None else ~noisy_rows
        loss_action = self._action_loss(
            prediction_action_b,
            target_action_b,
            timestep_action_b,
            action_is_pad,
            keep_rows=keep_rows,
        )

        # The detached L2 probe remains trainable, but cannot shape future_states.
        predicted_patches = self.mot.anticip.head(future_states)
        grid_f, grid_h, grid_w = video_pre_b["meta"]["grid_size"]
        prediction_future = self.video_expert.unpatchify(
            predicted_patches,
            (int(grid_f) - 1, int(grid_h), int(grid_w)),
        )
        future_pad = None
        if image_is_pad is not None:
            temporal_factor = int(self.vae.temporal_downsample_factor)
            future_pad = image_is_pad[
                :, : 1 + (self.anticip_latent_t - 1) * temporal_factor
            ]

        timestep_fm = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=target_patches.dtype,
        )
        noise_fm = torch.randn_like(target_patches)
        noised_patches = self.train_video_scheduler.add_noise(
            target_patches, noise_fm, timestep_fm
        )
        target_fm = self.train_video_scheduler.training_target(
            target_patches, noise_fm, timestep_fm
        )
        prediction_fm = gradient_checkpoint_forward(
            self.mot.anticip.head_fm,
            True,
            noised_patches,
            timestep_fm,
            future_states,
        )
        fm_per_token = F.mse_loss(
            prediction_fm.float(), target_fm.float(), reduction="none"
        ).mean(dim=2)
        if future_pad is None:
            fm_per_sample = fm_per_token.mean(dim=1)
        else:
            temporal_factor = int(self.vae.temporal_downsample_factor)
            latent_pad = future_pad[:, 1:].reshape(
                future_pad.shape[0], -1, temporal_factor
            ).all(dim=2)
            valid = (~latent_pad).to(fm_per_token.dtype).repeat_interleave(
                fm_per_token.shape[1] // latent_pad.shape[1], dim=1
            )
            fm_per_sample = (fm_per_token * valid).sum(dim=1) / valid.sum(dim=1).clamp(
                min=1.0
            )
        fm_weight = self.train_video_scheduler.training_weight(timestep_fm).to(
            device=fm_per_sample.device,
            dtype=fm_per_sample.dtype,
        )
        loss_fm = (fm_per_sample * fm_weight).mean()

        loss_absolute = self._compute_video_loss_per_sample(
            pred_video=prediction_future,
            target_video=target_future,
            image_is_pad=future_pad,
            include_initial_video_step=False,
        ).mean()
        if self.anticip_delta_mix > 0.0:
            reference = first_frame.detach()
            target_delta = target_future - reference
            channel_scale = (
                target_delta.float()
                .std(dim=(0, 2, 3, 4), keepdim=True)
                .clamp(min=1e-3)
                .detach()
                .to(target_delta.dtype)
            )
            loss_motion = self._compute_video_loss_per_sample(
                pred_video=(prediction_future - reference) / channel_scale,
                target_video=target_delta / channel_scale,
                image_is_pad=future_pad,
                include_initial_video_step=False,
            ).mean()
            loss_anticip = (
                (1.0 - self.anticip_delta_mix) * loss_absolute
                + self.anticip_delta_mix * loss_motion
            )
        else:
            loss_anticip = loss_absolute

        lambda_anticip = self._anneal_lambda(
            self.lambda_anticip, self.lambda_anticip_final
        )
        lambda_fm = self._anneal_lambda(
            self.anticip_fm_lambda, self.anticip_fm_lambda_final
        )
        loss_total = (
            self.loss_lambda_video * loss_video
            + self.lambda_action_anticip * loss_action
            + lambda_anticip * loss_anticip
            + lambda_fm * loss_fm
        )
        values = torch.stack(
            [
                loss_video.detach(),
                loss_action.detach(),
                loss_anticip.detach(),
                loss_fm.detach(),
                future_states.detach().float().pow(2).mean().sqrt(),
            ]
        ).float().tolist()
        loss_dict = {
            "loss_video": self.loss_lambda_video * values[0],
            "loss_action_anticip": self.lambda_action_anticip * values[1],
            "loss_anticip": lambda_anticip * values[2],
            "loss_fm": lambda_fm * values[3],
            "lambda_anticip": lambda_anticip,
            "lambda_fm": lambda_fm,
            "future_token_rms": values[4],
        }
        return loss_total, loss_dict

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        num_video_frames: int = 9,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        """Predict actions from one cached [frame 0 + anticipation grid] prefill."""
        del negative_prompt, text_cfg_scale
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must be [1, 3, H, W] or [3, H, W], got {tuple(input_image.shape)}."
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"Image dimensions must be multiples of 16, got {height}x{width}.")
        latent_t = (
            (num_video_frames - 1) // int(self.vae.temporal_downsample_factor) + 1
        )
        if latent_t != self.anticip_latent_t:
            raise ValueError(
                f"Inference produces latent_t={latent_t}, but the model was trained with "
                f"latent_t={self.anticip_latent_t}."
            )

        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but this model has no proprio encoder.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 2 or proprio.shape != (1, self.proprio_dim):
                raise ValueError(
                    f"`proprio` must have shape [1, {self.proprio_dim}], got {tuple(proprio.shape)}."
                )
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_latents = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame = self._encode_input_image_latents_tensor(
            input_image=input_image, tiled=tiled
        )

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt == use_context:
            raise ValueError("Provide exactly one of `prompt` or `context/context_mask`.")
        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError("`context/context_mask` must be [B, L, D]/[B, L].")
            context = context.to(device=self.device, dtype=self.torch_dtype)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        video_pre, video_tokens = self._anticip_video_pre(
            first_frame_latents=first_frame,
            context=context,
            context_mask=context_mask,
        )
        video_seq_len = int(video_tokens.shape[1])
        tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
        attention_mask = self._build_anticip_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=action_latents.shape[1],
            video_tokens_per_frame=tokens_per_frame,
            device=video_tokens.device,
        )
        video_cache = self.mot.prefill_video_cache(
            video_tokens=video_tokens,
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )

        timesteps, deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=action_latents.dtype,
            shift_override=sigma_shift,
        )
        for step_t, step_delta in zip(timesteps, deltas):
            timestep = step_t.unsqueeze(0).to(
                dtype=action_latents.dtype,
                device=self.device,
            )
            prediction = self._predict_action_noise_with_cache(
                latents_action=action_latents,
                timestep_action=timestep,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            action_latents = self.infer_action_scheduler.step(
                prediction, step_delta, action_latents
            )
        return {
            "action": action_latents[0].detach().to(device="cpu", dtype=torch.float32)
        }
