"""Batched online encoding without changing the frozen VAE cache identity."""
from __future__ import annotations

import torch

from .static_tensor_cache import cached_tensor, exact_device_key


class OnlineVaeEncoder:
    """Keep graph callables and device scales outside the model state dict."""

    def __init__(self, vae, *, cudagraphs: bool, batch_size: int | None):
        self.vae = vae
        self.cudagraphs = cudagraphs
        self.batch_size = batch_size
        self._compiled_encode = None

    @torch.no_grad()
    def __call__(self, videos: torch.Tensor) -> torch.Tensor:
        if videos.ndim != 5 or videos.shape[0] == 0:
            raise ValueError("Online VAE encoding requires a nonempty [B, C, T, H, W] batch.")
        if self.cudagraphs and videos.device.type != "cuda":
            raise ValueError("`vae_encode_mode=cudagraphs` requires CUDA; use `batched` on CPU.")
        if any(module.training for module in self.vae.modules()):
            raise RuntimeError("Online VAE encoding requires eval mode on every VAE module.")
        if any(parameter.requires_grad for parameter in self.vae.parameters()):
            raise RuntimeError("Online VAE encoding requires frozen VAE parameters.")

        # CUDA Graphs cannot capture CPU scale transfers. Preserve the original
        # CPU constants and source file, which are part of existing cache IDs.
        scale = cached_tensor(
            self, "scale", exact_device_key(videos.device),
            lambda: torch.stack([value.to(device=videos.device) for value in self.vae.scale]),
            sources=self.vae.scale,
        )
        encode = self.vae.model.encode
        if self.cudagraphs:
            if self._compiled_encode is None:
                self._compiled_encode = torch.compile(
                    encode, backend="cudagraphs", fullgraph=True, dynamic=False,
                )
            encode = self._compiled_encode

        chunks = videos.split(self.batch_size or videos.shape[0], dim=0)
        latents = []
        for chunk in chunks:
            encoded = encode(chunk, scale)
            # Graph replay reuses its output storage; each chunk/microbatch must
            # own its latents until the downstream backward has finished.
            latents.append(encoded.clone() if self.cudagraphs else encoded)
        return latents[0] if len(latents) == 1 else torch.cat(latents, dim=0)
