"""Bind packed episode latents to a frozen VAE without a per-window disk cache."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import default_collate


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _tensor_bytes(tensor):
    value = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return memoryview(value.numpy())


def tensor_digest(tensor):
    digest = hashlib.sha256(_json_bytes([str(tensor.dtype), list(tensor.shape)]))
    digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def identity_hash(identity):
    return hashlib.sha256(_json_bytes(identity)).hexdigest()


def shared_cache_identity(identity):
    """Frozen latent contract shared across encoder packages and runtimes."""
    return {field: identity[field] for field in
            ("format", "weights_sha256", "scale", "input_dtype")}


def precomputed_cache_identity(root, namespace, actual_identity):
    """Read an existing namespace without requiring the original encoder runtime."""
    if (not isinstance(namespace, str) or len(namespace) != 64
            or any(char not in "0123456789abcdef" for char in namespace)):
        raise ValueError("Invalid precomputed VAE cache namespace.")
    stored = json.loads((Path(root) / namespace / "identity.json").read_text())
    if identity_hash(stored) != namespace:
        raise RuntimeError("Precomputed VAE cache namespace identity mismatch.")
    for field in ("format", "weights_sha256", "scale", "input_dtype"):
        if stored.get(field) != actual_identity[field]:
            raise RuntimeError(f"Precomputed VAE cache differs from the loaded model: {field}.")
    return stored


def frozen_vae_identity(vae, dtype):
    digest = hashlib.sha256()
    for name, tensor in sorted(vae.state_dict().items()):
        digest.update(_json_bytes([name, tensor_digest(tensor)]))
    scale = [tensor_digest(value) if isinstance(value, torch.Tensor) else value
             for value in vae.scale]
    return {"format": 1, "weights_sha256": digest.hexdigest(),
            "scale": scale, "input_dtype": str(dtype)}


class VaeCacheCollator:
    """Collate packed latents without decoding or hashing RGB pixels."""
    def __call__(self, samples):
        batch = default_collate(samples)
        if "video_latents" in batch and "video" in batch:
            raise ValueError("A VAE cache batch cannot contain both pixels and indexed latents.")
        return batch


class VaeLatentCache:
    """Ordinary Python object: attaching it does not add model state_dict keys."""
    def __init__(self, root, vae, dtype, device, *, precomputed_namespace=None):
        self.vae, self.dtype, self.device = vae, dtype, torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self._assert_frozen()
        if precomputed_namespace is None:
            raise ValueError("VAE caching requires a packed episode index; online disk caching was removed.")
        self.identity = frozen_vae_identity(vae, dtype)
        cache_identity = precomputed_cache_identity(root, precomputed_namespace, self.identity)
        self.namespace = identity_hash(cache_identity)
        self.directory = Path(root) / self.namespace
        self._versions = self._version_signature()
        self.stats = {"hits": 0}

    def _assert_frozen(self):
        if any(module.training for module in self.vae.modules()):
            raise RuntimeError("VAE caching requires eval mode on every VAE module.")
        if any(parameter.requires_grad for parameter in self.vae.parameters()):
            raise RuntimeError("VAE caching requires frozen VAE parameters.")

    def _version_signature(self):
        tensors = list(self.vae.parameters()) + list(self.vae.buffers())
        tensors += [value for value in self.vae.scale if isinstance(value, torch.Tensor)]
        return ([(id(value), value._version, str(value.dtype), str(value.device)) for value in tensors]
                + [("scale_values", _json_bytes([
                    None if isinstance(value, torch.Tensor) else value for value in self.vae.scale]))])

    def _assert_current_identity(self):
        self._assert_frozen()
        if self._versions != self._version_signature():
            raise RuntimeError("VAE weights, buffers or scale changed after cache initialization.")

    @staticmethod
    def _validate_keys(keys, batch_size):
        if len(keys) != batch_size:
            raise ValueError("VAE cache key count does not match the batch.")
        if any(not isinstance(key, str) or len(key) != 64
               or any(char not in "0123456789abcdef" for char in key) for key in keys):
            raise ValueError("Invalid VAE cache content key.")

    @torch.no_grad()
    def accept_precomputed(self, latents, keys, namespaces, video_shape, tiled=False):
        """Accept worker-validated payloads without pixels, VAE encoding or GPU hashing."""
        self._assert_current_identity()
        if tiled:
            raise ValueError("Indexed VAE cache requires non-tiled encoding.")
        batch_size, channels, frames, height, width = video_shape
        if channels != 3 or min(batch_size, frames, height, width) <= 0:
            raise ValueError("Invalid original video shape for indexed VAE latents.")
        shape = (batch_size, self.vae.z_dim,
                 1 + (frames - 1) // self.vae.temporal_downsample_factor,
                 height // self.vae.upsampling_factor, width // self.vae.upsampling_factor)
        if (not isinstance(latents, torch.Tensor) or tuple(latents.shape) != shape
                or latents.dtype != self.dtype or latents.requires_grad
                or not bool(torch.isfinite(latents).all())):
            raise ValueError(f"Indexed VAE latents must be frozen {self.dtype} tensors with shape {shape}.")
        if latents.device.type != "cpu" and latents.device != self.device:
            raise ValueError("Indexed VAE latents must be on CPU or the model device.")
        self._validate_keys(keys, batch_size)
        if len(namespaces) != batch_size or any(value != self.namespace for value in namespaces):
            raise RuntimeError("Indexed VAE cache namespace differs from the loaded model.")
        self.stats["hits"] += batch_size
        return latents.to(device=self.device, non_blocking=True)
