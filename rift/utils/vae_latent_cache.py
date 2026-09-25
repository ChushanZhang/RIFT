"""Cache frozen VAE outputs or consume immutable precomputed training latents.

Online cache writes use the full encoding identity. Indexed inputs select their
existing namespace and require matching weights, scale and dtype; the encoder's
package, source and execution environment may differ when no encoding is run.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import tempfile

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


def device_uuid(device):
    device = torch.device(device)
    if device.type != "cuda":
        return str(device)
    value = getattr(torch.cuda.get_device_properties(device), "uuid", None)
    if value is None:
        raise RuntimeError("CUDA device UUID is unavailable for the parity report.")
    return str(value)


def numerical_flags():
    """Read mutable numerical settings without device queries or tensor hashing."""
    flags = {
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
    }
    for setting in ("flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled",
                    "cudnn_sdp_enabled", "fp16_bf16_reduction_math_sdp_allowed"):
        getter = getattr(torch.backends.cuda, setting, None)
        flags[setting] = getter() if getter is not None else None
    return flags


def encoding_identity(vae, dtype, device):
    """Compute once after model loading/resume; include weights outside state_dict."""
    device = torch.device(device)
    digest = hashlib.sha256()
    for name, tensor in sorted(vae.state_dict().items()):
        digest.update(_json_bytes([name, tensor_digest(tensor)]))
    scale = [tensor_digest(value) if isinstance(value, torch.Tensor) else value
             for value in vae.scale]
    source = Path(inspect.getfile(type(vae)))
    environment = {
        "torch": str(torch.__version__), "torch_git": torch.version.git_version,
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "device_type": device.type,
        **numerical_flags(),
    }
    if device.type == "cuda":
        environment.update(
            gpu_model=torch.cuda.get_device_name(device),
            compute_capability=list(torch.cuda.get_device_capability(device)),
            driver=subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
            ).splitlines()[0].strip(),
        )
    return {
        "format": 1, "vae_class": f"{type(vae).__module__}.{type(vae).__qualname__}",
        "weights_sha256": digest.hexdigest(), "scale": scale,
        "encode_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "input_dtype": str(dtype),
        "autocast_enabled": torch.is_autocast_enabled(device.type),
        "autocast_dtype": str(torch.get_autocast_dtype(device.type)),
        "environment": environment,
    }


class VaeCacheCollator:
    """Collate indexed latents, or append content keys for original CPU pixels."""
    def __call__(self, samples):
        batch = default_collate(samples)
        if "video_latents" in batch:
            if "video" in batch:
                raise ValueError("A VAE cache batch cannot contain both pixels and indexed latents.")
            return batch
        if batch["video"].device.type != "cpu":
            raise ValueError("VAE cache keys must be generated before device placement.")
        batch["vae_cache_keys"] = [tensor_digest(video) for video in batch["video"]]
        return batch


class VaeLatentCache:
    """Ordinary Python object: attaching it does not add model state_dict keys."""
    def __init__(self, root, vae, dtype, device, *, precomputed_namespace=None):
        self.vae, self.dtype, self.device = vae, dtype, torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self._assert_frozen()
        self.identity = encoding_identity(vae, dtype, device)
        self._precomputed_only = precomputed_namespace is not None
        cache_identity = (precomputed_cache_identity(root, precomputed_namespace, self.identity)
                          if self._precomputed_only else self.identity)
        self._numerical_flags = {
            key: self.identity["environment"][key] for key in numerical_flags()
        }
        self.namespace = identity_hash(cache_identity)
        self.directory = Path(root) / self.namespace
        self._versions = self._version_signature()
        self.stats = {"hits": 0, "misses": 0}
        if not self._precomputed_only:
            self._publish(self.directory / "identity.json", _json_bytes(cache_identity))
        if json.loads((self.directory / "identity.json").read_text()) != cache_identity:
            raise RuntimeError("VAE cache namespace identity mismatch.")

    def _assert_frozen(self):
        if any(module.training for module in self.vae.modules()):
            raise RuntimeError("VAE caching requires eval mode on every VAE module.")
        if any(parameter.requires_grad for parameter in self.vae.parameters()):
            raise RuntimeError("VAE caching requires frozen VAE parameters.")

    def _version_signature(self):
        tensors = list(self.vae.parameters()) + list(self.vae.buffers())
        tensors += [value for value in self.vae.scale if isinstance(value, torch.Tensor)]
        return [(id(value), value._version, str(value.dtype), str(value.device)) for value in tensors]

    @staticmethod
    def _publish(path, content):
        """Publish without overwriting another rank's entry; never expose partial bytes."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pending-", delete=False) as temp:
            temporary = Path(temp.name)
            try:
                if isinstance(content, bytes):
                    temp.write(content)
                else:
                    torch.save(content, temp)
                temp.flush()
                os.fsync(temp.fileno())
                try:
                    os.link(temporary, path)
                except FileExistsError:
                    pass
            finally:
                temporary.unlink(missing_ok=True)

    def _load(self, path, key, shape):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        latent = payload["latent"]
        if (payload["namespace"] != self.namespace or payload["key"] != key
                or latent.dtype != self.dtype or tuple(latent.shape) != shape
                or payload["sha256"] != tensor_digest(latent)):
            raise RuntimeError(f"Invalid or corrupted VAE latent cache: {path}")
        return latent

    def _assert_current_identity(self):
        self._assert_frozen()
        if self._versions != self._version_signature():
            raise RuntimeError("VAE weights, buffers or scale changed after cache initialization.")
        if numerical_flags() != self._numerical_flags:
            raise RuntimeError("VAE numerical settings changed after cache initialization.")
        if (torch.is_autocast_enabled(self.device.type) != self.identity["autocast_enabled"]
                or str(torch.get_autocast_dtype(self.device.type)) != self.identity["autocast_dtype"]):
            raise RuntimeError("VAE autocast context differs from cache initialization.")

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
                or latents.dtype != self.dtype or latents.requires_grad):
            raise ValueError(f"Indexed VAE latents must be frozen {self.dtype} tensors with shape {shape}.")
        if latents.device.type != "cpu" and latents.device != self.device:
            raise ValueError("Indexed VAE latents must be on CPU or the model device.")
        self._validate_keys(keys, batch_size)
        if len(namespaces) != batch_size or any(value != self.namespace for value in namespaces):
            raise RuntimeError("Indexed VAE cache namespace differs from the loaded model.")
        self.stats["hits"] += batch_size
        return latents.to(device=self.device, non_blocking=True)

    @torch.no_grad()
    def encode(self, videos, keys, tiled=False):
        if self._precomputed_only:
            raise RuntimeError("An indexed VAE cache accepts precomputed latents only.")
        self._assert_current_identity()
        if tiled or videos.dtype != self.dtype or videos.device != self.device:
            raise ValueError("VAE cache requires the original non-tiled device and dtype.")
        self._validate_keys(keys, len(videos))
        _, _, frames, height, width = videos.shape
        shape = (self.vae.z_dim, 1 + (frames - 1) // self.vae.temporal_downsample_factor,
                 height // self.vae.upsampling_factor, width // self.vae.upsampling_factor)
        latents = []
        for video, key in zip(videos, keys):
            path = self.directory / key[:2] / f"{key}.pt"
            try:
                latent = self._load(path, key, shape).to(device=videos.device)
                self.stats["hits"] += 1
            except FileNotFoundError:
                # These are the same calls and shapes used by WanVideoVAE.encode.
                latent = self.vae.single_encode(video.unsqueeze(0), videos.device).squeeze(0)
                if latent.dtype != self.dtype or tuple(latent.shape) != shape:
                    raise RuntimeError("Original VAE output differs from the cache contract.")
                stored = latent.detach().cpu().contiguous()
                payload = {"namespace": self.namespace, "key": key, "latent": stored,
                           "sha256": tensor_digest(stored)}
                self._publish(path, payload)
                winner = self._load(path, key, shape)
                if not torch.equal(winner.reshape(-1).view(torch.uint8), stored.reshape(-1).view(torch.uint8)):
                    raise RuntimeError(f"Concurrent VAE encodes differ bitwise for key {key}.")
                self.stats["misses"] += 1
            latents.append(latent)
        return torch.stack(latents)
