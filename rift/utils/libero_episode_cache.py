"""Packed, independently checksummed LIBERO VAE latents for one episode."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import operator
import os
from pathlib import Path
import re
import tempfile

from safetensors import safe_open
from safetensors.torch import save_file
import torch

from .vae_latent_cache import tensor_digest


FORMAT = "libero_episode_bf16_v1"
TENSOR_NAME = "latents"
LATENT_SHAPE = (48, 3, 14, 28)
_HEX_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_METADATA_FIELDS = frozenset({
    "format", "namespace", "dataset", "episode_index", "index_sha256",
    "window_count", "window_sha256", "window_keys",
})


def _digest_ok(value):
    return isinstance(value, str) and _HEX_DIGEST.fullmatch(value) is not None


def _digest_list_ok(values, count):
    # Keep every per-window type/width check, then use the C hex decoder once.
    # fromhex permits uppercase and whitespace: lowercase equality and the
    # exact decoded byte count preserve the original stricter alphabet.
    if (not isinstance(values, (list, tuple)) or len(values) != count
            or not all(isinstance(value, str) and len(value) == 64 for value in values)):
        return False
    joined = "".join(values)
    if joined != joined.lower():
        return False
    try:
        return len(bytes.fromhex(joined)) == count * 32
    except ValueError:
        return False


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_identity(namespace, dataset, episode_index, index_sha256, window_count):
    if not _digest_ok(namespace) or not _digest_ok(index_sha256):
        raise ValueError("Invalid packed LIBERO namespace or index SHA-256")
    if not isinstance(dataset, str) or not dataset:
        raise ValueError("Invalid packed LIBERO dataset")
    if type(episode_index) is not int or episode_index < 0:
        raise ValueError("Invalid packed LIBERO episode index")
    if type(window_count) is not int or window_count < 1:
        raise ValueError("Invalid packed LIBERO window count")


def pack_episode(path, latents, *, namespace, dataset, episode_index,
                 index_sha256, window_sha256, window_keys):
    """Write a new episode file and return its file SHA-256.

    The destination must not exist. The caller publishes its manifest only after
    validating every episode. `latents` must not be changed while this runs.
    """
    if not isinstance(latents, torch.Tensor):
        raise ValueError("Packed LIBERO latents must be a tensor")
    if (latents.device.type != "cpu" or latents.dtype != torch.bfloat16
            or latents.ndim != 5 or tuple(latents.shape[1:]) != LATENT_SHAPE
            or not latents.is_contiguous()):
        raise ValueError("Packed LIBERO latents must be contiguous CPU BF16 [N, 48, 3, 14, 28]")
    count = latents.shape[0]
    _check_identity(namespace, dataset, episode_index, index_sha256, count)
    if not _digest_list_ok(window_sha256, count):
        raise ValueError("Invalid packed LIBERO window SHA-256 list")
    if not _digest_list_ok(window_keys, count):
        raise ValueError("Invalid packed LIBERO window key list")
    for frame, expected in enumerate(window_sha256):
        latent = latents[frame]
        if not bool(torch.isfinite(latent).all()) or tensor_digest(latent) != expected:
            raise ValueError(f"Packed LIBERO window {frame} differs from its SHA-256")

    metadata = {
        "format": FORMAT,
        "namespace": namespace,
        "dataset": dataset,
        "episode_index": str(episode_index),
        "index_sha256": index_sha256,
        "window_count": str(count),
        "window_sha256": json.dumps(list(window_sha256), separators=(",", ":")),
        "window_keys": json.dumps(list(window_keys), separators=(",", ":")),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # save_file writes to a path and may overwrite it. A private temporary path
    # plus link gives no-overwrite publication at the requested destination.
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".writing-",
                                     suffix=".safetensors", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        save_file({TENSOR_NAME: latents}, temporary, metadata=metadata)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        digest = _file_sha256(temporary)
        os.link(temporary, path)
        return digest
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class _ValidatedHeader:
    metadata: dict[str, str]
    # One 32-byte digest per window avoids retaining two large Python string lists.
    keys: bytes
    checksums: bytes


class EpisodeHeaderCache:
    """Validated episode headers, private to the process using this reader.

    Only Python metadata is retained, never a safetensors handle or tensor. A
    reader populated before DataLoader forks must revalidate in each child.
    """

    def __init__(self):
        self._pid = os.getpid()
        self._entries: dict[tuple, _ValidatedHeader] = {}

    def _for_process(self):
        if self._pid != os.getpid():
            self._entries.clear()
            self._pid = os.getpid()

    def get(self, identity):
        self._for_process()
        return self._entries.get(identity)

    def remember(self, identity, metadata, keys, checksums):
        self._for_process()
        self._entries[identity] = _ValidatedHeader(
            metadata, bytes.fromhex("".join(keys)),
            bytes.fromhex("".join(checksums)),
        )


def _checked_slice(handle, window_count):
    if handle.keys() != [TENSOR_NAME]:
        raise ValueError("Packed LIBERO episode must contain only the latents tensor")
    slab = handle.get_slice(TENSOR_NAME)
    if (slab.get_shape() != [window_count, *LATENT_SHAPE]
            or slab.get_dtype() != "BF16"):
        raise ValueError("Packed LIBERO tensor shape or dtype differs")
    return slab


def validate_header(handle, *, namespace, dataset, episode_index,
                    index_sha256, window_count):
    """Check episode identity and tensor layout; return keys and checksums."""
    _check_identity(namespace, dataset, episode_index, index_sha256, window_count)
    metadata = handle.metadata()
    if not isinstance(metadata, dict) or set(metadata) != _METADATA_FIELDS:
        raise ValueError("Invalid packed LIBERO metadata fields")
    expected = {
        "format": FORMAT,
        "namespace": namespace,
        "dataset": dataset,
        "episode_index": str(episode_index),
        "index_sha256": index_sha256,
        "window_count": str(window_count),
    }
    if any(metadata.get(name) != value for name, value in expected.items()):
        raise ValueError("Packed LIBERO episode metadata differs from the index")
    _checked_slice(handle, window_count)
    try:
        checksums = json.loads(metadata["window_sha256"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid packed LIBERO window SHA-256 metadata") from exc
    if not _digest_list_ok(checksums, window_count):
        raise ValueError("Invalid packed LIBERO window SHA-256 metadata")
    try:
        keys = json.loads(metadata["window_keys"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid packed LIBERO window key metadata") from exc
    if not _digest_list_ok(keys, window_count):
        raise ValueError("Invalid packed LIBERO window key metadata")
    return tuple(keys), tuple(checksums)


def load_window(path, start_frame, *, key, namespace, dataset, episode_index,
                index_sha256, window_count, header_cache=None):
    """Read one window; cache only fully validated episode metadata per worker.

    A cache hit still opens the current file, checks its entire raw metadata and
    tensor layout against the validated header, and checks the requested window
    key, shape, dtype, and finite values. Payload hashes are verified when packing,
    not recomputed on each training read. Any changed metadata
    fails closed rather than silently updating the cache.
    """
    try:
        start_frame = operator.index(start_frame)
    except TypeError as exc:
        raise TypeError("LIBERO window start frame must be an integer") from exc
    if isinstance(start_frame, bool) or not 0 <= start_frame < window_count:
        raise IndexError(f"LIBERO window start frame out of range: {start_frame}")
    if not _digest_ok(key):
        raise ValueError("Invalid indexed LIBERO window key")
    _check_identity(namespace, dataset, episode_index, index_sha256, window_count)
    if header_cache is not None and not isinstance(header_cache, EpisodeHeaderCache):
        raise TypeError("header_cache must be an EpisodeHeaderCache")
    identity = (os.fspath(path), namespace, dataset, episode_index,
                index_sha256, window_count)
    header = header_cache.get(identity) if header_cache is not None else None
    with safe_open(path, framework="pt", device="cpu") as handle:
        if header is None:
            keys, checksums = validate_header(
                handle, namespace=namespace, dataset=dataset,
                episode_index=episode_index, index_sha256=index_sha256,
                window_count=window_count,
            )
            metadata = handle.metadata()
            slab = handle.get_slice(TENSOR_NAME)
        else:
            metadata = handle.metadata()
            if metadata != header.metadata:
                raise RuntimeError(f"Packed LIBERO VAE episode metadata changed: {path}")
            slab = _checked_slice(handle, window_count)
        if header is None:
            expected_key = keys[start_frame]
        else:
            offset = start_frame * 32
            expected_key = header.keys[offset:offset + 32].hex()
        if expected_key != key:
            raise RuntimeError(f"Packed LIBERO VAE window key differs from index: {path}:{start_frame}")
        latent = slab[start_frame]
        if (latent.dtype != torch.bfloat16 or tuple(latent.shape) != LATENT_SHAPE
                or not bool(torch.isfinite(latent).all())):
            raise RuntimeError(f"Invalid or corrupted packed LIBERO VAE window: {path}:{start_frame}")
        if header is None and header_cache is not None:
            header_cache.remember(identity, metadata, keys, checksums)
        return latent
