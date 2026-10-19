"""Bounded-memory RoboCOIN episode safetensors storage (no pickle payloads)."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile

import torch
from safetensors import safe_open

FORMAT = "robocoin-episode-v1"
LATENT_SHAPE = (48, 3, 24, 20)


def episode_path(directory, dataset, episode_index):
    if not dataset or Path(dataset).name != dataset or dataset in (".", ".."):
        raise ValueError("Invalid RoboCOIN dataset name")
    return Path(directory) / "episodes" / dataset / f"episode_{int(episode_index):06d}.safetensors"


def file_digest(path):
    """Offline audit only; never called by the training reader."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_name(frame, key):
    return f"{int(frame):08d}_{key}"


class EpisodeWriter:
    """Stream already encoded batches; publish only a complete checked episode."""
    def __init__(self, path, namespace, dataset, episode_index, keys):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.keys, self.position = list(keys), 0
        if not self.keys:
            raise ValueError("Cannot write an empty episode")
        size = 2
        for dimension in LATENT_SHAPE:
            size *= dimension
        header = {"__metadata__": {"format": FORMAT, "namespace": namespace,
                  "dataset": dataset, "episode_index": str(episode_index),
                  "windows": str(len(keys))}}
        for frame, key in enumerate(keys):
            header[tensor_name(frame, key)] = {"dtype": "BF16", "shape": list(LATENT_SHAPE),
                                               "data_offsets": [frame * size, (frame + 1) * size]}
        raw = json.dumps(header, separators=(",", ":")).encode()
        raw += b" " * (-len(raw) % 8)
        fd, name = tempfile.mkstemp(prefix=self.path.name + ".", suffix=".tmp", dir=self.path.parent)
        self.temporary = Path(name)
        self.file = os.fdopen(fd, "wb")
        self.file.write(struct.pack("<Q", len(raw)))
        self.file.write(raw)

    def append(self, latents):
        if (latents.dtype != torch.bfloat16 or tuple(latents.shape[1:]) != LATENT_SHAPE
                or latents.ndim != 5 or self.position + len(latents) > len(self.keys)
                or not bool(torch.isfinite(latents).all())):
            raise ValueError("Invalid RoboCOIN episode latent batch")
        values = latents.detach().cpu().contiguous().view(torch.uint8).numpy()
        self.file.write(memoryview(values).cast("B"))
        self.position += len(latents)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            if exc_type is None:
                if self.position != len(self.keys):
                    raise ValueError("Incomplete RoboCOIN episode; refusing publication")
                self.file.flush()
                os.fsync(self.file.fileno())
            self.file.close()
            if exc_type is None:
                os.replace(self.temporary, self.path)
        finally:
            self.file.close()
            self.temporary.unlink(missing_ok=True)


class EpisodeReader:
    """Lazy per-process LRU of mapped episode headers; read only requested tensors."""
    def __init__(self, directory, namespace, max_open=16):
        self.directory, self.namespace = Path(directory), namespace
        self.max_open = max_open
        self._pid, self._handles = None, OrderedDict()

    def __getstate__(self):
        return {**self.__dict__, "_pid": None, "_handles": OrderedDict()}

    def _open(self, dataset, episode_index, keys):
        if self._pid != os.getpid():
            self._handles = OrderedDict()
            self._pid = os.getpid()
        identity = (dataset, int(episode_index))
        path = episode_path(self.directory, *identity)
        try:
            stat = path.stat()
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Missing packed RoboCOIN episode: {path}. Regenerate old window caches.") from exc
        signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if identity in self._handles:
            previous, handle = self._handles[identity]
            if previous == signature:
                self._handles.move_to_end(identity)
                return handle
            del self._handles[identity]
        handle = safe_open(str(path), framework="pt", device="cpu")
        metadata = handle.metadata()
        expected = {"format": FORMAT, "namespace": self.namespace, "dataset": dataset,
                    "episode_index": str(episode_index), "windows": str(len(keys) // 32)}
        if metadata != expected:
            raise ValueError(f"RoboCOIN episode metadata mismatch: {path}")
        names = [tensor_name(i, keys[i * 32:(i + 1) * 32].hex()) for i in range(len(keys) // 32)]
        if set(handle.keys()) != set(names):
            raise ValueError(f"RoboCOIN episode key/input mapping mismatch: {path}")
        for name in names:
            tensor = handle.get_slice(name)
            if tuple(tensor.get_shape()) != LATENT_SHAPE or tensor.get_dtype() != "BF16":
                raise ValueError(f"RoboCOIN episode shape/dtype mismatch: {path}:{name}")
        self._handles[identity] = (signature, handle)
        while len(self._handles) > self.max_open:
            self._handles.popitem(last=False)
        return handle

    def load(self, dataset, episode_index, frame, keys):
        frame = int(frame)
        if not 0 <= frame < len(keys) // 32:
            raise IndexError(f"RoboCOIN frame out of range: {frame}")
        key = keys[frame * 32:(frame + 1) * 32].hex()
        latent = self._open(dataset, episode_index, keys).get_tensor(tensor_name(frame, key))
        if not bool(torch.isfinite(latent).all()):
            raise ValueError("Non-finite RoboCOIN latent")
        return latent
