"""Bounded, non-persistent caches for attention masks and fixed RoPE tensors."""
from __future__ import annotations

import torch


def exact_device_key(device):
    device = torch.device(device)
    index = device.index
    if device.type == "cuda" and index is None:
        index = torch.cuda.current_device()
    return device.type, index


def _version(tensor):
    try:
        return tensor._version
    except RuntimeError:  # Inference tensors do not track in-place mutations.
        return None


def _source_key(tensor):
    return (id(tensor), _version(tensor), tuple(tensor.shape), tensor.stride(),
            tensor.dtype, exact_device_key(tensor.device))


def cached_tensor(owner, slot, key, build, *, sources=()):
    """Keep one entry per fixed call-site slot, invalidating modified tensors."""
    sources = tuple(sources)
    if any(source.requires_grad for source in sources):
        return build()
    source_keys = tuple(_source_key(source) for source in sources)
    reusable = all(source_key[1] is not None for source_key in source_keys)
    cache = getattr(owner, "_static_tensor_cache", None)
    if cache is None:
        cache = owner._static_tensor_cache = {}
    full_key = (key, source_keys)
    entry = cache.get(slot)
    if reusable and entry is not None:
        previous_key, value, version, _sources = entry
        if previous_key == full_key and _version(value) == version:
            return value

    # A cache first populated during inference must remain usable by autograd.
    with torch.inference_mode(False), torch.no_grad():
        value = build()
        if torch.is_inference(value):
            original_stride = value.stride()
            value = value.clone(memory_format=torch.preserve_format)
            if value.stride() != original_stride:
                value = torch.empty_strided(value.shape, original_stride,
                                            dtype=value.dtype, device=value.device).copy_(value)
    if reusable:
        # Retain sources so Python cannot recycle their IDs while this entry lives.
        cache[slot] = (full_key, value, _version(value), sources)
    else:
        cache.pop(slot, None)
    return value
