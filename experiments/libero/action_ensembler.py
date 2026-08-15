"""Temporal action ensembling used by the LIBERO rollout worker."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


class ActionEnsembler:
    """Average all action-chunk predictions targeting the same control step."""

    def __init__(self) -> None:
        self.action_cache: dict[int, list[np.ndarray]] = defaultdict(list)

    def reset(self) -> None:
        self.action_cache.clear()

    def add_actions(self, action_chunk: np.ndarray, start_timestamp: int) -> None:
        action_chunk = np.asarray(action_chunk)
        if action_chunk.ndim == 3 and action_chunk.shape[0] == 1:
            action_chunk = action_chunk[0]
        if action_chunk.ndim != 2:
            raise ValueError(
                f"action_chunk must be [T, D] or [1, T, D], got {action_chunk.shape}."
            )
        for offset, action in enumerate(action_chunk):
            self.action_cache[int(start_timestamp) + offset].append(action.copy())

    def get_action(self, timestamp: int) -> np.ndarray:
        timestamp = int(timestamp)
        if timestamp not in self.action_cache:
            raise ValueError(f"No actions cached for timestamp {timestamp}.")
        action = np.mean(np.stack(self.action_cache[timestamp], axis=0), axis=0)
        self._cleanup(timestamp)
        return action

    def _cleanup(self, current_timestamp: int) -> None:
        for timestamp in tuple(self.action_cache):
            if timestamp < current_timestamp:
                del self.action_cache[timestamp]
