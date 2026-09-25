"""Checkpoint validation shared by standalone benchmark evaluators."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_rift_checkpoint_exact(model: Any, checkpoint: str | Path) -> dict:
    """Load a released RIFT payload and reject partial policy state."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    if not hasattr(model, "mot") or not hasattr(model, "load_checkpoint"):
        raise TypeError(
            f"{type(model).__name__} does not expose the RIFT checkpoint interface."
        )

    payload = model.load_checkpoint(str(checkpoint))
    if not isinstance(payload, dict) or not isinstance(payload.get("mot"), dict):
        raise ValueError(f"Invalid RIFT checkpoint payload: {checkpoint}")

    expected = set(model.mot.state_dict())
    actual = set(payload["mot"])
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(
            "RIFT checkpoint schema mismatch: "
            f"missing={missing[:10]} unexpected={unexpected[:10]}"
        )
    if getattr(model, "proprio_encoder", None) is not None and not isinstance(
        payload.get("proprio_encoder"), dict
    ):
        raise ValueError("RIFT checkpoint is missing `proprio_encoder` state.")
    return payload
