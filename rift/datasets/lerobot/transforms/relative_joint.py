from typing import Any


class RelativeJointTransform:
    """Express joint-position targets relative to the first observed state."""

    def __init__(self, keys: list[str]):
        self.keys = keys

    def forward(self, batch: dict[str, Any]):
        if "action" not in batch:
            return batch
        for key in self.keys:
            batch["action"][key] = (
                batch["action"][key] - batch["state"][key][..., :1, :]
            )
        return batch

    def backward(self, batch: dict[str, Any]):
        for key in self.keys:
            batch["action"][key] = (
                batch["action"][key] + batch["state"][key][..., :1, :]
            )
        return batch
