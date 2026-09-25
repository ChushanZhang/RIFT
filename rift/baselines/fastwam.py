"""Use indexed RIFT VAE inputs with the original FastWAM baseline models.

The external ``fastwam`` package remains responsible for model construction,
losses, attention, optimization, and checkpointing. Only input preparation and
the training collator are adapted here.
"""
from types import MethodType

from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.fastwam_joint import FastWAMJoint
from fastwam.runtime import create_fastwam as _create_fastwam
from fastwam.runtime import create_fastwam_joint as _create_fastwam_joint
from fastwam.trainer import Wan22Trainer

from rift.models.wan22.fastwam import FastWAM as _RiftInputModel
from rift.utils.vae_latent_cache import VaeCacheCollator, VaeLatentCache


def attach_indexed_vae_inputs(model):
    """Adapt exact supported baseline classes without replacing their losses."""
    if type(model) not in (FastWAM, FastWAMJoint):
        raise TypeError("Indexed baseline inputs support only original FastWAM and FastWAMJoint classes.")
    if hasattr(model, "_uncached_build_inputs"):
        if getattr(model.build_inputs, "__func__", None) is not _RiftInputModel.build_inputs:
            raise RuntimeError("Baseline build_inputs changed after attaching indexed VAE inputs.")
        return model
    model._uncached_build_inputs = model.build_inputs
    model.build_inputs = MethodType(_RiftInputModel.build_inputs, model)
    return model


def create_fastwam(*args, **kwargs):
    return attach_indexed_vae_inputs(_create_fastwam(*args, **kwargs))


def create_fastwam_joint(*args, **kwargs):
    return attach_indexed_vae_inputs(_create_fastwam_joint(*args, **kwargs))


class IndexedWan22Trainer(Wan22Trainer):
    """Original baseline trainer with a validated frozen-VAE cache input path."""

    def __init__(self, model, train_dataset, val_dataset=None, *, cfg):
        self.vae_cache_dir = cfg.get("vae_cache_dir", None)
        dataset_namespace = getattr(train_dataset, "vae_cache_namespace", None)
        if dataset_namespace is not None and not self.vae_cache_dir:
            raise ValueError("Indexed VAE datasets require top-level `vae_cache_dir`.")
        if self.vae_cache_dir and int(cfg.eval_every) > 0:
            raise ValueError("The indexed baseline bridge supports training only; set `eval_every=0`.")
        attach_indexed_vae_inputs(model)
        super().__init__(model, train_dataset, val_dataset, cfg=cfg)
        if self.vae_cache_dir:
            # Original constructor restores the checkpoint before cache identity checks.
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            with self.accelerator.autocast():
                unwrapped_model._vae_latent_cache = VaeLatentCache(
                    self.vae_cache_dir, unwrapped_model.vae,
                    unwrapped_model.torch_dtype, unwrapped_model.device,
                    precomputed_namespace=dataset_namespace,
                )

    def _build_loader(self, dataset, worker_init_fn=None):
        loader = super()._build_loader(dataset, worker_init_fn=worker_init_fn)
        if self.vae_cache_dir:
            loader.collate_fn = VaeCacheCollator()
        return loader
