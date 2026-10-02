"""Selective module sharding for DeepSpeed ZeRO-3 and FSDP.

Allows specifying which submodules of a pipeline should be sharded
(managed by the distributed backend) vs replicated (kept whole on
each rank).  Unsharded modules remain accessible as plain attributes.
"""

import torch.nn as nn


class ModuleShardingManager:
    """Manages selective sharding of pipeline submodules.

    Usage::

        manager = ModuleShardingManager(pipe, shard_models=["dit"])
        manager.detach()           # before accelerator.prepare()
        # ... accelerator.prepare(model) only shards dit ...

    Parameters
    ----------
    pipe : nn.Module
        The pipeline module whose children will be selectively sharded.
    shard_models : list[str]
        Names of child modules that should remain in the ``nn.Module``
        tree (and thus be sharded by the distributed backend).  All
        other children with parameters are detached.
    """

    def __init__(self, pipe: nn.Module, shard_models: list[str]):
        self._pipe = pipe
        self._shard_models = set(shard_models)
        self._detached: dict[str, nn.Module] = {}

    @property
    def detached_module_names(self) -> list[str]:
        """Names of modules that were detached (replicated on each rank)."""
        return list(self._detached.keys())

    def detach(self):
        """Remove non-sharded modules from the ``nn.Module`` tree.

        After this call, detached modules are still accessible as
        ``pipe.<name>`` but invisible to ``pipe.parameters()`` and
        ``pipe.named_children()``, so ``accelerator.prepare()`` won't
        shard them.

        Must be called **after** ``freeze_except()`` /
        ``cache_trainable_param_names()`` and **before**
        ``accelerator.prepare()``.
        """
        for name in list(self._pipe._modules.keys()):
            module = self._pipe._modules[name]
            if module is None or name in self._shard_models:
                continue
            if next(module.parameters(), None) is None:
                continue
            del self._pipe._modules[name]
            object.__setattr__(self._pipe, name, module)
            self._detached[name] = module

        if self._detached:
            sharded = [
                n for n, m in self._pipe._modules.items() if m is not None
            ]
            print(
                f"[ShardingManager] Detached (replicated): "
                f"{self.detached_module_names}"
            )
            print(
                f"[ShardingManager] Remaining (sharded):   {sharded}"
            )

    @staticmethod
    def derive_shard_models(trainable_models: str) -> list[str]:
        """Extract unique top-level module names from a trainable_models string.

        Examples::

            >>> ModuleShardingManager.derive_shard_models(
            ...     "dit.blocks.*.self_attn;dit.blocks.*.cam_encoder"
            ... )
            ['dit']
            >>> ModuleShardingManager.derive_shard_models("dit;text_encoder")
            ['dit', 'text_encoder']
        """
        top_levels = set()
        for name in trainable_models.split(";"):
            part = name.strip().split(".")[0]
            if part:
                top_levels.add(part)
        return sorted(top_levels)
