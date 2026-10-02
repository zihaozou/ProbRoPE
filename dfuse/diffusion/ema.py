"""
Distributed Exponential Moving Average (EMA) for model parameters.

Compatible with:
  - DDP (plain distributed data-parallel)
  - FSDP v1 (FullyShardedDataParallel)
  - FSDP v2 (torch.distributed._composable.fsdp)
  - DeepSpeed ZeRO stages 1/2/3

EMA shadow weights are stored as **local shards** (matching the shard on
each rank), so no extra communication is needed for the EMA update.

All checkpoint I/O is handled by CheckpointHandler + ModelLogger.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .checkpoint import get_local_data


class DistributedEMA:
    """Sharded EMA that operates on local parameter shards.

    Shadow weights are keyed by **canonical (unwrapped) parameter names**.
    All methods that accept a model expect the **unwrapped** model so that
    ``named_parameters()`` yields canonical names.

    Parameters
    ----------
    model : nn.Module
        The **prepared** (wrapped by accelerator) model.
    decay : float
        EMA decay rate (e.g. 0.999 or 0.9999).
    trainable_only : bool
        If ``True`` (default), only track parameters with
        ``requires_grad=True``.
    accelerator : optional
        If provided, used to unwrap the model and build the wrapped->canonical
        name mapping.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        trainable_only: bool = True,
        accelerator=None,
    ):
        self.decay = decay
        self.trainable_only = trainable_only
        self.num_updates = 0


        _to_canonical: Dict[str, str] = {}
        if accelerator is not None:
            unwrapped = accelerator.unwrap_model(model)
            canonical_names = {id(p): n for n, p in unwrapped.named_parameters()}
            for name, param in model.named_parameters():
                _to_canonical[name] = canonical_names.get(id(param), name)
        else:
            for name, _ in model.named_parameters():
                _to_canonical[name] = name


        self.shadows: Dict[str, torch.Tensor] = {}

        for name, param in model.named_parameters():
            if trainable_only and not param.requires_grad:
                continue
            local = get_local_data(param)
            canon = _to_canonical[name]
            self.shadows[canon] = local.clone().detach()


    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow weights with the current model parameters.

        Uses Karras-style warm-up decay:
            ``decay_t = min(decay, (1 + num_updates) / (10 + num_updates))``

        Args:
            model: The **unwrapped** model (canonical parameter names).
        """
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        for name, param in model.named_parameters():
            if name not in self.shadows:
                continue
            local = get_local_data(param)
            shadow = self.shadows[name]
            shadow.lerp_(local, 1.0 - decay)


    @torch.no_grad()
    def copy_from_model(self, model: nn.Module) -> None:
        """Overwrite shadow weights with the current model parameters.

        Resets ``num_updates`` to 0 so the warm-up schedule starts fresh.

        Args:
            model: The **unwrapped** model (canonical parameter names).
        """
        self.num_updates = 0
        for name, param in model.named_parameters():
            if name not in self.shadows:
                continue
            local = get_local_data(param)
            self.shadows[name] = local.clone().detach()


    @torch.no_grad()
    def apply_to_model(self, model: nn.Module) -> None:
        """Swap model params with EMA shadows (in-place on local shards).

        Args:
            model: The **unwrapped** model (canonical parameter names).
        """
        for name, param in model.named_parameters():
            if name not in self.shadows:
                continue
            local = get_local_data(param)
            shadow = self.shadows[name]
            tmp = local.clone()
            local.copy_(shadow)
            shadow.copy_(tmp)

    @torch.no_grad()
    def restore_model(self, model: nn.Module) -> None:
        """Restore model params from the backup stored in shadows.

        This is the inverse of apply_to_model. Calling twice is a no-op.

        Args:
            model: The **unwrapped** model (canonical parameter names).
        """
        self.apply_to_model(model)
