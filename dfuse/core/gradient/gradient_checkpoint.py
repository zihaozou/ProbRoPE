import torch


def _is_zero3_active(module):
    """Check if DeepSpeed ZeRO-3 is managing this module's parameters."""
    p = next(module.parameters(), None)
    return p is not None and hasattr(p, "ds_id")


def _ensure_requires_grad(args):
    """Ensure at least one float tensor has requires_grad for autograd connectivity.

    DeepSpeed's activation checkpoint needs at least one input with
    requires_grad=True to connect the autograd graph. Pre-block ops
    (patchify, embedding) may use frozen params, producing activations
    without requires_grad.
    """
    if any(isinstance(a, torch.Tensor) and a.requires_grad for a in args):
        return args
    args = list(args)
    for i, a in enumerate(args):
        if isinstance(a, torch.Tensor) and a.is_floating_point():
            args[i] = a.detach().requires_grad_(True)
            break
    return tuple(args)


def get_checkpoint_fn(module, checkpoint=True, offload=False):
    """Return a ZeRO-3-aware activation checkpoint function.

    Centralizes all ZeRO-3 vs standard checkpoint logic in one place.
    Call once before a block loop, then use the returned function per block.

    Args:
        module: The nn.Module whose parameters determine ZeRO-3 status.
        checkpoint: If True, enable activation checkpointing (recomputation).
            If False but offload=True, only offload activations to CPU
            without recomputation — useful for attention blocks where
            compute grows quadratically but activation I/O grows linearly.
        offload: If True, offload activations to CPU memory.

    ZeRO-3 partitions parameters after forward. PyTorch's non-reentrant
    checkpoint (use_reentrant=False) compares tensor metadata between forward
    and recompute — but partitioned params have shape [0], causing a mismatch.
    DeepSpeed's activation checkpoint re-runs the forward (triggering ZeRO-3
    gather hooks) and doesn't check metadata.

    Usage::

        _ckpt = get_checkpoint_fn(self, checkpoint=True, offload=use_activation_offload)
        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                x = _ckpt(block_forward_fn, *block_args)
            else:
                x = block(*block_args)
    """
    if not checkpoint and not offload:

        def _passthrough(fn, *args):
            return fn(*args)
        return _passthrough

    if not checkpoint and offload:

        def _offload_only(fn, *args):
            with torch.autograd.graph.save_on_cpu():
                return fn(*args)
        return _offload_only

    if _is_zero3_active(module):
        from deepspeed.runtime.activation_checkpointing.checkpointing import (
            checkpoint as _ds_checkpoint,
        )

        def _checkpoint(fn, *args):
            args = _ensure_requires_grad(args)
            if offload:
                with torch.autograd.graph.save_on_cpu():
                    return _ds_checkpoint(fn, *args)
            return _ds_checkpoint(fn, *args)

        return _checkpoint

    def _checkpoint(fn, *args):
        if offload:
            with torch.autograd.graph.save_on_cpu():
                return torch.utils.checkpoint.checkpoint(
                    fn, *args, use_reentrant=False,
                )
        return torch.utils.checkpoint.checkpoint(
            fn, *args, use_reentrant=False,
        )

    return _checkpoint


def gradient_checkpoint_forward(
    model,
    use_gradient_checkpointing,
    use_activation_offload,
    *args,
    **kwargs,
):
    """Checkpoint wrapper for model forward with optional offload."""
    if not use_gradient_checkpointing and not use_activation_offload:
        return model(*args, **kwargs)


    if kwargs:
        def fn(*inputs):
            return model(*inputs, **kwargs)
    else:
        def fn(*inputs):
            return model(*inputs)

    _ckpt = get_checkpoint_fn(model, checkpoint=use_gradient_checkpointing, offload=use_activation_offload)
    return _ckpt(fn, *args)
