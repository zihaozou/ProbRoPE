"""
UnevenTensor: A container for batched tensors with variable spatial dimensions.

Stores data in packed format where all non-channel dimensions are flattened into
a single "element" dimension. This enables efficient batching of videos with
different (T, H, W) per sample.

Key concepts:
- `channel_dim`: identifies which dimension is the channel/feature dimension
  (shared size across all samples). All other dimensions can vary per sample.
- Data is stored as (total_elements, channel_size) where total_elements is the
  sum of products of non-channel dims across all samples.
- cu_seqlens tracks per-sample boundaries for flash_attn_varlen.

Example usage:
    # Video latents (C, T, H, W) with channel_dim=0
    latents = [torch.randn(16, 5, 32, 32), torch.randn(16, 9, 16, 16)]
    ut = UnevenTensor(latents, channel_dim=0)
    ut.seq_lens      # [5*32*32, 9*16*16] = [5120, 2304]
    ut.sample_shapes  # [(5, 32, 32), (9, 16, 16)]
    ut[0]             # reconstructed (16, 5, 32, 32) tensor

    # Text embeddings (L, D) with channel_dim=-1
    texts = [torch.randn(17, 4096), torch.randn(33, 4096)]
    ut = UnevenTensor(texts, channel_dim=-1)
    ut.seq_lens      # [17, 33]
    ut.sample_shapes  # [(17,), (33,)]

    # Pack for varlen attention
    packed = ut.to_packed()
    # packed.data: (total_elements, channel_size)
    # packed.cu_seqlens: [0, 5120, 7424]
"""

from __future__ import annotations
import torch
import math
from typing import List, Callable, Optional, Tuple, Union, Any
from dataclasses import dataclass


@dataclass
class PackedSequence:
    """Packed sequence representation for variable-length attention.

    Attributes:
        data: Packed tensor (total_tokens, ...)
        cu_seqlens: Cumulative sequence lengths (batch_size + 1,)
        max_seqlen: Maximum sequence length in the batch
        seq_lens: List of individual sequence lengths
    """

    data: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int
    seq_lens: List[int]

    def to(self, device: torch.device) -> "PackedSequence":
        """Move to device."""
        return PackedSequence(
            data=self.data.to(device),
            cu_seqlens=self.cu_seqlens.to(device),
            max_seqlen=self.max_seqlen,
            seq_lens=self.seq_lens,
        )

    @property
    def batch_size(self) -> int:
        return len(self.seq_lens)

    @property
    def total_tokens(self) -> int:
        return self.data.shape[0]


@dataclass
class ContextEncoderOutput:
    """Output from the context encoder for cross-attention in the DiT.

    Attributes:
        packed_sequence: PackedSequence with context token embeddings
        camera_embed: (total_ctx_tokens, dim) camera embedding per token
        freqs: (num_rope_ctx_tokens, 1, head_dim//2) RoPE frequencies
        rope_mask: (total_ctx_tokens,) bool — True for spatial tokens
    """

    packed_sequence: PackedSequence
    camera_embed: torch.Tensor
    freqs: torch.Tensor
    rope_mask: torch.Tensor

    def to(self, device: torch.device) -> "ContextEncoderOutput":
        """Move to device."""
        return ContextEncoderOutput(
            packed_sequence=self.packed_sequence.to(device),
            camera_embed=self.camera_embed.to(device),
            freqs=self.freqs.to(device),
            rope_mask=self.rope_mask.to(device),
        )

    @property
    def batch_size(self) -> int:
        return self.packed_sequence.batch_size


class UnevenTensor:
    """Container for batched tensors with variable spatial dimensions.

    Data is stored in packed format: all non-channel dimensions are flattened
    and concatenated across samples into a single (total_elements, channel_size)
    tensor. Per-sample shapes are stored as metadata for reconstruction.

    Attributes:
        data: Packed tensor of shape (total_elements, channel_size).
        channel_dim: Which dimension in the original tensors is the channel dim.
        sample_shapes: Per-sample non-channel dimension shapes.
    """

    def __init__(self, tensors: List[torch.Tensor], channel_dim: int = 0):
        """Create UnevenTensor from a list of tensors.

        Args:
            tensors: List of tensors. The channel_dim must have the same size
                     across all samples. All other dimensions can differ.
            channel_dim: Which dimension is the channel/feature dimension.
                         For video latents (C, T, H, W), use channel_dim=0.
                         For text embeddings (L, D), use channel_dim=-1.
        """
        if not tensors:
            raise ValueError("Cannot create UnevenTensor from empty list")

        ndim = tensors[0].ndim

        if channel_dim < 0:
            channel_dim = ndim + channel_dim

        self.channel_dim = channel_dim
        self._ndim = ndim
        self._channel_size = tensors[0].shape[channel_dim]


        for i, t in enumerate(tensors[1:], 1):
            if t.ndim != ndim:
                raise ValueError(
                    f"Tensor ndim mismatch at index {i}: "
                    f"expected {ndim}, got {t.ndim}"
                )
            if t.shape[channel_dim] != self._channel_size:
                raise ValueError(
                    f"Channel size mismatch at index {i}: "
                    f"expected {self._channel_size}, got {t.shape[channel_dim]}"
                )


        self._perm = list(range(ndim))
        self._perm.pop(channel_dim)
        self._perm.append(channel_dim)
        self._inv_perm = [0] * ndim
        for i, p in enumerate(self._perm):
            self._inv_perm[p] = i


        packed_chunks = []
        self.sample_shapes: List[Tuple[int, ...]] = []
        for t in tensors:
            t_perm = t.permute(*self._perm)
            non_channel_shape = t_perm.shape[:-1]
            self.sample_shapes.append(tuple(non_channel_shape))
            packed_chunks.append(t_perm.reshape(-1, self._channel_size))

        self.data = torch.cat(packed_chunks, dim=0)


        self._seq_lens = [c.shape[0] for c in packed_chunks]
        self._cu_seqlens = torch.zeros(
            len(self._seq_lens) + 1, dtype=torch.int32, device=self.data.device
        )
        self._cu_seqlens[1:] = torch.cumsum(
            torch.tensor(self._seq_lens, dtype=torch.int32, device=self.data.device),
            dim=0,
        )


    def _clone_with_data(self, new_data: torch.Tensor) -> "UnevenTensor":
        """Create new UnevenTensor with different data but same metadata."""
        result = UnevenTensor.__new__(UnevenTensor)
        result.data = new_data
        result.channel_dim = self.channel_dim
        result.sample_shapes = self.sample_shapes
        result._ndim = self._ndim
        result._channel_size = self._channel_size
        result._perm = self._perm
        result._inv_perm = self._inv_perm
        result._seq_lens = self._seq_lens
        result._cu_seqlens = self._cu_seqlens
        return result

    @classmethod
    def _from_raw(
        cls,
        data: torch.Tensor,
        channel_dim: int,
        sample_shapes: List[Tuple[int, ...]],
        ndim: int,
        channel_size: int,
        cu_seqlens: torch.Tensor,
        seq_lens: List[int],
    ) -> "UnevenTensor":
        """From raw."""
        result = cls.__new__(cls)
        result.data = data
        result.channel_dim = channel_dim
        result.sample_shapes = sample_shapes
        result._ndim = ndim
        result._channel_size = channel_size
        result._seq_lens = seq_lens
        result._cu_seqlens = cu_seqlens

        perm = list(range(ndim))
        perm.pop(channel_dim)
        perm.append(channel_dim)
        result._perm = perm
        inv_perm = [0] * ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        result._inv_perm = inv_perm
        return result


    @property
    def seq_lens(self) -> List[int]:
        """Per-sample element counts (product of non-channel dims)."""
        return self._seq_lens

    @property
    def total_tokens(self) -> int:
        """Total elements across all samples."""
        return self.data.shape[0]

    @property
    def max_seqlen(self) -> int:
        """Maximum per-sample element count."""
        return max(self._seq_lens) if self._seq_lens else 0

    @property
    def batch_size(self) -> int:
        """Number of samples."""
        return len(self.sample_shapes)

    @property
    def device(self) -> torch.device:
        return self.data.device

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype

    @property
    def channel_size(self) -> int:
        """Size of the channel dimension."""
        return self._channel_size

    def __len__(self) -> int:
        return self.batch_size

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Reconstruct per-sample tensor in its original shape."""
        start = self._cu_seqlens[idx].item()
        end = self._cu_seqlens[idx + 1].item()
        chunk = self.data[start:end]

        shape = self.sample_shapes[idx] + (self._channel_size,)
        chunk = chunk.reshape(*shape)

        return chunk.permute(*self._inv_perm)

    def __iter__(self):
        for i in range(self.batch_size):
            yield self[i]

    def __repr__(self) -> str:
        return (
            f"UnevenTensor(batch_size={self.batch_size}, "
            f"channel_dim={self.channel_dim}, "
            f"channel_size={self._channel_size}, "
            f"sample_shapes={self.sample_shapes}, "
            f"dtype={self.dtype}, device={self.device})"
        )


    def to(self, device: torch.device) -> "UnevenTensor":
        """Move to device."""
        return UnevenTensor._from_raw(
            data=self.data.to(device),
            channel_dim=self.channel_dim,
            sample_shapes=self.sample_shapes,
            ndim=self._ndim,
            channel_size=self._channel_size,
            cu_seqlens=self._cu_seqlens.to(device),
            seq_lens=self._seq_lens,
        )

    def to_dtype(self, dtype: torch.dtype) -> "UnevenTensor":
        """Convert data to dtype."""
        return self._clone_with_data(self.data.to(dtype))


    def to_packed(self) -> PackedSequence:
        """To packed."""
        return PackedSequence(
            data=self.data,
            cu_seqlens=self._cu_seqlens,
            max_seqlen=self.max_seqlen,
            seq_lens=self._seq_lens,
        )

    @classmethod
    def from_packed(
        cls,
        packed: PackedSequence,
        channel_dim: int = 0,
        sample_shapes: Optional[List[Tuple[int, ...]]] = None,
    ) -> "UnevenTensor":
        """Reconstruct UnevenTensor from PackedSequence.

        Args:
            packed: PackedSequence with data shape (total, channel_size).
            channel_dim: Channel dimension in the original tensor layout.
            sample_shapes: Per-sample non-channel shapes. If None, assumes
                           1D shapes from seq_lens.
        """
        if sample_shapes is None:
            sample_shapes = [(sl,) for sl in packed.seq_lens]

        channel_size = packed.data.shape[-1] if packed.data.ndim > 1 else 1
        ndim = len(sample_shapes[0]) + 1

        return cls._from_raw(
            data=packed.data,
            channel_dim=channel_dim,
            sample_shapes=sample_shapes,
            ndim=ndim,
            channel_size=channel_size,
            cu_seqlens=packed.cu_seqlens,
            seq_lens=packed.seq_lens,
        )


    @staticmethod
    def apply(
        fn: Callable[..., torch.Tensor],
        *args,
        **kwargs,
    ) -> "UnevenTensor":
        """Apply a function per-sample across UnevenTensors.

        UnevenTensor arguments are unpacked to per-sample tensors via
        __getitem__. Lists/tuples matching batch_size are indexed per-sample.
        Other arguments are broadcast.

        Args:
            fn: Function taking tensors, returning a tensor.
            *args: Positional args (UnevenTensors auto-unwrapped).
            **kwargs: Keyword args (UnevenTensors auto-unwrapped).

        Returns:
            New UnevenTensor with results. Inherits channel_dim from the
            first UnevenTensor argument.
        """
        uneven_tensors = []
        for arg in args:
            if isinstance(arg, UnevenTensor):
                uneven_tensors.append(arg)
        for val in kwargs.values():
            if isinstance(val, UnevenTensor):
                uneven_tensors.append(val)

        if not uneven_tensors:
            raise ValueError("At least one UnevenTensor must be provided")

        batch_size = uneven_tensors[0].batch_size
        channel_dim = uneven_tensors[0].channel_dim
        for ut in uneven_tensors[1:]:
            if ut.batch_size != batch_size:
                raise ValueError(
                    f"Batch size mismatch: {batch_size} vs {ut.batch_size}"
                )

        results = []
        for i in range(batch_size):
            unwrapped_args = [
                a[i] if isinstance(a, UnevenTensor)
                else a[i] if isinstance(a, (list, tuple)) and len(a) == batch_size
                else a
                for a in args
            ]
            unwrapped_kwargs = {
                k: v[i] if isinstance(v, UnevenTensor)
                else v[i] if isinstance(v, (list, tuple)) and len(v) == batch_size
                else v
                for k, v in kwargs.items()
            }
            results.append(fn(*unwrapped_args, **unwrapped_kwargs))

        return UnevenTensor(results, channel_dim=channel_dim)


    def extend(self, other: "UnevenTensor", dim: int) -> "UnevenTensor":
        """Concatenate with another UnevenTensor along a non-channel dim.

        For video latents (C, T, H, W) with channel_dim=0, dim=1 concatenates
        along the temporal dimension (e.g., [target | source]).

        Within a sample, the non-extended dimensions must match between self
        and other.

        Args:
            other: Another UnevenTensor with same batch_size and channel_dim.
            dim: Dimension in the original tensor shape (not packed shape).
        """
        if dim == self.channel_dim:
            raise ValueError(
                f"Cannot extend along channel_dim={self.channel_dim}"
            )
        if self.batch_size != other.batch_size:
            raise ValueError(
                f"Batch size mismatch: {self.batch_size} vs {other.batch_size}"
            )
        if self.channel_dim != other.channel_dim:
            raise ValueError(
                f"channel_dim mismatch: {self.channel_dim} vs {other.channel_dim}"
            )


        nc_dim = dim if dim < self.channel_dim else dim - 1

        if nc_dim == 0:


            new_chunks = []
            new_shapes = []
            new_seq_lens = []
            for i in range(self.batch_size):
                s1 = self._cu_seqlens[i].item()
                e1 = self._cu_seqlens[i + 1].item()
                s2 = other._cu_seqlens[i].item()
                e2 = other._cu_seqlens[i + 1].item()
                new_chunks.append(self.data[s1:e1])
                new_chunks.append(other.data[s2:e2])

                old_shape = self.sample_shapes[i]
                oth_shape = other.sample_shapes[i]
                if old_shape[1:] != oth_shape[1:] and oth_shape[0] > 0:
                    raise ValueError(
                        f"Cannot extend along nc_dim=0: non-extended dims "
                        f"differ for sample {i}: {old_shape[1:]} vs "
                        f"{oth_shape[1:]}. Ensure source and target have "
                        f"the same spatial resolution."
                    )
                new_shape = (old_shape[0] + oth_shape[0],) + old_shape[1:]
                new_shapes.append(new_shape)
                new_seq_lens.append((e1 - s1) + (e2 - s2))

            new_data = torch.cat(new_chunks, dim=0)
            new_cu = torch.zeros(
                self.batch_size + 1, dtype=torch.int32, device=new_data.device
            )
            new_cu[1:] = torch.cumsum(
                torch.tensor(new_seq_lens, dtype=torch.int32, device=new_data.device),
                dim=0,
            )
            return UnevenTensor._from_raw(
                data=new_data,
                channel_dim=self.channel_dim,
                sample_shapes=new_shapes,
                ndim=self._ndim,
                channel_size=self._channel_size,
                cu_seqlens=new_cu,
                seq_lens=new_seq_lens,
            )
        else:

            tensors = []
            for i in range(self.batch_size):
                t1 = self[i]
                t2 = other[i]
                tensors.append(torch.cat([t1, t2], dim=dim))
            return UnevenTensor(tensors, channel_dim=self.channel_dim)


    def clone(self) -> "UnevenTensor":
        """Deep copy (clones the packed data tensor)."""
        return self._clone_with_data(self.data.clone())

    def detach(self) -> "UnevenTensor":
        """Detach from computation graph."""
        return self._clone_with_data(self.data.detach())

    def requires_grad_(self, requires_grad: bool = True) -> "UnevenTensor":
        """Set requires_grad on the packed data tensor."""
        self.data.requires_grad_(requires_grad)
        return self

    def float(self) -> "UnevenTensor":
        """Convert to float32."""
        return self._clone_with_data(self.data.float())


    def __add__(
        self, other: Union[float, int, torch.Tensor, "UnevenTensor"]
    ) -> "UnevenTensor":
        if isinstance(other, UnevenTensor):
            return self._clone_with_data(self.data + other.data)
        return self._clone_with_data(self.data + other)

    def __radd__(self, other: Union[float, int, torch.Tensor]) -> "UnevenTensor":
        return self.__add__(other)

    def __sub__(
        self, other: Union[float, int, torch.Tensor, "UnevenTensor"]
    ) -> "UnevenTensor":
        if isinstance(other, UnevenTensor):
            return self._clone_with_data(self.data - other.data)
        return self._clone_with_data(self.data - other)

    def __rsub__(self, other: Union[float, int, torch.Tensor]) -> "UnevenTensor":
        return self._clone_with_data(other - self.data)

    def __mul__(
        self, other: Union[float, int, torch.Tensor, "UnevenTensor"]
    ) -> "UnevenTensor":
        if isinstance(other, UnevenTensor):
            return self._clone_with_data(self.data * other.data)
        return self._clone_with_data(self.data * other)

    def __rmul__(self, other: Union[float, int, torch.Tensor]) -> "UnevenTensor":
        return self.__mul__(other)

    def __truediv__(
        self, other: Union[float, int, torch.Tensor, "UnevenTensor"]
    ) -> "UnevenTensor":
        if isinstance(other, UnevenTensor):
            return self._clone_with_data(self.data / other.data)
        return self._clone_with_data(self.data / other)

    def __rtruediv__(self, other: Union[float, int, torch.Tensor]) -> "UnevenTensor":
        return self._clone_with_data(other / self.data)

    def __neg__(self) -> "UnevenTensor":
        return self._clone_with_data(-self.data)

    def __iadd__(
        self, other: Union[float, int, torch.Tensor, "UnevenTensor"]
    ) -> "UnevenTensor":
        if isinstance(other, UnevenTensor):
            self.data += other.data
        else:
            self.data += other
        return self

    def __isub__(
        self, other: Union[float, int, torch.Tensor, "UnevenTensor"]
    ) -> "UnevenTensor":
        if isinstance(other, UnevenTensor):
            self.data -= other.data
        else:
            self.data -= other
        return self

    def __imul__(
        self, other: Union[float, int, torch.Tensor, "UnevenTensor"]
    ) -> "UnevenTensor":
        if isinstance(other, UnevenTensor):
            self.data *= other.data
        else:
            self.data *= other
        return self

    def __itruediv__(
        self, other: Union[float, int, torch.Tensor, "UnevenTensor"]
    ) -> "UnevenTensor":
        if isinstance(other, UnevenTensor):
            self.data /= other.data
        else:
            self.data /= other
        return self


def compute_cu_seqlens(seq_lens: List[int], device: torch.device) -> torch.Tensor:
    """Compute cumulative sequence lengths for flash_attn_varlen.

    Args:
        seq_lens: List of sequence lengths.
        device: Device for output tensor.

    Returns:
        cu_seqlens tensor of shape (len(seq_lens) + 1,) with dtype int32.
    """
    cu_seqlens = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.cumsum(
        torch.tensor(seq_lens, dtype=torch.int32, device=device), dim=0
    )
    return cu_seqlens


def patch_deepspeed_for_uneven_tensor():
    """Patch DeepSpeed's apply_to_tensors_only to handle UnevenTensor.

    DeepSpeed's ZeRO-3 hooks use apply_to_tensors_only to detect tensors in
    module inputs/outputs for device placement and gradient tracking. Without
    this patch, UnevenTensor is opaque to DeepSpeed, causing missed hooks and
    warnings.
    """
    try:
        import deepspeed.runtime.zero.utils as ds_zero_utils
    except ImportError:
        return

    _original = ds_zero_utils.apply_to_tensors_only

    def _patched(function, value, warning_msg_fn=None):
        if isinstance(value, UnevenTensor):
            new_data = _original(function, value.data, warning_msg_fn)
            new_cu = _original(function, value._cu_seqlens, warning_msg_fn)
            result = value._clone_with_data(new_data)
            result._cu_seqlens = new_cu
            return result
        return _original(function, value, warning_msg_fn)

    ds_zero_utils.apply_to_tensors_only = _patched
