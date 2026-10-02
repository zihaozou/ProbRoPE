"""Dfuse."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, List, Optional, Literal
from einops import rearrange

from ..core.uneven_tensor import UnevenTensor, PackedSequence, compute_cu_seqlens


try:
    import flash_attn_interface

    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn

    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn, sageattn_varlen

    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False

try:
    from torch.nn.attention.varlen import varlen_attn

    TORCH_VARLEN_ATTN_AVAILABLE = True
except ImportError:
    TORCH_VARLEN_ATTN_AVAILABLE = False


def _sdpa_varlen_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    num_heads: int,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Per-sequence SDPA loop fallback (CPU or when no varlen kernel is available)."""
    batch_size = len(cu_seqlens_q) - 1
    outputs = []
    for i in range(batch_size):
        q_s, q_e = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
        k_s, k_e = cu_seqlens_k[i].item(), cu_seqlens_k[i + 1].item()
        q_i = rearrange(q[q_s:q_e].unsqueeze(0), "b s (n d) -> b n s d", n=num_heads)
        k_i = rearrange(k[k_s:k_e].unsqueeze(0), "b s (n d) -> b n s d", n=num_heads)
        v_i = rearrange(v[k_s:k_e].unsqueeze(0), "b s (n d) -> b n s d", n=num_heads)
        out_i = F.scaled_dot_product_attention(q_i, k_i, v_i, dropout_p=dropout_p)
        out_i = rearrange(out_i, "b n s d -> b s (n d)", n=num_heads)
        outputs.append(out_i.squeeze(0))
    return torch.cat(outputs, dim=0)


def _dispatch_flash3(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    num_heads,
    dropout_p,
):
    q = rearrange(q, "s (n d) -> s n d", n=num_heads)
    k = rearrange(k, "s (n d) -> s n d", n=num_heads)
    v = rearrange(v, "s (n d) -> s n d", n=num_heads)
    x = flash_attn_interface.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=dropout_p,
    )
    if isinstance(x, tuple):
        x = x[0]
    return rearrange(x, "s n d -> s (n d)", n=num_heads)


def _dispatch_flash2(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    num_heads,
    dropout_p,
):
    q = rearrange(q, "s (n d) -> s n d", n=num_heads)
    k = rearrange(k, "s (n d) -> s n d", n=num_heads)
    v = rearrange(v, "s (n d) -> s n d", n=num_heads)
    x = flash_attn.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=dropout_p,
    )
    return rearrange(x, "s n d -> s (n d)", n=num_heads)


def _dispatch_sage(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    num_heads,
    dropout_p,
):
    q = rearrange(q, "s (n d) -> s n d", n=num_heads)
    k = rearrange(k, "s (n d) -> s n d", n=num_heads)
    v = rearrange(v, "s (n d) -> s n d", n=num_heads)
    x = sageattn_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
    return rearrange(x, "s n d -> s (n d)", n=num_heads)


def _dispatch_torch_varlen(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    num_heads,
    dropout_p,
):
    q = rearrange(q, "s (n d) -> s n d", n=num_heads)
    k = rearrange(k, "s (n d) -> s n d", n=num_heads)
    v = rearrange(v, "s (n d) -> s n d", n=num_heads)
    x = varlen_attn(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
    return rearrange(x, "s n d -> s (n d)", n=num_heads)


def _dispatch_torch_sdpa(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    num_heads,
    dropout_p,
):
    return _sdpa_varlen_fallback(
        q, k, v, cu_seqlens_q, cu_seqlens_k, num_heads, dropout_p
    )


_ATTN_BACKENDS = {
    "flash_attention_3": (_dispatch_flash3, FLASH_ATTN_3_AVAILABLE),
    "flash_attention_2": (_dispatch_flash2, FLASH_ATTN_2_AVAILABLE),
    "sage_attention": (_dispatch_sage, SAGE_ATTN_AVAILABLE),
    "torch_varlen": (_dispatch_torch_varlen, TORCH_VARLEN_ATTN_AVAILABLE),
    "torch": (None, True),
}


def flash_attention_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_heads: int,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """
    Variable-length attention for packed sequences.

    Respects the ``DIFFSYNTH_ATTENTION_IMPLEMENTATION`` env var:
    - Not set: auto-detect (flash3 > flash2 > sage > torch_varlen > SDPA loop)
    - ``flash_attention_3`` / ``flash_attention_2`` / ``sage_attention`` /
      ``torch_varlen``: use that backend (must be installed)
    - ``torch``: try ``torch.nn.attention.varlen.varlen_attn`` first,
      fall back to per-sequence SDPA loop

    Args:
        q: Query tensor (total_q, dim) where dim = num_heads * head_dim
        k: Key tensor (total_k, dim)
        v: Value tensor (total_k, dim)
        cu_seqlens_q: Cumulative query sequence lengths (batch_size + 1,), int32
        cu_seqlens_k: Cumulative key sequence lengths (batch_size + 1,), int32
        max_seqlen_q: Maximum query sequence length
        max_seqlen_k: Maximum key sequence length
        num_heads: Number of attention heads
        dropout_p: Attention dropout probability (default 0.0)

    Returns:
        Output tensor (total_q, dim)
    """
    import os

    impl = os.environ.get("DIFFSYNTH_ATTENTION_IMPLEMENTATION", "").lower() or None
    use_cuda = q.is_cuda
    args = (
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        num_heads,
        dropout_p,
    )

    if impl is not None:

        if impl == "torch":

            if use_cuda and TORCH_VARLEN_ATTN_AVAILABLE:
                return _dispatch_torch_varlen(*args)
            return _sdpa_varlen_fallback(
                q, k, v, cu_seqlens_q, cu_seqlens_k, num_heads, dropout_p
            )

        if impl in _ATTN_BACKENDS:
            fn, available = _ATTN_BACKENDS[impl]
            if not available:
                raise RuntimeError(
                    f"DIFFSYNTH_ATTENTION_IMPLEMENTATION={impl!r} requested but "
                    f"the required package is not installed."
                )
            if not use_cuda:
                raise RuntimeError(
                    f"DIFFSYNTH_ATTENTION_IMPLEMENTATION={impl!r} requires CUDA tensors."
                )
            return fn(*args)

        raise ValueError(
            f"Unknown DIFFSYNTH_ATTENTION_IMPLEMENTATION={impl!r}. "
            f"Valid options: {', '.join(_ATTN_BACKENDS.keys())}"
        )


    if use_cuda and FLASH_ATTN_3_AVAILABLE:
        return _dispatch_flash3(*args)
    elif use_cuda and FLASH_ATTN_2_AVAILABLE:
        return _dispatch_flash2(*args)
    elif use_cuda and SAGE_ATTN_AVAILABLE:
        return _dispatch_sage(*args)
    elif use_cuda and TORCH_VARLEN_ATTN_AVAILABLE:
        return _dispatch_torch_varlen(*args)
    else:
        return _sdpa_varlen_fallback(
            q, k, v, cu_seqlens_q, cu_seqlens_k, num_heads, dropout_p
        )


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    """Adaptive layer norm modulation."""
    return x * (1 + scale) + shift


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor):
    """Sinusoidal positional embedding for timestep."""
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(
            10000,
            -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(
                dim // 2
            ),
        ),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    """Precompute RoPE frequencies for 1D positions."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    """Precompute 3D RoPE frequencies for (frame, height, width)."""
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def rope_freqs_direct(
    positions: torch.Tensor, dim: int, theta: float = 10000.0
) -> torch.Tensor:
    """Compute 1D RoPE complex frequencies for arbitrary (possibly fractional) positions.

    Args:
        positions: (N,) float tensor of position values (may be non-integer)
        dim: RoPE dimension
        theta: RoPE base frequency

    Returns:
        freqs_cis: (N, dim//2) complex tensor
    """
    half = dim // 2
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(0, dim, 2, device=positions.device, dtype=torch.float64)[:half]
            / dim
        )
    )
    angles = positions.double().unsqueeze(1) * inv_freq.unsqueeze(0)
    return torch.polar(
        torch.ones_like(angles), angles
    )


def rope_apply_varlen(x: torch.Tensor, freqs: torch.Tensor, num_heads: int):
    """
    Apply rotary position embedding to packed (variable-length) sequences.

    Args:
        x: Packed input tensor (total_tokens, dim) where dim = num_heads * head_dim
        freqs: RoPE frequencies (total_tokens, 1, head_dim//2) complex
        num_heads: Number of attention heads

    Returns:
        Output tensor (total_tokens, dim) with RoPE applied
    """
    x = rearrange(x, "s (n d) -> s n d", n=num_heads)
    x_out = torch.view_as_complex(
        x.to(torch.float64).reshape(x.shape[0], x.shape[1], -1, 2)
    )
    x_out = torch.view_as_real(x_out * freqs).flatten(1)
    return x_out.to(x.dtype)


def expand_per_token(
    per_batch: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """
    Expand a per-batch tensor to per-token for packed sequences.

    Args:
        per_batch: (B, ...) tensor with one entry per batch element
        cu_seqlens: (B+1,) cumulative sequence lengths

    Returns:
        (total_tokens, ...) tensor with values repeated per token
    """
    B = len(cu_seqlens) - 1
    parts = []
    for i in range(B):
        seq_len = cu_seqlens[i + 1].item() - cu_seqlens[i].item()
        parts.append(per_batch[i : i + 1].expand(seq_len, *per_batch.shape[1:]))
    return torch.cat(parts, dim=0)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        x_float = x.float()
        norm = x_float * torch.rsqrt(
            x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return norm.to(dtype) * self.weight


class SelfAttention(nn.Module):
    """Original Wan self-attention adapted for packed (variable-length) sequences.

    Single set of q/k/v/o projections with RMSNorm on q/k and standard RoPE.
    No MoE routing, no modulation, no camera projection.
    """

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        """
        Args:
            x: (total_tokens, dim) — packed target-only tokens.
            freqs: (total_tokens, 1, head_dim//2) complex — 3D RoPE frequencies.
            cu_seqlens: (B+1,) int32 — per-sample cumulative target token lengths.
            max_seqlen: Max per-sample target token count.
            dropout_p: Attention dropout probability.

        Returns:
            (total_tokens, dim) — attention output.
        """
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply_varlen(q, freqs, self.num_heads)
        k = rope_apply_varlen(k, freqs, self.num_heads)
        out = flash_attention_varlen(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            self.num_heads,
            dropout_p=dropout_p,
        )
        return self.o(out)


class ProbRoPE(nn.Module):
    """Multi-view fusion attention with per-modality MoE routing.

    Plain 3D RoPE (frame, height, width) on Q/K, per-modality Q/K/V/O
    projections with RMSNorm on q/k.  Camera conditioning is handled outside
    this module by the block-level cam_encoder + projector residual pair —
    ProbRoPE itself is camera-agnostic.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_modalities: int = 1,
        eps: float = 1e-6,
        modulation_type: Optional[Literal["sinc", "gaussian", "learned"]] = None,
        enable_position_phase: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_modalities = num_modalities
        self.modulation_type = modulation_type
        self.enable_position_phase = enable_position_phase


        self.q_list = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in range(num_modalities)]
        )
        self.k_list = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in range(num_modalities)]
        )
        self.v_list = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in range(num_modalities)]
        )
        self.o_list = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in range(num_modalities)]
        )
        self.norm_q_list = nn.ModuleList(
            [RMSNorm(dim, eps=eps) for _ in range(num_modalities)]
        )
        self.norm_k_list = nn.ModuleList(
            [RMSNorm(dim, eps=eps) for _ in range(num_modalities)]
        )


        head_dim = self.head_dim
        self.rope_head_dim = head_dim
        f_dim = head_dim - 2 * (head_dim // 3)
        self.f_dim_half = f_dim // 2
        theta = 10000.0

        if modulation_type is not None:

            inv_freq = 1.0 / (
                theta
                ** (
                    torch.arange(0, f_dim, 2, dtype=torch.float64)[: self.f_dim_half]
                    / f_dim
                )
            )
            self.register_buffer("rope_theta_d", inv_freq)

        if modulation_type == "gaussian":
            self.pos_sigma_mlp = nn.Sequential(
                nn.Linear(dim, dim),
                nn.SiLU(),
                nn.Linear(dim, num_heads * self.f_dim_half * 2),
            )

        if modulation_type in ("sinc", "gaussian") and enable_position_phase:
            self.pos_phase_mlp = nn.Sequential(
                nn.Linear(dim, dim),
                nn.SiLU(),
                nn.Linear(dim, num_heads * self.f_dim_half * 2),
            )

        if modulation_type == "learned":
            rope_hd = self.rope_head_dim
            rope_dim = num_heads * rope_hd
            self.qk_V_log = nn.Parameter(torch.zeros(num_heads, rope_hd, rope_hd))
            for prefix in ("q", "k"):
                setattr(
                    self,
                    f"{prefix}_singular_raw",
                    nn.Parameter(
                        torch.full((num_heads, self.f_dim_half), math.log(math.e - 1))
                    ),
                )
                setattr(
                    self,
                    f"{prefix}_feat_mlp",
                    nn.Sequential(
                        nn.Linear(rope_dim + 1, dim),
                        nn.SiLU(),
                        nn.Linear(dim, num_heads * self.f_dim_half * 2),
                    ),
                )


    def _apply_position_modulation(
        self,
        x_in: torch.Tensor,
        freqs: torch.Tensor,
        cond: torch.Tensor,
        fps_ratio: torch.Tensor,
        prefix: str,
    ) -> torch.Tensor:
        """Compute modulator, multiply into temporal RoPE freqs, apply RoPE.

        All modes ("sinc", "gaussian", "learned") only modulate temporal
        frequencies.  "sinc" / "gaussian" work in the standard basis;
        "learned" works in an orthogonal V basis but still restricts
        modulation to temporal channels (identity for spatial).

        Args:
            x_in:      (total_tokens, rope_dim) — q or k tokens (full head_dim
                       in plucker mode since there's no camera split).
            freqs:     (total_tokens, 1, rope_head_dim//2) complex — RoPE freqs.
            cond:      (total_tokens, dim) — conditioning signal (original x).
            fps_ratio: (total_tokens,) — fps_video / fps_target per token.
            prefix:    "q" or "k" — selects parameter set.

        Returns:
            (total_tokens, rope_dim) — tokens with modulated RoPE applied.
        """
        total_tokens = x_in.shape[0]
        half_d = self.rope_head_dim // 2
        delta_t = 3.0 / (4.0 * fps_ratio)

        if self.modulation_type == "sinc":
            theta_d = self.rope_theta_d
            arg = theta_d.unsqueeze(0) * delta_t.unsqueeze(1) / 2
            amplitude = torch.where(
                arg.abs() < torch.finfo(arg.dtype).eps,
                torch.ones_like(arg),
                torch.sin(arg) / arg,
            )
            amplitude = amplitude.unsqueeze(1)

            if self.enable_position_phase:
                mlp_input = (
                    sinusoidal_embedding_1d(self.dim, fps_ratio).to(cond.dtype) + cond
                )
                phase_raw = self.pos_phase_mlp(mlp_input)
                phase_all = torch.tanh(phase_raw).reshape(
                    total_tokens, self.num_heads, self.f_dim_half * 2
                )
                q_phase, k_phase = phase_all.split(self.f_dim_half, dim=-1)
                phase = q_phase if prefix == "q" else k_phase
                theta_d_phase = theta_d.view(1, 1, self.f_dim_half)
                phase = phase.double() * delta_t.view(-1, 1, 1) / 2 * theta_d_phase
                z = amplitude.double() * torch.polar(torch.ones_like(phase), phase)
            else:
                z = amplitude.to(freqs.dtype)


            mod_freqs = freqs.repeat(
                1, self.num_heads if self.enable_position_phase else 1, 1
            )
            mod_freqs = mod_freqs.clone()
            mod_freqs[:, :, : self.f_dim_half] = freqs[:, :, : self.f_dim_half] * z
            return rope_apply_varlen(x_in, mod_freqs, self.num_heads)

        elif self.modulation_type == "gaussian":
            theta_d = self.rope_theta_d
            mlp_input = (
                sinusoidal_embedding_1d(self.dim, fps_ratio).to(cond.dtype) + cond
            )
            sigma = F.softplus(self.pos_sigma_mlp(mlp_input)).reshape(
                total_tokens, self.num_heads, self.f_dim_half * 2
            )
            q_sigma, k_sigma = sigma.split(self.f_dim_half, dim=-1)
            sigma_prefix = q_sigma if prefix == "q" else k_sigma
            theta_d_exp = theta_d.view(1, 1, self.f_dim_half)
            amplitude = torch.exp(-0.5 * sigma_prefix.double() ** 2 * theta_d_exp**2)

            if self.enable_position_phase:
                mlp_input_ph = (
                    sinusoidal_embedding_1d(self.dim, fps_ratio).to(cond.dtype) + cond
                )
                phase_raw = self.pos_phase_mlp(mlp_input_ph)
                phase_all = torch.tanh(phase_raw).reshape(
                    total_tokens, self.num_heads, self.f_dim_half * 2
                )
                q_phase, k_phase = phase_all.split(self.f_dim_half, dim=-1)
                phase = q_phase if prefix == "q" else k_phase
                theta_d_phase = theta_d.view(1, 1, self.f_dim_half)
                phase = phase.double() * delta_t.view(-1, 1, 1) / 2 * theta_d_phase
                z = amplitude * torch.polar(torch.ones_like(phase), phase)
            else:
                z = amplitude.to(freqs.dtype)

            mod_freqs = freqs.repeat(1, self.num_heads, 1)
            mod_freqs = mod_freqs.clone()
            mod_freqs[:, :, : self.f_dim_half] = freqs[:, :, : self.f_dim_half] * z
            return rope_apply_varlen(x_in, mod_freqs, self.num_heads)

        else:
            V_log = self.qk_V_log.float()
            singular_raw = getattr(self, f"{prefix}_singular_raw")
            feat_mlp = getattr(self, f"{prefix}_feat_mlp")


            V_skew = V_log - V_log.transpose(-1, -2)
            V = torch.matrix_exp(V_skew).to(x_in.dtype)

            sigma = F.softplus(singular_raw)

            x_heads = x_in.view(total_tokens, self.num_heads, self.rope_head_dim)


            v_proj = torch.einsum("hed, the -> thd", V, x_heads)


            v_proj_flat = v_proj.reshape(total_tokens, -1)
            log_fps = (
                torch.log2(fps_ratio).unsqueeze(-1).to(v_proj_flat.dtype)
            )
            mlp_input = torch.cat(
                [v_proj_flat, log_fps], dim=-1
            )
            mlp_out = feat_mlp(mlp_input)

            scale_raw, angles_raw = mlp_out.split(
                [self.num_heads * self.f_dim_half, self.num_heads * self.f_dim_half],
                dim=-1,
            )

            scale = (torch.sigmoid(scale_raw) * 2).reshape(
                total_tokens, self.num_heads, self.f_dim_half
            )
            theta_d = self.rope_theta_d.view(1, 1, self.f_dim_half)
            angles = (
                torch.tanh(angles_raw).reshape(
                    total_tokens, self.num_heads, self.f_dim_half
                )
                * (delta_t.view(-1, 1, 1) / 2)
                * theta_d
            )

            amplitude = sigma.unsqueeze(0) * scale
            z = torch.polar(amplitude.double(), angles.double())


            v_complex = torch.view_as_complex(
                v_proj.to(torch.float64).reshape(
                    total_tokens, self.num_heads, half_d, 2
                )
            )

            mod = torch.ones(
                total_tokens,
                self.num_heads,
                half_d,
                dtype=torch.complex128,
                device=x_in.device,
            )
            mod[:, :, : self.f_dim_half] = z

            result_complex = v_complex * mod * freqs
            result = (
                torch.view_as_real(result_complex)
                .reshape(total_tokens, self.num_heads, self.rope_head_dim)
                .to(x_in.dtype)
            )


            result = torch.einsum("hde, the -> thd", V, result)
            return result.reshape(total_tokens, -1)

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        modality_indices: torch.Tensor,
        dropout_p: float = 0.0,
        fps_ratio: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fusion attention with per-modality MoE routing and 3D RoPE.

        Args:
            x:                (total_tokens, dim) — packed tokens (target + source).
            freqs:            (total_tokens, 1, head_dim // 2) complex — 3D RoPE
                              frequencies from _compute_rope_freqs_3d_per_video.
            cu_seqlens:       (B+1,) int32 — per-sample cumulative lengths.
            max_seqlen:       Max per-sample token count.
            modality_indices: (total_tokens,) long — modality id per token.
            dropout_p:        Attention dropout probability.
            fps_ratio:        (total_tokens,) float — fps_video / fps_target per
                              token. Required when ``modulation_type`` is set.

        Returns:
            (total_tokens, dim) — attention output.
        """

        q = torch.empty_like(x)
        k = torch.empty_like(x)
        v = torch.empty_like(x)
        for m in range(self.num_modalities):
            indices_m = (modality_indices == m).nonzero(as_tuple=True)[0]
            x_m = x[indices_m]
            q[indices_m] = self.norm_q_list[m](self.q_list[m](x_m))
            k[indices_m] = self.norm_k_list[m](self.k_list[m](x_m))
            v[indices_m] = self.v_list[m](x_m)


        if self.modulation_type is not None and fps_ratio is not None:
            q = self._apply_position_modulation(q, freqs, x, fps_ratio, "q")
            k = self._apply_position_modulation(k, freqs, x, fps_ratio, "k")
        else:
            q = rope_apply_varlen(q, freqs, self.num_heads)
            k = rope_apply_varlen(k, freqs, self.num_heads)


        out = flash_attention_varlen(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            self.num_heads,
            dropout_p=dropout_p,
        )


        result = torch.empty_like(out)
        for m in range(self.num_modalities):
            indices_m = (modality_indices == m).nonzero(as_tuple=True)[0]
            result[indices_m] = self.o_list[m](out[indices_m])

        return result


class FFN(nn.Module):
    """Modality-specific feed-forward network.

    Each modality has its own MLP expert.  Tokens are routed to the
    appropriate expert using a per-token modality indicator
    (MoE-style gather/scatter), mirroring the routing in SelfAttention.
    """

    def __init__(self, dim: int, ffn_dim: int, num_modalities: int = 1):
        super().__init__()
        self.num_modalities = num_modalities
        self.ffn_list = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, ffn_dim),
                    nn.GELU(approximate="tanh"),
                    nn.Linear(ffn_dim, dim),
                )
                for _ in range(num_modalities)
            ]
        )

    def forward(self, x: torch.Tensor, modality_indices: torch.Tensor) -> torch.Tensor:
        """Route tokens to modality-specific FFN experts.

        Args:
            x: (total_tokens, dim) packed input tokens.
            modality_indices: (total_tokens,) long — modality id per token.

        Returns:
            (total_tokens, dim) — FFN output with per-modality routing.
        """
        result = torch.empty_like(x)
        for m in range(self.num_modalities):
            indices_m = (modality_indices == m).nonzero(as_tuple=True)[0]
            result[indices_m] = self.ffn_list[m](x[indices_m])
        return result


class CrossAttention(nn.Module):
    """Cross-attention for text conditioning with packed sequences.

    Simple varlen cross-attention: no camera embed or RoPE on K side.
    Text-only context (no spatial context).
    """

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_kv: int,
        dropout_p: float = 0.0,
    ):
        """
        Args:
            x: Packed query tokens (total_q, dim)
            context: Packed context tokens (total_kv, dim)
            cu_seqlens_q: Cumulative query lengths (B+1,) int32
            cu_seqlens_kv: Cumulative context lengths (B+1,) int32
            max_seqlen_q: Maximum query sequence length
            max_seqlen_kv: Maximum context sequence length
            dropout_p: Attention dropout probability (default 0.0)
        """
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(context))
        v = self.v(context)
        x = flash_attention_varlen(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
            self.num_heads,
            dropout_p=dropout_p,
        )
        return self.o(x)


class CausalConv3DCamEncoder(nn.Module):
    """Causal 3D convolution encoder for Plücker rays.

    Compresses temporally-uncompressed Plücker rays (6 channels) into
    per-latent-frame camera embeddings using a 1D causal convolution along
    the temporal axis.  Spatial dimensions are preserved (kernel 1×1 in H, W).

    Input:  (1, 6, T_video, H_patch, W_patch)  — raw Plücker rays
    Output: (1, dim, T_lat, H_patch, W_patch)  — camera embeddings
    where T_lat = ceil(T_video / vae_temporal_patch_size)
    """

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 1536,
        vae_temporal_patch_size: int = 4,
    ):
        super().__init__()
        self.vae_temporal_patch_size = vae_temporal_patch_size
        self.temporal_pad = vae_temporal_patch_size - 1
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(vae_temporal_patch_size, 1, 1),
            stride=(vae_temporal_patch_size, 1, 1),
            padding=(0, 0, 0),
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 0, 0, 0, self.temporal_pad, 0))
        return self.conv(x)


class DiTBlock(nn.Module):
    """Diffusion Transformer block with sequential self-attention + fusion attention.

    Self-attention processes target-only tokens with integer-indexed 3D RoPE.
    ProbRoPE is camera-agnostic (plain 3D RoPE with per-video reset +
    FPS-aware positions, per-modality MoE routing) and processes all tokens
    (target + source). Camera conditioning is injected around fusion_attn:
    a CausalConv3DCamEncoder encodes Plücker rays into a per-token residual
    added before the attention, and a learned linear projector transforms
    the attention output before the residual gate.

    Fusion has its own modulation (shift/scale/gate) and layer norm.

    All inputs are packed (variable-length) sequences.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        num_modalities: int = 1,
        eps: float = 1e-6,
        modulation_type: Optional[Literal["sinc", "gaussian", "learned"]] = None,
        enable_position_phase: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim


        self.self_attn = SelfAttention(dim, num_heads, eps)


        self.fusion_attn = ProbRoPE(
            dim,
            num_heads,
            num_modalities,
            eps,
            modulation_type=modulation_type,
            enable_position_phase=enable_position_phase,
        )

        self.cross_attn = CrossAttention(dim, num_heads, eps)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.norm_fuse = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.ffn = FFN(dim, ffn_dim, num_modalities)
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.modulation_fuse = nn.Parameter(torch.randn(1, 3, dim) / dim**0.5)

        self.cam_encoder = CausalConv3DCamEncoder(
            in_channels=6, out_channels=dim, vae_temporal_patch_size=4
        )
        self.projector = nn.Linear(dim, dim)
        nn.init.zeros_(self.cam_encoder.conv.weight)
        nn.init.zeros_(self.cam_encoder.conv.bias)
        nn.init.eye_(self.projector.weight)
        nn.init.zeros_(self.projector.bias)

    def _encode_cam_rays(
        self,
        target_cam_rays_list: List[torch.Tensor],
        source_cam_rays_list: List[torch.Tensor],
        source_view_T_list: List[List[int]],
        source_lat_T_list: List[List[int]],
        seq_lens: List[int],
    ) -> torch.Tensor:
        """Encode Plücker rays per sample and pack into (total_tokens, dim).

        Source rays are encoded **per view** independently to avoid temporal
        boundary artifacts with CausalConv3D.

        ``source_view_T_list`` always carries the original video frame counts
        (even for dropped views) so that cam_encoder is called the same number
        of times on every rank.  Dropped views (lat_T_j == 0) use batch_dim=0
        input to cam_encoder: the Conv3d processes 0 elements (no kernel
        constraint issue), produces an empty output that naturally flows
        through the autograd graph, and contributes 0 tokens to the result.

        Args:
            target_cam_rays_list: List of (T_video_tgt_i, 6, H, W) per sample
            source_cam_rays_list: List of (T_video_src_total_i, 6, H, W) per sample
            source_view_T_list:   List of per-view raw video frame counts per sample
                                  (always the original count, even for dropped views)
            source_lat_T_list:    List of per-view latent frame counts per sample
                                  (0 for dropped views)
            seq_lens: List of expected token counts (f_i * h * w) per sample

        Returns:
            (total_tokens, dim) packed camera embeddings
        """
        B = len(target_cam_rays_list)
        parts = []
        for i in range(B):

            tgt_rays = target_cam_rays_list[i].unsqueeze(0).permute(0, 2, 1, 3, 4)
            tgt_encoded = self.cam_encoder(tgt_rays)


            src_encoded_parts = []
            offset = 0
            src_T_list_i = source_view_T_list[i] if source_view_T_list is not None else []
            src_rays_i = source_cam_rays_list[i] if source_cam_rays_list is not None else None
            lat_T_list_i = source_lat_T_list[i] if source_lat_T_list is not None else []
            for j, T_j in enumerate(src_T_list_i):
                lat_T_j = lat_T_list_i[j] if j < len(lat_T_list_i) else 0
                rays_j = src_rays_i[offset: offset + T_j]
                if T_j == 2:
                    frame_parts = []
                    for fi in range(2):
                        f_ray = rays_j[fi: fi + 1].unsqueeze(0).permute(0, 2, 1, 3, 4)
                        encoded = self.cam_encoder(f_ray)
                        if lat_T_j == 0:


                            encoded = encoded[:, :, :0, :, :]
                        frame_parts.append(encoded)
                    src_encoded_parts.append(torch.cat(frame_parts, dim=2))
                else:
                    rays_5d = rays_j.unsqueeze(0).permute(0, 2, 1, 3, 4)
                    encoded = self.cam_encoder(rays_5d)
                    if lat_T_j == 0:
                        encoded = encoded[:, :, :0, :, :]
                    src_encoded_parts.append(encoded)
                offset += T_j

            if src_encoded_parts:
                src_encoded = torch.cat(src_encoded_parts, dim=2)
                cam_encoded = torch.cat([tgt_encoded, src_encoded], dim=2)
            else:
                cam_encoded = tgt_encoded
            cam_tokens = rearrange(cam_encoded, "b c f h w -> b (f h w) c").squeeze(0)

            assert cam_tokens.shape[0] == seq_lens[i], (
                f"Sample {i}: cam_tokens {cam_tokens.shape[0]} != seq_lens {seq_lens[i]}"
            )
            parts.append(cam_tokens)

        return torch.cat(parts, dim=0)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        t_mod: torch.Tensor,
        freqs_3d_target: torch.Tensor,
        freqs_3d_all: torch.Tensor,
        cu_seqlens_x: torch.Tensor,
        cu_seqlens_ctx: torch.Tensor,
        max_seqlen_x: int,
        max_seqlen_ctx: int,
        cu_seqlens_target: torch.Tensor,
        max_seqlen_target: int,
        target_mask: torch.Tensor,
        modality_indices: torch.Tensor,
        target_cam_rays_list: List[torch.Tensor],
        source_cam_rays_list: List[torch.Tensor],
        source_view_T_list: List[List[int]],
        source_lat_T_list: List[List[int]],
        seq_lens: List[int],
        cam_keep_mask: Optional[torch.Tensor],
        fps_ratio: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x: Packed tokens (total_tokens, dim) — all tokens (target + source).
            context: Packed text tokens (total_ctx, dim).
            t_mod: Per-token timestep modulation (total_tokens, 6, dim).
            freqs_3d_target: 3D RoPE for self_attn (integer frame indices) —
                (total_target_tokens, 1, head_dim // 2) complex.
            freqs_3d_all: 3D RoPE for fusion_attn (per-video reset, FPS-aware) —
                (total_tokens, 1, head_dim // 2) complex.
            cu_seqlens_x: Per-sample total token lengths (B+1,) int32.
            cu_seqlens_ctx: Cumulative text lengths (B+1,) int32.
            max_seqlen_x: Max per-sample total token count.
            max_seqlen_ctx: Max text sequence length.
            cu_seqlens_target: Per-sample target token lengths (B+1,) int32.
            max_seqlen_target: Max per-sample target token count.
            target_mask: (total_tokens,) bool — True for target tokens.
            modality_indices: (total_tokens,) long — modality id per token.
            target_cam_rays_list: List of (T_video_tgt_i, 6, H, W) per sample.
            source_cam_rays_list: List of (T_video_src_total_i, 6, H, W) per sample.
            source_view_T_list: Per-view raw video frame counts per sample
                (always original count, even for dropped views).
            source_lat_T_list: Per-view latent frame counts per sample
                (0 for dropped views).
            seq_lens: Pre-dropout per-sample token counts (f_total_i * h * w).
            cam_keep_mask: (total_tokens_pre_dropout,) bool — source-dropout
                survival mask, or None when source_dropout == 0.
            fps_ratio: (total_tokens,) float — fps_video / fps_target per
                token. Required only when ``fusion_attn`` was configured
                with ``modulation_type`` set.
        """

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=1)
        shift_msa = shift_msa.squeeze(1)
        scale_msa = scale_msa.squeeze(1)
        gate_msa = gate_msa.squeeze(1)
        shift_mlp = shift_mlp.squeeze(1)
        scale_mlp = scale_mlp.squeeze(1)
        gate_mlp = gate_mlp.squeeze(1)


        shift_fuse, scale_fuse, gate_fuse = (
            self.modulation_fuse.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod[:, :3]
        ).chunk(3, dim=1)
        shift_fuse = shift_fuse.squeeze(1)
        scale_fuse = scale_fuse.squeeze(1)
        gate_fuse = gate_fuse.squeeze(1)


        x_norm_target = modulate(
            self.norm1(x[target_mask]), shift_msa[target_mask], scale_msa[target_mask]
        )
        self_out = self.self_attn(
            x_norm_target, freqs_3d_target,
            cu_seqlens_target, max_seqlen_target,
        )
        self_residual = torch.zeros_like(x)
        self_residual[target_mask] = gate_msa[target_mask] * self_out
        x = x + self_residual


        x_fuse = modulate(self.norm_fuse(x), shift_fuse, scale_fuse)

        cam_broadcast = self._encode_cam_rays(
            target_cam_rays_list,
            source_cam_rays_list,
            source_view_T_list,
            source_lat_T_list,
            seq_lens,
        )
        if cam_keep_mask is not None:
            cam_broadcast = cam_broadcast[cam_keep_mask]
        x_fuse = x_fuse + cam_broadcast

        fusion_out = self.fusion_attn(
            x_fuse, freqs_3d_all, cu_seqlens_x, max_seqlen_x,
            modality_indices,
            fps_ratio=fps_ratio,
        )

        fusion_out = self.projector(fusion_out)
        x = x + gate_fuse * fusion_out


        x = x + self.cross_attn(
            self.norm3(x),
            context,
            cu_seqlens_x,
            cu_seqlens_ctx,
            max_seqlen_x,
            max_seqlen_ctx,
        )


        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.ffn(x_norm, modality_indices)

        return x


class Head(nn.Module):
    """Output head with adaptive layer norm for packed sequences."""

    def __init__(
        self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float
    ):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ):
        """
        Args:
            x: Packed tokens (total_tokens, dim)
            t: Per-token timestep embedding (total_tokens, dim)

        Returns:
            (total_tokens, out_dim * prod(patch_size))
        """

        mod = self.modulation.to(dtype=t.dtype, device=t.device) + t.unsqueeze(1)
        shift, scale = mod.chunk(2, dim=1)
        shift = shift.squeeze(1)
        scale = scale.squeeze(1)

        x = self.head(self.norm(x) * (1 + scale) + shift)
        return x


class DFuse(torch.nn.Module):
    """
    D-FUSE DFuse Wan Video DiT with parallel self-attention + fusion attention.

    Self-attention is the original Wan architecture (target-only, integer 3D RoPE).
    ProbRoPE is camera-agnostic (plain 3D RoPE with per-video reset +
    FPS-aware positions, per-modality MoE routing) and processes all tokens.
    Camera conditioning is injected per-block via a CausalConv3DCamEncoder
    that encodes raw Plücker rays into a residual added before fusion_attn,
    followed by a learned linear projector on the fusion_attn output.

    Takes concatenated [target, source] latents along temporal dimension and
    raw Plücker rays describing camera pose per video frame for both target
    and each source view.

    All inputs are UnevenTensor (list of per-sample tensors with variable T).

    Args:
        dim: Hidden dimension (1536 for 1.3B)
        in_dim: Input latent channels (16 for Wan VAE)
        ffn_dim: FFN intermediate dimension (8960 for 1.3B)
        out_dim: Output latent channels (16)
        text_dim: Text encoder dimension (4096 for T5)
        freq_dim: Timestep embedding dimension (256)
        eps: Layer norm epsilon
        patch_size: Patch size for video (t, h, w) = (1, 2, 2)
        num_heads: Number of attention heads (12 for 1.3B)
        num_layers: Number of transformer blocks (30 for 1.3B)
    """

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, *, revision=None, cache_dir=None,
        token=None, local_files_only=False, torch_dtype=torch.bfloat16,
        device="cpu",
    ):
        """Load config and complete weights from a Hub repository or local folder."""
        import json
        from pathlib import Path
        from huggingface_hub import snapshot_download
        from ..core.loader import load_model

        folder = Path(pretrained_model_name_or_path)
        if not folder.is_dir():
            folder = Path(snapshot_download(
                repo_id=str(pretrained_model_name_or_path), revision=revision,
                cache_dir=cache_dir, token=token, local_files_only=local_files_only,
                allow_patterns=["config.json", "dfuse.safetensors"],
            ))
        config = json.loads((folder / "config.json").read_text())
        if config.get("model_type") != "dfuse":
            raise ValueError("Expected a D-FUSE model configuration.")
        return load_model(
            cls, str(folder / "dfuse.safetensors"), config["model_config"],
            torch_dtype=torch_dtype, device=device, strict=True,
        )

    def __init__(
        self,
        dim: int = 1536,
        in_dim: int = 16,
        ffn_dim: int = 8960,
        out_dim: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        eps: float = 1e-6,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        num_heads: int = 12,
        num_layers: int = 30,
        num_modalities: int = 1,
        modulation_type: Optional[Literal["sinc", "gaussian", "learned"]] = None,
        enable_position_phase: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.freq_dim = freq_dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.cam_embed_mode = "plucker"
        self.num_modalities = num_modalities


        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )


        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )


        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))


        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim,
                    num_heads,
                    ffn_dim,
                    num_modalities,
                    eps,
                    modulation_type=modulation_type,
                    enable_position_phase=enable_position_phase,
                )
                for _ in range(num_layers)
            ]
        )


        self.head = Head(dim, out_dim, patch_size, eps)


        head_dim = dim // num_heads
        self.freqs_3d = precompute_freqs_cis_3d(head_dim)


    def _patchify_single(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """Patchify a single video tensor.

        Args:
            x: (C, T, H, W) single video latent

        Returns:
            tokens: (N, dim) where N = f * h * w
            grid_size: (f, h, w) after patchification
        """
        x = x.unsqueeze(0)
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c")
        return x.squeeze(0), tuple(grid_size)

    def _patchify_batch(
        self, x: UnevenTensor
    ) -> Tuple[List[torch.Tensor], List[Tuple[int, int, int]]]:
        """Patchify all samples in a batch.

        Args:
            x: UnevenTensor of (C, T_i, H, W) per sample

        Returns:
            all_tokens: List of (N_i, dim) tensors
            grid_sizes: List of (f_i, h, w) tuples
        """
        all_tokens = []
        grid_sizes = []
        for i in range(x.batch_size):
            tokens_i, grid_i = self._patchify_single(x[i])
            all_tokens.append(tokens_i)
            grid_sizes.append(grid_i)
        return all_tokens, grid_sizes


    def _unpatchify_single(
        self, x: torch.Tensor, grid_size: Tuple[int, int, int]
    ) -> torch.Tensor:
        """Unpatchify a single sample's tokens back to video.

        Args:
            x: (N, out_dim * prod(patch_size)) tokens
            grid_size: (f, h, w)

        Returns:
            (C, T, H, W) video tensor
        """
        f, h, w = grid_size
        x = x.unsqueeze(0)
        x = rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=f,
            h=h,
            w=w,
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )
        return x.squeeze(0)


    def _compute_rope_freqs_3d_target(
        self,
        target_grid_sizes: List[Tuple[int, int, int]],
        device: torch.device,
        target_frame_offsets: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Compute standard 3D RoPE frequencies for target-only self-attention.

        Standard integer-indexed (frame, height, width) with the original Wan
        unequal dimension split.

        Args:
            target_grid_sizes: List of (f_tgt_i, h, w) per sample.
            device: Target device.
            target_frame_offsets: Optional per-sample integer offset added to
                the frame index. Used for FIFO-style streaming where the
                queue holds latents at non-zero global frame positions.
                Defaults to all zeros (no offset, current behavior).

        Returns:
            freqs: (total_target_tokens, 1, head_dim//2) complex tensor
        """
        f_table, h_table, w_table = self.freqs_3d

        all_freqs = []
        for i, (f, h, w) in enumerate(target_grid_sizes):
            offset = target_frame_offsets[i] if target_frame_offsets is not None else 0
            f_cis = f_table[offset : offset + f].to(device).view(f, 1, 1, -1).expand(f, h, w, -1)
            h_cis = h_table[:h].to(device).view(1, h, 1, -1).expand(f, h, w, -1)
            w_cis = w_table[:w].to(device).view(1, 1, w, -1).expand(f, h, w, -1)
            freqs_i = torch.cat([f_cis, h_cis, w_cis], dim=-1)
            freqs_i = freqs_i.reshape(f * h * w, 1, -1)
            all_freqs.append(freqs_i)

        return torch.cat(all_freqs, dim=0)

    def _compute_rope_freqs_3d_per_video(
        self,
        per_video_grid_sizes: List[List[Tuple[int, int, int]]],
        device: torch.device,
        fps_per_video: Optional[List[List[float]]] = None,
        per_video_frame_offsets: Optional[List[List[int]]] = None,
    ) -> torch.Tensor:
        """Compute 3D RoPE frequencies for fusion attention over all videos.

        Frame indices restart from 0 for each individual video (target and
        each source view).  When ``fps_per_video`` is provided, the temporal
        axis uses FPS-aware continuous positions aligned to the target
        video's timeline:
            pos_j = (4*j - 1.5) * fps_tgt / (4 * fps_v) + 0.375
        where fps_tgt = fps_per_video[i][0] (video 0 is always target).

        Uses the same ``self.freqs_3d`` tables and dimension split as
        ``_compute_rope_freqs_3d_target`` (i.e. f_dim = head_dim - 2*(head_dim//3),
        h_dim = w_dim = head_dim // 3).

        Args:
            per_video_grid_sizes: per_video_grid_sizes[i][v] = (f_v, h, w).
                v=0 is always target; v>=1 are source views.
            device: Target device.
            fps_per_video: Optional per-sample, per-video FPS list. If None,
                falls back to integer frame indices.

        Returns:
            freqs: (total_tokens, 1, head_dim // 2) complex tensor
        """
        f_table, h_table, w_table = self.freqs_3d
        head_dim = self.dim // self.num_heads
        f_dim = head_dim - 2 * (head_dim // 3)

        all_freqs = []
        for si, vgs in enumerate(per_video_grid_sizes):
            fps_tgt = fps_per_video[si][0] if fps_per_video is not None else None
            for vi, (f, h, w) in enumerate(vgs):
                if f == 0:
                    continue
                offset = (
                    per_video_frame_offsets[si][vi]
                    if per_video_frame_offsets is not None
                    else 0
                )
                if fps_per_video is not None:
                    fps_v = fps_per_video[si][vi]
                    token_indices = (
                        torch.arange(f, device=device, dtype=torch.float64) + offset
                    )
                    positions = (4.0 * token_indices - 1.5) * fps_tgt / (
                        4.0 * fps_v
                    ) + 0.375
                    f_cis = rope_freqs_direct(positions, f_dim).to(device)
                    f_cis = f_cis.view(f, 1, 1, -1).expand(f, h, w, -1)
                else:
                    f_cis = (
                        f_table[offset : offset + f]
                        .to(device)
                        .view(f, 1, 1, -1)
                        .expand(f, h, w, -1)
                    )
                h_cis = h_table[:h].to(device).view(1, h, 1, -1).expand(f, h, w, -1)
                w_cis = w_table[:w].to(device).view(1, 1, w, -1).expand(f, h, w, -1)
                freqs_v = torch.cat([f_cis, h_cis, w_cis], dim=-1).reshape(
                    f * h * w, 1, -1
                )
                all_freqs.append(freqs_v)
        return torch.cat(all_freqs, dim=0)


    def _embed_text(
        self, context: UnevenTensor
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Apply text embedding MLP per sample and pack.

        Args:
            context: UnevenTensor of (L_i, text_dim) per sample

        Returns:
            text_packed: (total_text_tokens, dim)
            cu_seqlens_ctx: (B+1,) int32
            max_seqlen_ctx: int
        """
        text_list = []
        for i in range(context.batch_size):
            ctx_i = context[i].unsqueeze(0)
            ctx_i = self.text_embedding(ctx_i).squeeze(0)
            text_list.append(ctx_i)

        text_packed = torch.cat(text_list, dim=0)
        seq_lens = [t.shape[0] for t in text_list]
        cu_seqlens_ctx = compute_cu_seqlens(seq_lens, text_packed.device)
        max_seqlen_ctx = max(seq_lens)
        return text_packed, cu_seqlens_ctx, max_seqlen_ctx


    def _pack_tokens(
        self, all_tokens: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Pack per-sample token lists into a single packed tensor.

        Args:
            all_tokens: List of (N_i, dim) tensors

        Returns:
            x_packed: (total_tokens, dim)
            cu_seqlens: (B+1,) int32
            max_seqlen: int
        """
        x_packed = torch.cat(all_tokens, dim=0)
        seq_lens = [t.shape[0] for t in all_tokens]
        cu_seqlens = compute_cu_seqlens(seq_lens, x_packed.device)
        max_seqlen = max(seq_lens)
        return x_packed, cu_seqlens, max_seqlen


    def forward(
        self,
        x: UnevenTensor,
        source_latents: UnevenTensor,
        timestep: torch.Tensor,
        context: UnevenTensor,
        source_lat_T_list: List[List[int]],
        target_modality: Optional[List[int]] = None,
        source_modality_list: Optional[List[List[int]]] = None,
        fps_target: Optional[List[float]] = None,
        fps_source_list: Optional[List[List[float]]] = None,
        source_timesteps: Optional[List[List[float]]] = None,
        use_gradient_checkpointing: bool = False,
        use_activation_offload: bool = False,
        source_dropout: float = 0.0,

        target_cam_rays: Optional[List[torch.Tensor]] = None,
        source_cam_rays: Optional[List[torch.Tensor]] = None,
        source_view_T_video: Optional[List[List[int]]] = None,
        return_hidden_layers: Optional[List[int]] = None,
        frame_index_offsets: Optional[List[List[int]]] = None,
        target_timesteps_per_frame: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> UnevenTensor:
        """Forward."""
        device = timestep.device
        dtype = x[0].dtype
        B = x.batch_size


        target_lat_T_list = [x[i].shape[1] for i in range(B)]
        x_combined = x.extend(source_latents, dim=1)


        all_tokens, grid_sizes = self._patchify_batch(x_combined)


        if target_modality is None:
            target_modality = [0] * B
        if source_modality_list is None:
            source_modality_list = [[0] * len(source_lat_T_list[i]) for i in range(B)]


        t_batch = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(dtype)
        )


        has_fps = fps_target is not None and fps_source_list is not None
        per_video_grid_sizes: List[List[Tuple[int, int, int]]] = []
        modality_parts: List[torch.Tensor] = []
        target_mask_parts: List[torch.Tensor] = []
        fps_ratio_parts: List[torch.Tensor] = []
        t_parts: List[torch.Tensor] = []

        for i in range(B):
            _, h_i, w_i = grid_sizes[i]
            hw = h_i * w_i
            f_tgt = target_lat_T_list[i]

            vgs_i: List[Tuple[int, int, int]] = [(f_tgt, h_i, w_i)]


            modality_parts.append(
                torch.full(
                    (f_tgt * hw,), target_modality[i], dtype=torch.long, device=device
                )
            )
            target_mask_parts.append(
                torch.ones(f_tgt * hw, dtype=torch.bool, device=device)
            )
            if has_fps:
                fps_ratio_parts.append(torch.ones(f_tgt * hw, device=device))


            if target_timesteps_per_frame is not None:
                tt_i = target_timesteps_per_frame[i]
                if tt_i.shape[0] != f_tgt:
                    raise ValueError(
                        f"target_timesteps_per_frame[{i}] has shape "
                        f"{tt_i.shape}, expected ({f_tgt},)"
                    )
                t_per_frame = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, tt_i.to(device)).to(dtype)
                )
                t_parts.append(
                    t_per_frame.unsqueeze(1).expand(f_tgt, hw, -1).reshape(f_tgt * hw, -1)
                )
            else:
                t_parts.append(t_batch[i].unsqueeze(0).expand(f_tgt * hw, -1))


            for j, T_lat_j in enumerate(source_lat_T_list[i]):
                vgs_i.append((T_lat_j, h_i, w_i))
                m_j = source_modality_list[i][j]
                modality_parts.append(
                    torch.full((T_lat_j * hw,), m_j, dtype=torch.long, device=device)
                )
                target_mask_parts.append(
                    torch.zeros(T_lat_j * hw, dtype=torch.bool, device=device)
                )
                if has_fps:
                    ratio_j = fps_source_list[i][j] / fps_target[i]
                    fps_ratio_parts.append(
                        torch.full((T_lat_j * hw,), ratio_j, device=device)
                    )
                st_j = source_timesteps[i][j] if source_timesteps is not None else 0.0
                st_tensor = torch.tensor([st_j], dtype=dtype, device=device)
                t_src_j = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, st_tensor).to(dtype)
                )
                t_parts.append(t_src_j.expand(T_lat_j * hw, -1))

            per_video_grid_sizes.append(vgs_i)


        modality_indices = torch.cat(modality_parts, dim=0)
        target_mask = torch.cat(target_mask_parts, dim=0)
        fps_ratio_packed: Optional[torch.Tensor] = (
            torch.cat(fps_ratio_parts, dim=0) if has_fps else None
        )
        t = torch.cat(t_parts, dim=0)
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))


        if frame_index_offsets is not None:
            if len(frame_index_offsets) != B:
                raise ValueError(
                    f"frame_index_offsets must have {B} entries, got "
                    f"{len(frame_index_offsets)}"
                )
            for i in range(B):
                expected = 1 + len(source_lat_T_list[i])
                if len(frame_index_offsets[i]) != expected:
                    raise ValueError(
                        f"frame_index_offsets[{i}] must have {expected} entries "
                        f"(1 target + {len(source_lat_T_list[i])} source views), "
                        f"got {len(frame_index_offsets[i])}"
                    )
            target_frame_offsets = [frame_index_offsets[i][0] for i in range(B)]
        else:
            target_frame_offsets = None


        target_grid_sizes = [
            (target_lat_T_list[i], grid_sizes[i][1], grid_sizes[i][2]) for i in range(B)
        ]
        freqs_3d_target = self._compute_rope_freqs_3d_target(
            target_grid_sizes, device, target_frame_offsets=target_frame_offsets
        )


        fps_per_video: Optional[List[List[float]]] = None
        if has_fps:
            fps_per_video = [[fps_target[i]] + fps_source_list[i] for i in range(B)]
        freqs_3d_all = self._compute_rope_freqs_3d_per_video(
            per_video_grid_sizes,
            device,
            fps_per_video=fps_per_video,
            per_video_frame_offsets=frame_index_offsets,
        )


        text_packed, cu_seqlens_ctx, max_seqlen_ctx = self._embed_text(context)


        x_packed, cu_seqlens_x, max_seqlen_x = self._pack_tokens(all_tokens)


        target_seq_lens = [
            target_lat_T_list[i] * grid_sizes[i][1] * grid_sizes[i][2] for i in range(B)
        ]
        cu_seqlens_target = compute_cu_seqlens(target_seq_lens, device)
        max_seqlen_target = max(target_seq_lens)


        pre_dropout_seq_lens = [
            sum(f_v * h_i * w_i for (f_v, h_i, w_i) in per_video_grid_sizes[i])
            for i in range(B)
        ]


        cam_keep_mask: Optional[torch.Tensor] = None
        if source_dropout > 0.0:
            keep_mask = target_mask.clone()
            source_positions = (~target_mask).nonzero(as_tuple=True)[0]
            source_keep = (
                torch.rand(source_positions.shape[0], device=device) >= source_dropout
            )
            keep_mask[source_positions] = source_keep
            cam_keep_mask = keep_mask


            old_cu = cu_seqlens_x
            new_seq_lens = [
                keep_mask[old_cu[i].item() : old_cu[i + 1].item()].sum().item()
                for i in range(B)
            ]


            x_packed = x_packed[keep_mask]
            freqs_3d_all = freqs_3d_all[keep_mask]
            modality_indices = modality_indices[keep_mask]
            target_mask = target_mask[keep_mask]
            t_mod = t_mod[keep_mask]
            t = t[keep_mask]
            if fps_ratio_packed is not None:
                fps_ratio_packed = fps_ratio_packed[keep_mask]

            cu_seqlens_x = compute_cu_seqlens(new_seq_lens, device)
            max_seqlen_x = max(new_seq_lens)


        from ..core.gradient import get_checkpoint_fn

        _ckpt = get_checkpoint_fn(
            self,
            checkpoint=use_gradient_checkpointing,
            offload=use_activation_offload,
        )

        hidden_states = {} if return_hidden_layers is not None else None


        if target_cam_rays is None:
            raise ValueError(
                "D-FUSE requires target_cam_rays (cam_embed_mode='plucker'); "
                "pass raw Plücker rays shaped (T_video_tgt, 6, H_patch, W_patch) per sample"
            )
        if source_view_T_video is None:
            raise ValueError(
                "source_view_T_video must be provided when target_cam_rays is set"
            )

        for block_idx, block in enumerate(self.blocks):
            block_args = (
                x_packed,
                text_packed,
                t_mod,
                freqs_3d_target,
                freqs_3d_all,
                cu_seqlens_x,
                cu_seqlens_ctx,
                max_seqlen_x,
                max_seqlen_ctx,
                cu_seqlens_target,
                max_seqlen_target,
                target_mask,
                modality_indices,
                target_cam_rays,
                source_cam_rays,
                source_view_T_video,
                source_lat_T_list,
                pre_dropout_seq_lens,
                cam_keep_mask,
                fps_ratio_packed,
            )

            def _block_fn(*args, _blk=block):
                return _blk(*args)

            x_packed = _ckpt(_block_fn, *block_args)

            if return_hidden_layers is not None and block_idx in return_hidden_layers:
                target_parts = []
                _offset = 0
                for i in range(B):
                    _, h_i, w_i = grid_sizes[i]
                    hw = h_i * w_i
                    n_target_i = target_lat_T_list[i] * hw
                    n_total_i = (cu_seqlens_x[i + 1] - cu_seqlens_x[i]).item()
                    target_parts.append(x_packed[_offset : _offset + n_target_i])
                    _offset += n_total_i
                hidden_states[block_idx] = torch.cat(target_parts, dim=0)


        target_outputs = []
        offset = 0
        for i in range(B):
            _, h_i, w_i = grid_sizes[i]
            hw = h_i * w_i
            n_target_i = target_lat_T_list[i] * hw
            n_total_i = (cu_seqlens_x[i + 1] - cu_seqlens_x[i]).item()
            target_x_i = x_packed[offset : offset + n_target_i]
            target_t_i = t[offset : offset + n_target_i]
            out_i = self.head(target_x_i, target_t_i)
            target_outputs.append(
                self._unpatchify_single(out_i, (target_lat_T_list[i], h_i, w_i))
            )
            offset += n_total_i
        output = UnevenTensor(target_outputs, channel_dim=0)

        if return_hidden_layers is not None:
            return output, hidden_states
        return output
