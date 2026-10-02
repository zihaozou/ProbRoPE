"""Fvd."""
import math
import os
import urllib.request

import numpy as np
import torch

from .fvd_i3d import InceptionI3d

_I3D_WEIGHTS_URL = (
    "https://huggingface.co/Xiaodong/FVD_I3D/resolve/main/i3d_pretrained_400.pt"
)
_DEFAULT_CACHE = "~/.cache/dfuse/fvd/i3d_pretrained_400.pt"


def _resolve_i3d_weights(weights_path: str | None) -> str:
    """Return a filesystem path to the I3D checkpoint.

    If `weights_path` is None, resolve to the dfuse cache location and
    auto-download from the upstream HuggingFace mirror when missing. The download
    lands in a `.tmp` sibling and is atomically renamed so a partial fetch
    cannot leave a corrupted cache file behind.
    """
    if weights_path is not None:
        expanded = os.path.expanduser(weights_path)
        if not os.path.exists(expanded):
            raise FileNotFoundError(f"I3D weights not found: {expanded}")
        return expanded

    expanded = os.path.expanduser(_DEFAULT_CACHE)
    if os.path.exists(expanded):
        return expanded

    os.makedirs(os.path.dirname(expanded), exist_ok=True)
    tmp_path = expanded + ".tmp"
    print(f"[fvd] Downloading I3D weights to {expanded} ...")
    try:
        urllib.request.urlretrieve(_I3D_WEIGHTS_URL, tmp_path)
        os.rename(tmp_path, expanded)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    print(f"[fvd] Done: {expanded}")
    return expanded


def _symmetric_matrix_square_root(mat, eps=1e-10):
    u, s, v = torch.svd(mat)
    si = torch.where(s < eps, s, torch.sqrt(s))
    return torch.matmul(torch.matmul(u, torch.diag(si)), v.t())


def _trace_sqrt_product(sigma, sigma_v):
    sqrt_sigma = _symmetric_matrix_square_root(sigma)
    sqrt_a_sigmav_a = torch.matmul(sqrt_sigma, torch.matmul(sigma_v, sqrt_sigma))
    return torch.trace(_symmetric_matrix_square_root(sqrt_a_sigmav_a))


def _cov(m, rowvar=False):
    if m.dim() > 2:
        raise ValueError("m has more than 2 dimensions")
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    fact = 1.0 / (m.size(1) - 1)
    m = m - torch.mean(m, dim=1, keepdim=True)
    mt = m.t()
    return fact * m.matmul(mt).squeeze()


def _frechet_distance(x1: torch.Tensor, x2: torch.Tensor) -> float:
    x1 = x1.flatten(start_dim=1)
    x2 = x2.flatten(start_dim=1)
    m, m_w = x1.mean(dim=0), x2.mean(dim=0)
    sigma, sigma_w = _cov(x1, rowvar=False), _cov(x2, rowvar=False)
    mean = torch.sum((m - m_w) ** 2)
    if x1.shape[0] > 1:
        sqrt_trace_component = _trace_sqrt_product(sigma, sigma_w)
        trace = torch.trace(sigma + sigma_w) - 2.0 * sqrt_trace_component
        fd = trace + mean
    else:
        fd = np.real(mean)
    return float(fd)


def _decode_video(path: str, num_frames: int) -> torch.Tensor:
    """Decode the first `num_frames` of an mp4 into a (T, H, W, C) uint8 tensor.

    Raises ValueError if the video has fewer than `num_frames` frames.
    """
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(path)
    total = decoder.metadata.num_frames
    if total < num_frames:
        raise ValueError(
            f"Video {path} has {total} frames, fewer than required num_frames={num_frames}"
        )

    clip = decoder[0:num_frames]
    del decoder
    return clip.permute(0, 2, 3, 1).contiguous()


def _preprocess_single(video: torch.Tensor, resolution: int = 224) -> torch.Tensor:
    """Port of videogpt preprocess_single. Input: (T, H, W, C) uint8. Output: (C, T, H, W) in [-1, 1].

    Note: the upstream `sequence_length` temporal-crop parameter is omitted; callers
    are responsible for ensuring the input clip is already truncated to the desired
    number of frames (the public API does this via `_decode_video`).
    """
    import torch.nn.functional as F

    video = video.permute(0, 3, 1, 2).float() / 255.0
    t, c, h, w = video.shape
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode="bilinear", align_corners=False)
    t, c, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start:h_start + resolution, w_start:w_start + resolution]
    video = video.permute(1, 0, 2, 3).contiguous()
    video = video - 0.5
    return video * 2.0


def _preprocess_videos(videos: list[torch.Tensor], resolution: int = 224) -> torch.Tensor:
    """Stack per-video preprocess outputs into (B, C, T, H, W)."""
    return torch.stack([_preprocess_single(v, resolution) for v in videos])


def _extract_logits(i3d: torch.nn.Module, videos: torch.Tensor, device: torch.device, batch_size: int) -> torch.Tensor:
    """Run I3D over `videos` (B, C, T, H, W) in batches; return (B, 400) logits on `device`."""
    with torch.no_grad():
        out = []
        for i in range(0, videos.shape[0], batch_size):
            batch = videos[i:i + batch_size].to(device)
            out.append(i3d(batch))
        return torch.cat(out, dim=0)


def compute_fvd_from_videos(
    generated_videos: list[str],
    reference_videos: list[tuple[str, int]],
    device: torch.device,
    num_frames: int,
    weights_path: str | None = None,
    batch_size: int = 10,
) -> float:
    """Compute videogpt FVD between two sets of videos.

    Args:
        generated_videos: List of generated mp4 paths.
        reference_videos: List of (mp4_path, n_frames_in_file) tuples. The
            integer field is kept for signature parity with
            `compute_fid_from_videos`; FVD itself only uses the paths
            (every video is truncated to `num_frames`).
        device: Torch device for I3D inference and Frechet computation.
        num_frames: Clip length every video is truncated to. Must be >= 10.
        weights_path: Optional override for the I3D checkpoint. When None,
            resolve to ~/.cache/dfuse/fvd/i3d_pretrained_400.pt and
            auto-download if missing.
        batch_size: Batch size for I3D forward passes.

    Returns:
        FVD score (float). Lower is better.
    """
    if num_frames < 10:
        raise ValueError(f"num_frames must be >= 10 for I3D; got {num_frames}")
    if not generated_videos:
        raise ValueError("generated_videos is empty")
    if not reference_videos:
        raise ValueError("reference_videos is empty")

    resolved = _resolve_i3d_weights(weights_path)
    i3d = InceptionI3d(400, in_channels=3).eval().to(device)
    i3d.load_state_dict(torch.load(resolved, map_location=device, weights_only=True))

    gen_clips = [_decode_video(p, num_frames) for p in generated_videos]
    ref_clips = [_decode_video(p, num_frames) for (p, _) in reference_videos]

    gen_batch = _preprocess_videos(gen_clips)
    ref_batch = _preprocess_videos(ref_clips)

    gen_logits = _extract_logits(i3d, gen_batch, device, batch_size)
    ref_logits = _extract_logits(i3d, ref_batch, device, batch_size)

    return _frechet_distance(gen_logits, ref_logits)
