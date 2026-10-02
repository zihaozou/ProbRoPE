"""CLIP-F: mean CLIP image-image cosine similarity over adjacent frames.

For each video, decode the first `num_frames` frames, encode them with
the CLIP image encoder, compute cosine similarity between every
(frame_t, frame_{t+1}) pair, average across the `num_frames - 1` pairs,
then average across videos. Higher = smoother, less flicker.
"""
import os
from typing import Union

import torch
import torch.nn.functional as F
from PIL import Image

from .clip_backbone import load as clip_load


_DEFAULT_CACHE = "~/.cache/dfuse/clip"


def _resolve_clip_load_args(model_name: str, weights_path: str | None):
    """Return `(name_arg, download_root_or_none)` for `clip_backbone.load(...)`.

    If `weights_path` is provided, it must point to an existing .pt file;
    we pass that path directly as the `name` argument to `load()`, which
    triggers its `os.path.isfile(name)` branch and skips the download.
    Otherwise, return the upstream model name plus the dfuse cache
    root so `load()` calls `_download()` into the right place.
    """
    if weights_path is not None:
        expanded = os.path.expanduser(weights_path)
        if not os.path.exists(expanded):
            raise FileNotFoundError(f"CLIP weights not found: {expanded}")
        return expanded, None
    return model_name, os.path.expanduser(_DEFAULT_CACHE)


def _decode_video_frames(path: str, num_frames: int) -> list[Image.Image]:
    """Decode the first `num_frames` of an mp4 as a list of PIL RGB images."""
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(path)
    total = decoder.metadata.num_frames
    if total < num_frames:
        raise ValueError(
            f"Video {path} has {total} frames, fewer than required num_frames={num_frames}"
        )
    clip = decoder[0:num_frames]
    del decoder
    return [
        Image.fromarray(clip[t].permute(1, 2, 0).numpy(), mode="RGB")
        for t in range(num_frames)
    ]


def _encode_frames(model, preprocess, frames: list[Image.Image], device: torch.device, batch_size: int) -> torch.Tensor:
    """Run CLIP image encoder over `frames` in chunks of `batch_size`.

    Returns L2-normalized embeddings of shape (T, D) on `device`.
    """
    embs = []
    for i in range(0, len(frames), batch_size):
        chunk = frames[i:i + batch_size]
        batch = torch.stack([preprocess(f) for f in chunk]).to(device)
        with torch.no_grad():
            e = model.encode_image(batch).float()
        e = F.normalize(e, dim=-1)
        embs.append(e)
    return torch.cat(embs, dim=0)


def compute_clip_f_from_videos(
    generated_videos: list[str],
    device: torch.device,
    num_frames: Union[int, list[int]],
    model_name: str = "ViT-B/32",
    weights_path: str | None = None,
    batch_size: int = 32,
) -> float:
    """Mean CLIP image-image cosine similarity over adjacent frame pairs.

    Args:
        generated_videos: List of mp4 paths.
        device: Torch device for CLIP forward passes.
        num_frames: Either an int applied to every video or a list of
            per-video frame counts (same length as `generated_videos`).
        model_name: OpenAI CLIP model name; default "ViT-B/32".
        weights_path: Optional explicit .pt path. None = use
            ~/.cache/dfuse/clip and auto-download.
        batch_size: Image-encoder batch size.

    Returns:
        Mean per-video CLIP-F score (float). Returns 0.0 when every
        video has < 2 frames (no adjacent pairs).
    """
    if not generated_videos:
        raise ValueError("generated_videos is empty")
    if isinstance(num_frames, int):
        num_frames_list = [num_frames] * len(generated_videos)
    else:
        if len(num_frames) != len(generated_videos):
            raise ValueError(
                f"num_frames list length {len(num_frames)} != "
                f"generated_videos length {len(generated_videos)}"
            )
        num_frames_list = list(num_frames)

    name_arg, download_root = _resolve_clip_load_args(model_name, weights_path)
    load_kwargs = {} if download_root is None else {"download_root": download_root}
    model, preprocess = clip_load(name_arg, device=device, **load_kwargs)
    model.eval()

    per_video_scores = []
    try:
        for path, n in zip(generated_videos, num_frames_list):
            if n < 2:
                per_video_scores.append(0.0)
                continue
            frames = _decode_video_frames(path, n)
            embs = _encode_frames(model, preprocess, frames, device, batch_size)
            with torch.no_grad():
                sims = F.cosine_similarity(embs[:-1], embs[1:], dim=-1)
            per_video_scores.append(float(sims.mean().item()))
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not per_video_scores:
        return 0.0
    return sum(per_video_scores) / len(per_video_scores)
