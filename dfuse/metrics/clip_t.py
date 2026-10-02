"""CLIP-T: mean CLIP image-text cosine similarity over (frame, prompt) pairs.

For each (video, prompt), decode the first `num_frames` frames, encode
them with the CLIP image encoder, encode the prompt once with the CLIP
text encoder, average cosine similarity across frames, then average
across videos. Higher = better text alignment.
"""
from typing import Union

import torch
import torch.nn.functional as F

from .clip_backbone import load as clip_load
from .clip_backbone import tokenize as clip_tokenize
from .clip_f import (
    _decode_video_frames,
    _encode_frames,
    _resolve_clip_load_args,
)


def compute_clip_t_from_videos(
    generated_videos: list[str],
    prompts: list[str],
    device: torch.device,
    num_frames: Union[int, list[int]],
    model_name: str = "ViT-B/32",
    weights_path: str | None = None,
    batch_size: int = 32,
) -> float:
    """Mean CLIP image-text cosine similarity over (frame, prompt) pairs.

    Args:
        generated_videos: List of mp4 paths.
        prompts: List of text prompts, same length as `generated_videos`.
        device: Torch device for CLIP forward passes.
        num_frames: Either an int applied to every video or a list of
            per-video frame counts (same length as `generated_videos`).
        model_name: OpenAI CLIP model name; default "ViT-B/32".
        weights_path: Optional explicit .pt path. None = use
            ~/.cache/dfuse/clip and auto-download.
        batch_size: Image-encoder batch size.

    Returns:
        Mean per-video CLIP-T score (float).
    """
    if not generated_videos:
        raise ValueError("generated_videos is empty")
    if len(prompts) != len(generated_videos):
        raise ValueError(
            f"len(prompts)={len(prompts)} != "
            f"len(generated_videos)={len(generated_videos)}"
        )
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
        for path, prompt, n in zip(generated_videos, prompts, num_frames_list):
            frames = _decode_video_frames(path, n)
            img_embs = _encode_frames(model, preprocess, frames, device, batch_size)
            tokens = clip_tokenize([prompt], truncate=True).to(device)
            with torch.no_grad():
                txt_emb = model.encode_text(tokens).float()
            txt_emb = F.normalize(txt_emb, dim=-1)
            with torch.no_grad():
                sims = F.cosine_similarity(img_embs, txt_emb.expand_as(img_embs), dim=-1)
            per_video_scores.append(float(sims.mean().item()))
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return sum(per_video_scores) / len(per_video_scores)
