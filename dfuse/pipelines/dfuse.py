"""Dfuse."""

import random

import torch
from more_itertools import map_reduce
import torch.nn.functional as F
from typing import Optional, Union, List, Tuple
from tqdm import tqdm
from PIL import Image

from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit
from ..core.uneven_tensor import UnevenTensor

from ..models.dfuse import (
    DFuse,
)
from ..models.wan_video_text_encoder import WanTextEncoder, HuggingfaceTokenizer
from ..models.wan_video_vae import WanVideoVAE
from ..utils.ray.rays import cameras_to_rays


def _as_list(val, n: int) -> list:
    """Ensure val is a list of length n; broadcast scalars."""
    return val if isinstance(val, list) else [val] * n


def _infer_batch_size(
    num_frames: Union[int, List[int]],
    input_video: Optional[List] = None,
    source_video: Optional[List] = None,
) -> int:
    """Infer batch size from whichever argument is a list."""
    if isinstance(num_frames, list):
        return len(num_frames)
    if source_video is not None and isinstance(source_video[0], list):
        return len(source_video)
    if input_video is not None and isinstance(input_video[0], list):
        return len(input_video)
    return 1


def _get_shape_key(video) -> tuple:
    """Extract (T, H, W) from a video (list of PIL images or tensor)."""
    if isinstance(video, torch.Tensor):
        if video.ndim == 4:
            return (video.shape[0], video.shape[2], video.shape[3])
        elif video.ndim == 3:
            return (1, video.shape[1], video.shape[2])
    if isinstance(video, list) and len(video) > 0:
        frame = video[0]
        if isinstance(frame, Image.Image):
            w, h = frame.size
            return (len(video), h, w)
        if isinstance(frame, torch.Tensor):
            return (len(video), frame.shape[-2], frame.shape[-1])
    raise ValueError(f"Cannot infer shape from video of type {type(video)}")


def _batched_vae_encode(pipe, videos, tiled, tile_size, tile_stride):
    """Encode a list of videos, batching those with identical (T, H, W).

    Args:
        videos: list of N video inputs (each a list of PIL frames or a tensor).

    Returns:
        list of N latent tensors, each (C, T_lat, H_lat, W_lat),
        in the same order as the input.
    """
    tagged = [(idx, vid, _get_shape_key(vid)) for idx, vid in enumerate(videos)]
    groups = map_reduce(tagged, keyfunc=lambda x: x[2])

    results = [None] * len(videos)
    for shape_key, members in groups.items():
        indices, vids = zip(*[(m[0], m[1]) for m in members])
        batch_tensor = torch.stack(
            [pipe.preprocess_video(v).squeeze(0) for v in vids]
        )
        batch_latents = pipe.vae.encode(
            batch_tensor,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        for i, lat in zip(indices, batch_latents):
            results[i] = lat.to(dtype=pipe.torch_dtype, device=pipe.device)

    return results


def _compute_plucker_rays(intrinsics, extrinsics, H_patch, W_patch, device, dtype):
    """Compute Plücker rays at patch-grid resolution → (T, 6, H_patch, W_patch)."""
    T = extrinsics.shape[0]
    all_rays = []
    for t in range(T):
        K_t = intrinsics[t : t + 1].to(device=device, dtype=torch.float32)
        E_t = extrinsics[t : t + 1].to(device=device, dtype=torch.float32)
        rays = cameras_to_rays(
            intrinsics=K_t,
            extrinsics=E_t,
            num_patches_x=W_patch,
            num_patches_y=H_patch,
            use_plucker=True,
        )
        all_rays.append(rays.to_spatial().squeeze(0))
    return torch.stack(all_rays, dim=0).to(dtype=dtype)


def _normalize_intrinsics(Ks, image_height, image_width):
    """Normalize 3x3 intrinsics by image dimensions → unitless."""
    Ks_norm = torch.zeros_like(Ks)
    Ks_norm[..., 0, 0] = Ks[..., 0, 0] / image_width
    Ks_norm[..., 1, 1] = Ks[..., 1, 1] / image_height
    Ks_norm[..., 0, 2] = Ks[..., 0, 2] / image_width - 0.5
    Ks_norm[..., 1, 2] = Ks[..., 1, 2] / image_height - 0.5
    Ks_norm[..., 2, 2] = 1.0
    return Ks_norm


def _lift_K(Ks):
    """Lift 3x3 → homogeneous 4x4."""
    out = torch.zeros(Ks.shape[:-2] + (4, 4), device=Ks.device, dtype=Ks.dtype)
    out[..., :3, :3] = Ks
    out[..., 3, 3] = 1.0
    return out


def _invert_K(Ks):
    """Invert 3x3 intrinsics (no skew)."""
    out = torch.zeros_like(Ks)
    out[..., 0, 0] = 1.0 / Ks[..., 0, 0]
    out[..., 1, 1] = 1.0 / Ks[..., 1, 1]
    out[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
    out[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
    out[..., 2, 2] = 1.0
    return out


def _invert_SE3(T):
    """Invert 4x4 SE(3) matrix."""
    Rinv = T[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(T)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, T[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def _compute_prope_matrices(
    intrinsics,
    extrinsics,
    image_height,
    image_width,
    device,
    dtype,
):
    """Compute per-latent-frame PRoPE projection matrices.

    The Wan VAE uses causal 3D convolution: latent frame 0 sees only video
    frame 0 (causal padding), and subsequent latent frames see groups of 4
    video frames.  We pick the representative video frame for each latent
    frame as the temporal center of its receptive field:

    - j=0 → video frame 0  (causal, first frame uncompressed)
    - j≥1 → video frame clamp(round(4j - 1.5), 0, T-1)

    Args:
        intrinsics: (T_video, 3, 3) camera intrinsics.
        extrinsics: (T_video, 4, 4) camera extrinsics (camera←world).
        image_height, image_width: Original image resolution for normalization.
        device, dtype: Target device and dtype.

    Returns:
        P_T:   (T_lat, 4, 4) — P^T per latent frame (for Q).
        P_inv: (T_lat, 4, 4) — P^{-1} per latent frame (for K).
    """
    T_video = extrinsics.shape[0]
    T_lat = (T_video - 1) // 4 + 1
    lat_indices = torch.clamp(
        torch.round(4.0 * torch.arange(T_lat).float() - 1.5).long(),
        min=0,
        max=T_video - 1,
    )
    Ks = intrinsics[lat_indices].to(device=device, dtype=torch.float32)
    viewmats = extrinsics[lat_indices].to(device=device, dtype=torch.float32)

    Ks_norm = _normalize_intrinsics(Ks, image_height, image_width)
    P = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm), viewmats)
    P_T = P.transpose(-1, -2)
    P_inv = torch.einsum(
        "...ij,...jk->...ik", _invert_SE3(viewmats), _lift_K(_invert_K(Ks_norm))
    )
    return P_T.to(dtype=dtype), P_inv.to(dtype=dtype)


def _broadcast_slot0(ut: UnevenTensor, n: int) -> UnevenTensor:
    """Return an UnevenTensor of length n, every slot a clone of ut[0].

    Used by the multi-RGB ensemble path so that all N branches denoise the
    same target latent (NoiseInitializer otherwise produces N different
    noises from seed+i).
    """
    slot0 = ut[0]
    return UnevenTensor([slot0.clone() for _ in range(n)], channel_dim=ut.channel_dim)


def _average_uneven(velocity_per_slot: List[torch.Tensor]) -> torch.Tensor:
    """Mean a list of identical-shape tensors → single tensor.

    All slots are guaranteed identical-shape because they were derived from
    a single replicated slot-0 target.
    """
    return torch.stack(velocity_per_slot, dim=0).mean(dim=0)


def _detect_multi_rgb_ensemble(
    source_video,
    source_modality_list,
    cfg_drop_source: bool,
) -> Tuple[bool, int, List[int], Optional[int]]:
    """Detect whether the multi-RGB ensemble path should run.

    Trigger: source_modality_list[0] has more than one modality-0 (RGB) entry.
    Auto-trigger only — no opt-in flag.

    Returns:
        (use_ensemble, n_branches, rgb_indices, event_idx).
        rgb_indices and event_idx are positional indices into
        source_modality_list[0] / source_video[0].

    Raises:
        ValueError: if multi-RGB is detected but the event-view count is not
            exactly 1, or if cfg_drop_source is False.
        NotImplementedError: if multi-RGB is detected with batch_size > 1.
    """
    if source_video is None or source_modality_list is None:
        return (False, 1, [], None)
    if len(source_modality_list) == 0:
        return (False, 1, [], None)

    modalities = list(source_modality_list[0])
    rgb_indices = [j for j, m in enumerate(modalities) if m == 0]
    event_indices = [j for j, m in enumerate(modalities) if m == 1]
    n_rgb = len(rgb_indices)

    if n_rgb <= 1:
        return (False, 1, [], None)

    if len(source_video) > 1:
        raise NotImplementedError(
            "Multi-RGB ensemble inference is not supported with batch_size > 1 "
            f"(got {len(source_video)} samples)."
        )
    if len(event_indices) != 1:
        raise ValueError(
            "Multi-RGB ensemble inference requires exactly 1 event view; "
            f"got {len(event_indices)} event views."
        )
    if not cfg_drop_source:
        raise ValueError(
            "Multi-RGB ensemble inference requires cfg_drop_source=True "
            "(the math assumes the unconditional branch has no source)."
        )

    return (True, n_rgb, rgb_indices, event_indices[0])


def _fan_out_for_ensemble(
    *,
    source_video,
    source_modality_list,
    source_intrinsics,
    source_extrinsics,
    source_view_T_video,
    source_view_fps,
    prompt,
    negative_prompt,
    input_video,
    target_intrinsics,
    target_extrinsics,
    target_fps,
    target_modality,
    heights,
    widths,
    num_frames,
    rgb_indices: List[int],
    event_idx: int,
) -> dict:
    """Build N-branch versions of every input that needs replicating.

    For each rgb_idx in rgb_indices, produce one branch with source views
    [rgb_view, event_view]. Targets/prompt/shape are replicated N times so
    pipeline units treat the call as B=N.

    Returns a dict of overrides for the local variables in __call__. Only
    keys that need replacement are present; callers should `update` their
    locals from this dict.
    """
    n = len(rgb_indices)


    src_orig = source_video[0]
    out_source_video = [[src_orig[i], src_orig[event_idx]] for i in rgb_indices]
    out_source_modality_list = [[0, 1] for _ in range(n)]

    view_Ts = list(source_view_T_video[0])
    T_event = view_Ts[event_idx]
    out_source_view_T_video = [[view_Ts[i], T_event] for i in rgb_indices]

    out_source_view_fps = None
    if source_view_fps is not None:
        fps_orig = list(source_view_fps[0])
        f_event = fps_orig[event_idx]
        out_source_view_fps = [[fps_orig[i], f_event] for i in rgb_indices]


    out_source_intrinsics = None
    out_source_extrinsics = None
    if source_intrinsics is not None and source_extrinsics is not None:
        K_full = source_intrinsics[0]
        E_full = source_extrinsics[0]
        K_per_view = list(torch.split(K_full, view_Ts, dim=0))
        E_per_view = list(torch.split(E_full, view_Ts, dim=0))
        out_source_intrinsics = [
            torch.cat([K_per_view[i], K_per_view[event_idx]], dim=0)
            for i in rgb_indices
        ]
        out_source_extrinsics = [
            torch.cat([E_per_view[i], E_per_view[event_idx]], dim=0)
            for i in rgb_indices
        ]


    def _rep_scalar_or_list(v):
        """If v is already a list, replicate v[0] N times. Else replicate v itself."""
        if isinstance(v, list):
            return [v[0]] * n
        return [v] * n

    out = {
        "source_video": out_source_video,
        "source_modality_list": out_source_modality_list,
        "source_view_T_video": out_source_view_T_video,
        "source_view_fps": out_source_view_fps,
        "source_intrinsics": out_source_intrinsics,
        "source_extrinsics": out_source_extrinsics,
        "prompt": _rep_scalar_or_list(prompt),
        "negative_prompt": _rep_scalar_or_list(negative_prompt),
        "heights": _rep_scalar_or_list(heights),
        "widths": _rep_scalar_or_list(widths),
        "num_frames": _rep_scalar_or_list(num_frames),
    }

    if input_video is not None:
        out["input_video"] = [input_video[0]] * n
    if target_intrinsics is not None:
        out["target_intrinsics"] = [target_intrinsics[0]] * n
    if target_extrinsics is not None:
        out["target_extrinsics"] = [target_extrinsics[0]] * n
    if target_fps is not None:
        out["target_fps"] = _rep_scalar_or_list(target_fps)
    if target_modality is not None:
        out["target_modality"] = [target_modality[0]] * n

    return out


class DFuseUnit_ShapeChecker(PipelineUnit):
    """Check and adjust video dimensions to be divisible by required factors."""

    def __init__(self):
        super().__init__(
            input_params=("heights", "widths", "num_frames"),
            output_params=("heights", "widths", "num_frames"),
        )

    def process(self, pipe: "DFusePipeline", heights, widths, num_frames):

        if not isinstance(heights, list):
            heights, widths, num_frames = pipe.check_resize_height_width(
                heights, widths, num_frames
            )
        else:
            checked_h, checked_w = [], []
            for h, w in zip(heights, widths):
                ch, cw, _ = pipe.check_resize_height_width(h, w, 1)
                checked_h.append(ch)
                checked_w.append(cw)
            heights, widths = checked_h, checked_w
        return {"heights": heights, "widths": widths, "num_frames": num_frames}


class DFuseUnit_NoiseInitializer(PipelineUnit):
    """Initialize Gaussian noise as UnevenTensor for batched diffusion."""

    def __init__(self):
        super().__init__(
            input_params=(
                "heights",
                "widths",
                "num_frames",
                "seed",
                "rand_device",
                "source_video",
                "input_video",
            ),
            output_params=("noise",),
        )

    def process(
        self,
        pipe: "DFusePipeline",
        heights,
        widths,
        num_frames,
        seed,
        rand_device,
        source_video=None,
        input_video=None,
    ):
        B = _infer_batch_size(num_frames, input_video, source_video)
        nf_list = _as_list(num_frames, B)
        h_list = _as_list(heights, B)
        w_list = _as_list(widths, B)

        C = pipe.vae.model.z_dim
        vf = pipe.vae.upsampling_factor

        noise_list = []
        for i, (nf, h, w) in enumerate(zip(nf_list, h_list, w_list)):
            shape = (1, C, (nf - 1) // 4 + 1, h // vf, w // vf)
            noise_i = pipe.generate_noise(
                shape,
                seed=(seed + i if seed is not None else None),
                rand_device=rand_device,
            )
            noise_list.append(noise_i.squeeze(0))

        return {"noise": UnevenTensor(noise_list, channel_dim=0)}


class DFuseUnit_InputVideoEmbedder(PipelineUnit):
    """Encode target video (ground truth) for training or V2V.

    Contract:
    - input_video=None (inference): latents = noise, input_latents = None
    - Training: latents = noise, input_latents = vae.encode(input_video)
      FlowMatchSFTLoss will handle noising.
    - V2V inference: latents = (1-sigma)*input_latents + sigma*noise

    Camera output depends on ``pipe.dit.cam_embed_mode``:
    - **plucker**: outputs ``target_cam_rays`` (Plücker rays).
    - **prope**:  outputs ``target_cam_P_T`` / ``target_cam_P_inv`` (projection
      matrices per latent frame).
    """

    def __init__(self):
        super().__init__(
            input_params=(
                "input_video",
                "noise",
                "tiled",
                "tile_size",
                "tile_stride",
                "target_fps",
                "target_intrinsics",
                "target_extrinsics",
                "heights",
                "widths",
            ),
            output_params=(
                "latents",
                "input_latents",
                "target_fps",
                "target_cam_rays",
                "target_cam_P_T",
                "target_cam_P_inv",
            ),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe: "DFusePipeline",
        input_video,
        noise: UnevenTensor,
        tiled,
        tile_size,
        tile_stride,
        target_fps=None,
        target_intrinsics=None,
        target_extrinsics=None,
        heights=None,
        widths=None,
    ):
        h_list = _as_list(heights, noise.batch_size)
        w_list = _as_list(widths, noise.batch_size)

        cam_mode = (
            getattr(pipe.dit, "cam_embed_mode", "plucker") if pipe.dit else "plucker"
        )
        target_cam_rays = None
        target_cam_P_T = None
        target_cam_P_inv = None

        if target_intrinsics is not None and target_extrinsics is not None:
            if cam_mode == "prope":
                target_cam_P_T = []
                target_cam_P_inv = []
                for i in range(len(target_intrinsics)):
                    P_T_i, P_inv_i = _compute_prope_matrices(
                        target_intrinsics[i],
                        target_extrinsics[i],
                        h_list[i],
                        w_list[i],
                        pipe.device,
                        pipe.torch_dtype,
                    )
                    target_cam_P_T.append(P_T_i)
                    target_cam_P_inv.append(P_inv_i)
            else:
                spatial_factor = pipe.height_division_factor
                target_cam_rays = [
                    _compute_plucker_rays(
                        target_intrinsics[i],
                        target_extrinsics[i],
                        h_list[i] // spatial_factor,
                        w_list[i] // spatial_factor,
                        pipe.device,
                        pipe.torch_dtype,
                    )
                    for i in range(len(target_intrinsics))
                ]

        if input_video is None:
            return {
                "latents": noise,
                "input_latents": None,
                "target_fps": target_fps,
                "target_cam_rays": target_cam_rays,
                "target_cam_P_T": target_cam_P_T,
                "target_cam_P_inv": target_cam_P_inv,
            }

        pipe.load_models_to_device(self.onload_model_names)

        if isinstance(input_video[0], (list, torch.Tensor)):
            latents_list = _batched_vae_encode(
                pipe, input_video, tiled, tile_size, tile_stride
            )
        else:
            latents_list = _batched_vae_encode(
                pipe, [input_video], tiled, tile_size, tile_stride
            )

        input_latents = UnevenTensor(latents_list, channel_dim=0)

        if pipe.scheduler.training:
            latents = noise
        else:
            sigma = pipe.scheduler.sigmas[0]
            latents = (1 - sigma) * input_latents + sigma * noise

        return {
            "latents": latents,
            "input_latents": input_latents,
            "target_fps": target_fps,
            "target_cam_rays": target_cam_rays,
            "target_cam_P_T": target_cam_P_T,
            "target_cam_P_inv": target_cam_P_inv,
        }


class DFuseUnit_SourceVideoEmbedder(PipelineUnit):
    """Encode source/conditioning views with VAE.

    Uses ``seperate_cfg=True`` so that outputs go to inputs_posi / inputs_nega:

    * **Training** (cfg_scale=1): only the positive branch runs.  Modality
      dropout (``p_modality_drop``) and source noise are applied here.
    * **Inference posi** (cfg_scale>1): encode source normally.
    * **Inference nega** (cfg_scale>1, ``drop_source=True``): skip encoding,
      produce zero-length source latents and ``None`` camera rays.

    ``target_cam_rays`` stays in ``inputs_shared`` (from InputVideoEmbedder).
    The DiT skips camera encoding when ``source_cam_rays`` or
    ``cfg_source_view_T_video`` is ``None``.
    """

    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"drop_source": "drop_source"},
            input_params_nega={"drop_source": "drop_source"},
            input_params=(
                "source_video",
                "tiled",
                "tile_size",
                "tile_stride",
                "source_noise_timestep_range",
                "noise_level_rgb",
                "noise_level_event",
                "source_view_fps",
                "source_modality_list",
                "p_modality_drop",
                "heights",
                "widths",
                "source_intrinsics",
                "source_extrinsics",
                "source_view_T_video",
            ),
            output_params=(
                "source_latents",
                "source_lat_T_list",
                "source_fps",
                "source_timesteps",
                "source_cam_rays",
                "cfg_source_view_T_video",
                "source_cam_P_T",
                "source_cam_P_inv",
            ),
            onload_model_names=("vae",),
        )

    @staticmethod
    def _add_noise_to_view(pipe, lat_j, noise_level):
        """Add noise at a given timestep. Returns (noised_lat, timestep_float)."""
        if noise_level is None or noise_level <= 0:
            return lat_j, 0.0
        noise = pipe.generate_noise(
            (1,) + lat_j.shape, seed=None, rand_device=pipe.device
        ).squeeze(0)
        noised = pipe.scheduler.add_noise(
            lat_j, noise, torch.tensor(float(noise_level), device=pipe.device)
        )
        return noised, float(noise_level)

    @staticmethod
    def _compute_drop_set(source_video, source_modality_list, p_modality_drop):
        """Pre-compute (sample_idx, view_idx) pairs to drop via modality dropout."""
        if not p_modality_drop:
            return set()
        drop_set = set()
        for i, views in enumerate(source_video):
            for j in range(len(views)):
                mod = source_modality_list[i][j] if source_modality_list else 0
                p = p_modality_drop[mod] if mod < len(p_modality_drop) else 0.0
                if p > 0 and random.random() < p:
                    drop_set.add((i, j))
        return drop_set

    @staticmethod
    def _build_source_prope_matrices(
        source_intrinsics,
        source_extrinsics,
        source_view_T_video,
        lat_T_list,
        h_list,
        w_list,
        device,
        dtype,
    ):
        """Compute per-sample source PRoPE projection matrices.

        Dropped views (lat_T_list[i][j] == 0) contribute empty (0, 4, 4)
        matrices so that the view structure is preserved across all ranks
        (required for ZeRO-3 consistency).

        Returns:
            source_P_T:   List of (T_lat_src_total_i, 4, 4) per sample.
            source_P_inv: List of (T_lat_src_total_i, 4, 4) per sample.
        """
        B = len(source_intrinsics)
        source_P_T = []
        source_P_inv = []
        for i in range(B):
            view_Ts = source_view_T_video[i]
            int_per_view = torch.split(source_intrinsics[i], view_Ts, dim=0)
            ext_per_view = torch.split(source_extrinsics[i], view_Ts, dim=0)

            pt_parts, pinv_parts = [], []
            for j in range(len(view_Ts)):
                if lat_T_list[i][j] > 0:
                    pt_j, pinv_j = _compute_prope_matrices(
                        int_per_view[j],
                        ext_per_view[j],
                        h_list[i],
                        w_list[i],
                        device,
                        dtype,
                    )
                    pt_parts.append(pt_j)
                    pinv_parts.append(pinv_j)
                else:

                    pt_parts.append(torch.zeros(0, 4, 4, device=device, dtype=dtype))
                    pinv_parts.append(torch.zeros(0, 4, 4, device=device, dtype=dtype))

            if pt_parts:
                source_P_T.append(torch.cat(pt_parts, dim=0))
                source_P_inv.append(torch.cat(pinv_parts, dim=0))
            else:
                source_P_T.append(torch.zeros(0, 4, 4, device=device, dtype=dtype))
                source_P_inv.append(torch.zeros(0, 4, 4, device=device, dtype=dtype))
        return source_P_T, source_P_inv

    @staticmethod
    def _build_source_cam_rays(
        source_intrinsics,
        source_extrinsics,
        source_view_T_video,
        lat_T_list,
        h_list,
        w_list,
        device,
        dtype,
        spatial_factor=16,
    ):
        """Compute per-sample source Plücker rays.

        Dropped views (lat_T_list[i][j] == 0) are kept with zero-filled dummy
        rays at the original video frame count.  This preserves the view
        structure so that cam_encoder is called the same number of times on
        every rank (required for ZeRO-3).
        """
        B = len(source_intrinsics)
        source_cam_rays = []
        updated_view_T_video = []
        for i in range(B):
            H_patch, W_patch = h_list[i] // spatial_factor, w_list[i] // spatial_factor
            view_Ts = source_view_T_video[i]
            int_per_view = torch.split(source_intrinsics[i], view_Ts, dim=0)
            ext_per_view = torch.split(source_extrinsics[i], view_Ts, dim=0)

            all_ray_parts = []
            for j, T_j in enumerate(view_Ts):
                if lat_T_list[i][j] > 0:
                    rays_j = _compute_plucker_rays(
                        int_per_view[j], ext_per_view[j],
                        H_patch, W_patch, device, dtype,
                    )
                    all_ray_parts.append(rays_j)
                else:


                    all_ray_parts.append(
                        torch.zeros(T_j, 6, H_patch, W_patch, device=device, dtype=dtype)
                    )


            updated_view_T_video.append(list(view_Ts))
            if all_ray_parts:
                source_cam_rays.append(torch.cat(all_ray_parts, dim=0))
            else:
                source_cam_rays.append(
                    torch.zeros(0, 6, H_patch, W_patch, device=device, dtype=dtype)
                )
        return source_cam_rays, updated_view_T_video

    def process(
        self,
        pipe: "DFusePipeline",
        source_video,
        tiled,
        tile_size,
        tile_stride,
        source_noise_timestep_range=None,
        noise_level_rgb=None,
        noise_level_event=None,
        source_view_fps=None,
        source_modality_list=None,
        p_modality_drop=None,
        heights=None,
        widths=None,
        source_intrinsics=None,
        source_extrinsics=None,
        source_view_T_video=None,
        drop_source=None,
    ):
        B = len(source_video)
        h_list = _as_list(heights, B)
        w_list = _as_list(widths, B)
        C = pipe.vae.model.z_dim
        vf = pipe.vae.upsampling_factor


        if drop_source:
            latents_list = [
                torch.zeros(
                    C,
                    0,
                    h_list[i] // vf,
                    w_list[i] // vf,
                    dtype=pipe.torch_dtype,
                    device=pipe.device,
                )
                for i in range(B)
            ]
            lat_T_list = [[0] * len(v) for v in source_video]


            source_cam_rays = None
            cfg_source_view_T_video = None
            source_cam_P_T = None
            source_cam_P_inv = None
            if (
                source_intrinsics is not None
                and source_extrinsics is not None
                and source_view_T_video is not None
            ):
                cam_mode = (
                    getattr(pipe.dit, "cam_embed_mode", "plucker")
                    if pipe.dit
                    else "plucker"
                )
                if cam_mode == "prope":
                    source_cam_P_T, source_cam_P_inv = (
                        self._build_source_prope_matrices(
                            source_intrinsics,
                            source_extrinsics,
                            source_view_T_video,
                            lat_T_list,
                            h_list,
                            w_list,
                            pipe.device,
                            pipe.torch_dtype,
                        )
                    )
                    cfg_source_view_T_video = source_view_T_video
                else:
                    source_cam_rays, cfg_source_view_T_video = (
                        self._build_source_cam_rays(
                            source_intrinsics,
                            source_extrinsics,
                            source_view_T_video,
                            lat_T_list,
                            h_list,
                            w_list,
                            pipe.device,
                            pipe.torch_dtype,
                            spatial_factor=pipe.height_division_factor,
                        )
                    )

            return {
                "source_latents": UnevenTensor(latents_list, channel_dim=0),
                "source_lat_T_list": lat_T_list,
                "source_fps": source_view_fps,
                "source_timesteps": [[0.0] * len(v) for v in source_video],
                "source_cam_rays": source_cam_rays,
                "cfg_source_view_T_video": cfg_source_view_T_video,
                "source_cam_P_T": source_cam_P_T,
                "source_cam_P_inv": source_cam_P_inv,
            }

        pipe.load_models_to_device(self.onload_model_names)


        if pipe.scheduler.training:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                drop_set_holder = [
                    self._compute_drop_set(
                        source_video, source_modality_list, p_modality_drop
                    )
                    if dist.get_rank() == 0
                    else None
                ]
                dist.broadcast_object_list(drop_set_holder, src=0)
                drop_set = drop_set_holder[0]
            else:
                drop_set = self._compute_drop_set(
                    source_video, source_modality_list, p_modality_drop
                )
        else:
            drop_set = set()
        inference_noise = {0: noise_level_rgb, 1: noise_level_event}


        encode_jobs = []
        for i, views in enumerate(source_video):
            for j, view_j in enumerate(views):
                if (i, j) not in drop_set:
                    encode_jobs.append((i, j, view_j))


        if encode_jobs:
            encoded = _batched_vae_encode(
                pipe,
                [job[2] for job in encode_jobs],
                tiled,
                tile_size,
                tile_stride,
            )
        else:
            encoded = []


        H_lat = [h_list[i] // vf for i in range(B)]
        W_lat = [w_list[i] // vf for i in range(B)]
        view_latents = [
            [
                torch.zeros(C, 0, H_lat[i], W_lat[i],
                            dtype=pipe.torch_dtype, device=pipe.device)
                for _ in views
            ]
            for i, views in enumerate(source_video)
        ]
        for enc_idx, (i_job, j_job, _) in enumerate(encode_jobs):
            view_latents[i_job][j_job] = encoded[enc_idx]


        lat_T_list = []
        source_timesteps_list = []
        for i, views in enumerate(source_video):
            view_lat_Ts = []
            view_timesteps = []
            for j in range(len(views)):
                lat_j = view_latents[i][j]
                view_lat_Ts.append(lat_j.shape[1])


                if not pipe.scheduler.training and (i, j) not in drop_set:
                    mod = source_modality_list[i][j] if source_modality_list else 0
                    lat_j, t_val = self._add_noise_to_view(
                        pipe, lat_j, inference_noise.get(mod)
                    )
                    view_latents[i][j] = lat_j
                    view_timesteps.append(t_val)
                else:
                    view_timesteps.append(0.0)


            if pipe.scheduler.training and source_noise_timestep_range is not None:
                if random.random() < 0.7:
                    idx_min, idx_max = source_noise_timestep_range
                    idx = random.randint(idx_min, idx_max)
                    t = pipe.scheduler.timesteps[idx].item()
                    for j in range(len(views)):
                        lat_j = view_latents[i][j]
                        mod = source_modality_list[i][j] if source_modality_list else 0
                        if mod == 0 and lat_j.shape[1] > 0:
                            noise = pipe.generate_noise(
                                (1,) + lat_j.shape, seed=None, rand_device=pipe.device
                            ).squeeze(0)
                            view_latents[i][j] = pipe.scheduler.add_noise(
                                lat_j, noise, torch.tensor(t, device=pipe.device)
                            )
                            view_timesteps[j] = t

            lat_T_list.append(view_lat_Ts)
            source_timesteps_list.append(view_timesteps)


        latents_list = [torch.cat(view_latents[i], dim=1) for i in range(B)]


        cam_mode = (
            getattr(pipe.dit, "cam_embed_mode", "plucker") if pipe.dit else "plucker"
        )
        source_cam_rays = None
        source_cam_P_T = None
        source_cam_P_inv = None
        updated_source_view_T_video = source_view_T_video

        if source_intrinsics is not None and source_extrinsics is not None:
            if cam_mode == "prope":
                source_cam_P_T, source_cam_P_inv = self._build_source_prope_matrices(
                    source_intrinsics,
                    source_extrinsics,
                    source_view_T_video,
                    lat_T_list,
                    h_list,
                    w_list,
                    pipe.device,
                    pipe.torch_dtype,
                )
            else:
                source_cam_rays, updated_source_view_T_video = (
                    self._build_source_cam_rays(
                        source_intrinsics,
                        source_extrinsics,
                        source_view_T_video,
                        lat_T_list,
                        h_list,
                        w_list,
                        pipe.device,
                        pipe.torch_dtype,
                        spatial_factor=pipe.height_division_factor,
                    )
                )

        return {
            "source_latents": UnevenTensor(latents_list, channel_dim=0),
            "source_lat_T_list": lat_T_list,
            "source_fps": source_view_fps,
            "source_timesteps": source_timesteps_list,
            "source_cam_rays": source_cam_rays,
            "cfg_source_view_T_video": updated_source_view_T_video,
            "source_cam_P_T": source_cam_P_T,
            "source_cam_P_inv": source_cam_P_inv,
        }


class DFuseUnit_PromptEmbedder(PipelineUnit):
    """Encode text prompts using T5 encoder."""

    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "drop_text": "drop_text"},
            input_params_nega={"prompt": "negative_prompt", "drop_text": "drop_text"},
            input_params=("p_text_dropout",),
            output_params=("context",),
            onload_model_names=("text_encoder",),
        )

    def _encode_prompt(self, pipe, prompt: Union[str, List[str]]) -> UnevenTensor:
        if isinstance(prompt, list):
            ids_list, mask_list = [], []
            for p in prompt:
                ids_p, mask_p = pipe.tokenizer(
                    p, return_mask=True, add_special_tokens=True
                )
                ids_list.append(ids_p)
                mask_list.append(mask_p)
            ids = torch.cat(ids_list, dim=0).to(pipe.device)
            mask = torch.cat(mask_list, dim=0).to(pipe.device)
        else:
            ids, mask = pipe.tokenizer(
                prompt, return_mask=True, add_special_tokens=True
            )
            ids = ids.to(pipe.device)
            mask = mask.to(pipe.device)

        seq_lens = mask.gt(0).sum(dim=1).long()
        text_emb = pipe.text_encoder(ids, mask)
        B = text_emb.shape[0]

        trimmed = []
        for i in range(B):
            L_i = seq_lens[i].item()
            trimmed.append(text_emb[i, :L_i])

        return UnevenTensor(trimmed, channel_dim=-1)

    def process(
        self,
        pipe: "DFusePipeline",
        prompt,
        p_text_dropout=0.0,
        drop_text=None,
    ) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        context = self._encode_prompt(pipe, prompt)

        if drop_text:

            null_ctx = self._encode_prompt(pipe, [""] * context.batch_size)
            context = null_ctx
        elif pipe.scheduler.training and p_text_dropout and p_text_dropout > 0:

            null_ctx = None
            new_tensors = []
            for i in range(context.batch_size):
                if torch.rand(1).item() < p_text_dropout:
                    if null_ctx is None:
                        null_ctx = self._encode_prompt(pipe, [""])
                    new_tensors.append(null_ctx[0])
                else:
                    new_tensors.append(context[i])
            context = UnevenTensor(new_tensors, channel_dim=-1)

        return {"context": context}


def _merge_prope_matrices(target_list, source_list):
    """Concatenate per-sample [target, source] PRoPE matrices along frame dim.

    Args:
        target_list: List of (T_lat_tgt, 4, 4) per sample, or None.
        source_list: List of (T_lat_src, 4, 4) per sample, or None.

    Returns:
        List of (T_lat_total, 4, 4) per sample, or None if both inputs are None.
    """
    if target_list is None and source_list is None:
        return None
    B = len(target_list) if target_list is not None else len(source_list)
    merged = []
    for i in range(B):
        parts = []
        if target_list is not None and target_list[i] is not None:
            parts.append(target_list[i])
        if source_list is not None and source_list[i] is not None:
            parts.append(source_list[i])
        merged.append(torch.cat(parts, dim=0) if parts else None)
    return merged


def model_fn_dfuse(
    dit: DFuse,
    latents: UnevenTensor,
    source_latents: UnevenTensor,
    timestep: torch.Tensor,
    context: UnevenTensor,
    source_lat_T_list: List[List[int]],
    target_fps: Optional[List[float]] = None,
    source_fps: Optional[List[List[float]]] = None,
    target_modality: Optional[List[int]] = None,
    source_modality_list: Optional[List[List[int]]] = None,
    source_timesteps: Optional[List[List[float]]] = None,
    use_gradient_checkpointing: bool = False,
    use_activation_offload: bool = False,
    source_dropout: float = 0.0,

    target_cam_rays: Optional[List[torch.Tensor]] = None,
    source_cam_rays: Optional[List[torch.Tensor]] = None,
    cfg_source_view_T_video: Optional[List[List[int]]] = None,

    target_cam_P_T: Optional[List[torch.Tensor]] = None,
    target_cam_P_inv: Optional[List[torch.Tensor]] = None,
    source_cam_P_T: Optional[List[torch.Tensor]] = None,
    source_cam_P_inv: Optional[List[torch.Tensor]] = None,

    return_hidden_layers: Optional[List[int]] = None,
    **kwargs,
) -> UnevenTensor:
    """Forward function for D-FUSE DiT."""

    cam_P_T_list = _merge_prope_matrices(target_cam_P_T, source_cam_P_T)
    cam_P_inv_list = _merge_prope_matrices(target_cam_P_inv, source_cam_P_inv)

    return dit(
        x=latents,
        source_latents=source_latents,
        timestep=timestep,
        context=context,
        source_lat_T_list=source_lat_T_list,
        target_modality=target_modality,
        source_modality_list=source_modality_list,
        fps_target=target_fps,
        fps_source_list=source_fps,
        source_timesteps=source_timesteps,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_activation_offload=use_activation_offload,
        source_dropout=source_dropout,
        target_cam_rays=target_cam_rays,
        source_cam_rays=source_cam_rays,
        source_view_T_video=cfg_source_view_T_video,
        cam_P_T_list=cam_P_T_list,
        cam_P_inv_list=cam_P_inv_list,
        return_hidden_layers=return_hidden_layers,
    )


class DFusePipeline(BasePipeline):
    """Event+RGB Interpolation Pipeline.

    Batched variable-length version using UnevenTensor.
    Supports multi-modality source views (RGB, events) with per-modality
    attention routing in the DiT.
    """

    def __init__(
        self,
        device="cuda",
        torch_dtype=torch.bfloat16,
    ):
        super().__init__(
            device=device,
            torch_dtype=torch_dtype,
            height_division_factor=16,
            width_division_factor=16,
            time_division_factor=4,
            time_division_remainder=1,
        )
        self.scheduler = FlowMatchScheduler("Wan")
        self.tokenizer: HuggingfaceTokenizer = None
        self.text_encoder: WanTextEncoder = None
        self.dit: DFuse = None
        self.vae: WanVideoVAE = None
        self.cpu_offload = False
        self.in_iteration_models = ("dit",)
        self.units = [
            DFuseUnit_ShapeChecker(),
            DFuseUnit_NoiseInitializer(),
            DFuseUnit_InputVideoEmbedder(),
            DFuseUnit_SourceVideoEmbedder(),
            DFuseUnit_PromptEmbedder(),
        ]
        self.model_fn = model_fn_dfuse

    @classmethod
    def from_checkpoint_files(
        cls, dit_path, vae_path, text_encoder_path, tokenizer_path,
        modulation_type="learned", num_modalities=2, enable_position_phase=False,
        device="cuda", torch_dtype=torch.bfloat16, cpu_offload=True,
        model_config=None,
    ):
        """Load complete inference checkpoints using the shared model implementations."""
        from copy import deepcopy
        from ..configs import MODEL_CONFIGS
        from ..core.loader import load_model
        from ..utils.state_dict_converters.wan_video_vae import WanVideoVAEStateDictConverter

        model_config = deepcopy(model_config) if model_config is not None else deepcopy(next(c["extra_kwargs"] for c in MODEL_CONFIGS
                                     if c["model_name"] == "dfuse"))
        model_config.update(num_modalities=num_modalities,
                            modulation_type=modulation_type,
                            enable_position_phase=enable_position_phase)
        pipe = cls(device=device, torch_dtype=torch_dtype)
        load_device = "cpu" if cpu_offload else device
        pipe.dit = load_model(DFuse, str(dit_path), model_config,
                              torch_dtype=torch_dtype, device=load_device, strict=True)
        pipe.vae = load_model(WanVideoVAE, str(vae_path), torch_dtype=torch_dtype,
                              device=load_device, strict=True,
                              state_dict_converter=WanVideoVAEStateDictConverter)
        pipe.text_encoder = load_model(WanTextEncoder, str(text_encoder_path),
                                       torch_dtype=torch_dtype, device=load_device, strict=True)
        pipe.tokenizer = HuggingfaceTokenizer(name=str(tokenizer_path), seq_len=512,
                                             clean="whitespace")
        pipe.height_division_factor = pipe.vae.upsampling_factor * 2
        pipe.width_division_factor = pipe.vae.upsampling_factor * 2
        pipe.cpu_offload = cpu_offload
        pipe.eval()
        pipe.requires_grad_(False)
        return pipe

    def load_models_to_device(self, model_names):
        """Move whole inference models between CPU and the active device as needed."""
        if not self.cpu_offload:
            return super().load_models_to_device(model_names)
        for name in ("dit", "vae", "text_encoder"):
            model = getattr(self, name)
            if name not in model_names:
                model.to("cpu")
        if torch.device(self.device).type == "cuda":
            torch.cuda.empty_cache()
        for name in model_names:
            getattr(self, name).to(self.device)

    @staticmethod
    def from_pretrained(
        pretrained_model_name_or_path=None,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"
        ),
        redirect_common_files: bool = True,
        vram_limit: float = None,
        strict_dit: bool = False,
        revision=None, cache_dir=None, token=None, local_files_only=False,
        cpu_offload=True,
    ):
        """Load the pipeline from pretrained models."""
        if pretrained_model_name_or_path is not None:
            import json
            from pathlib import Path
            from huggingface_hub import snapshot_download, hf_hub_download

            options = dict(cache_dir=cache_dir, token=token,
                           local_files_only=local_files_only)
            folder = Path(pretrained_model_name_or_path)
            if not folder.is_dir():
                folder = Path(snapshot_download(
                    str(pretrained_model_name_or_path), revision=revision,
                    allow_patterns=["config.json", "dfuse.safetensors"], **options,
                ))
            config = json.loads((folder / "config.json").read_text())
            if config.get("model_type") != "dfuse":
                raise ValueError("Expected a D-FUSE model configuration.")
            components = config["components"]
            paths = {}
            for name in ("vae", "text_encoder"):
                component = components[name]
                paths[name] = hf_hub_download(
                    component["repo_id"], component["filename"],
                    revision=component.get("revision"), **options,
                )
            tokenizer = components["tokenizer"]
            tokenizer_root = snapshot_download(
                tokenizer["repo_id"], revision=tokenizer.get("revision"),
                allow_patterns=[tokenizer["subfolder"] + "/*"], **options,
            )
            model_config = config["model_config"]
            return DFusePipeline.from_checkpoint_files(
                dit_path=folder / "dfuse.safetensors", vae_path=paths["vae"],
                text_encoder_path=paths["text_encoder"],
                tokenizer_path=Path(tokenizer_root) / tokenizer["subfolder"],
                modulation_type=model_config["modulation_type"],
                num_modalities=model_config["num_modalities"],
                enable_position_phase=model_config["enable_position_phase"],
                model_config=model_config, device=device, torch_dtype=torch_dtype,
                cpu_offload=cpu_offload,
            )


        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": (
                    "DiffSynth-Studio/Wan-Series-Converted-Safetensors",
                    "models_t5_umt5-xxl-enc-bf16.safetensors",
                ),
                "Wan2.1_VAE.pth": (
                    "DiffSynth-Studio/Wan-Series-Converted-Safetensors",
                    "Wan2.1_VAE.safetensors",
                ),
            }
            for model_config in model_configs:
                if (
                    model_config.origin_file_pattern is None
                    or model_config.model_id is None
                ):
                    continue
                if (
                    model_config.origin_file_pattern in redirect_dict
                    and model_config.model_id
                    != redirect_dict[model_config.origin_file_pattern][0]
                ):
                    print(
                        f"To avoid repeatedly downloading model files, "
                        f"({model_config.model_id}, {model_config.origin_file_pattern}) "
                        f"is redirected to {redirect_dict[model_config.origin_file_pattern]}. "
                        f"You can use `redirect_common_files=False` to disable file redirection."
                    )
                    model_config.model_id = redirect_dict[
                        model_config.origin_file_pattern
                    ][0]
                    model_config.origin_file_pattern = redirect_dict[
                        model_config.origin_file_pattern
                    ][1]


        if not strict_dit:
            for model_config in model_configs:
                pattern = model_config.origin_file_pattern or ""
                is_t5 = "umt5" in pattern or "t5" in pattern.lower()
                is_vae = "VAE" in pattern or "vae" in pattern.lower()
                if not is_t5 and not is_vae:
                    model_config.strict = False

        pipe = DFusePipeline(
            device=device,
            torch_dtype=torch_dtype,
        )
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)

        pipe.text_encoder = model_pool.fetch_model("wan_video_text_encoder")
        pipe.dit = (
            model_pool.fetch_model("dfuse")
        )
        if isinstance(pipe.dit, DFuse):
            pipe.model_fn = model_fn_dfuse
        pipe.vae = model_pool.fetch_model("wan_video_vae")


        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = HuggingfaceTokenizer(
                name=tokenizer_config.path, seq_len=512, clean="whitespace"
            )

        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @torch.no_grad()
    def __call__(
        self,

        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = "",

        input_video=None,

        source_video=None,

        heights: Union[int, List[int]] = 480,
        widths: Union[int, List[int]] = 832,
        num_frames: Union[int, List[int]] = 81,

        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",

        cfg_scale: Optional[float] = 5.0,
        cfg_drop_source: bool = True,

        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,

        tiled: Optional[bool] = True,
        tile_size: Optional[tuple] = (30, 52),
        tile_stride: Optional[tuple] = (15, 26),

        target_fps: Optional[Union[float, List[float]]] = None,
        source_view_fps: Optional[Union[List[float], List[List[float]]]] = None,

        use_gradient_checkpointing: bool = False,
        use_activation_offload: bool = False,

        target_modality: Optional[List[int]] = None,
        source_modality_list: Optional[List[List[int]]] = None,

        noise_level_rgb: Optional[float] = None,
        noise_level_event: Optional[float] = None,

        target_intrinsics: Optional[List[torch.Tensor]] = None,
        target_extrinsics: Optional[List[torch.Tensor]] = None,
        source_intrinsics: Optional[List[torch.Tensor]] = None,
        source_extrinsics: Optional[List[torch.Tensor]] = None,
        source_view_T_video: Optional[List[List[int]]] = None,

        progress_bar_cmd=tqdm,
    ) -> List[List[Image.Image]]:
        """Generate video(s) via diffusion-based interpolation.

        CFG separation is handled structurally by the pipeline units:

        * **SourceVideoEmbedder** (``seperate_cfg=True``):
          posi encodes source normally; nega produces zero-length source
          when ``cfg_drop_source=True``.
        * **PromptEmbedder** (``seperate_cfg=True``):
          posi encodes the prompt; nega produces zero-length context
          when ``negative_prompt`` is None or empty, otherwise encodes
          the negative prompt as text conditioning.

        This means ``inputs_shared`` only ever holds data that is identical
        for both branches (latents, shape, etc.), and updating
        ``inputs_shared["latents"]`` in the denoising loop is automatically
        visible to both the conditional and unconditional forward passes.
        """
        self.scheduler.set_timesteps(
            num_inference_steps,
            denoising_strength=1.0,
            shift=sigma_shift,
        )


        if source_video is not None:
            if isinstance(source_video[0], (list, Image.Image)):
                if not isinstance(source_video[0], list) or (
                    len(source_video[0]) > 0
                    and isinstance(source_video[0][0], Image.Image)
                ):
                    if isinstance(source_video[0], Image.Image):
                        source_video = [[source_video]]
                    else:
                        source_video = [source_video]

            if source_view_fps is not None:
                if isinstance(source_view_fps[0], (int, float)):
                    source_view_fps = [source_view_fps]


        if target_fps is not None and not isinstance(target_fps, list):
            target_fps = [target_fps]


        use_ensemble, n_branches, rgb_indices, event_idx = (
            _detect_multi_rgb_ensemble(source_video, source_modality_list, cfg_drop_source)
        )
        if use_ensemble:
            fanned = _fan_out_for_ensemble(
                source_video=source_video,
                source_modality_list=source_modality_list,
                source_intrinsics=source_intrinsics,
                source_extrinsics=source_extrinsics,
                source_view_T_video=source_view_T_video,
                source_view_fps=source_view_fps,
                prompt=prompt,
                negative_prompt=negative_prompt,
                input_video=input_video,
                target_intrinsics=target_intrinsics,
                target_extrinsics=target_extrinsics,
                target_fps=target_fps,
                target_modality=target_modality,
                heights=heights,
                widths=widths,
                num_frames=num_frames,
                rgb_indices=rgb_indices,
                event_idx=event_idx,
            )
            source_video = fanned["source_video"]
            source_modality_list = fanned["source_modality_list"]
            source_view_T_video = fanned["source_view_T_video"]
            source_view_fps = fanned["source_view_fps"]
            source_intrinsics = fanned["source_intrinsics"]
            source_extrinsics = fanned["source_extrinsics"]
            prompt = fanned["prompt"]
            negative_prompt = fanned["negative_prompt"]
            heights = fanned["heights"]
            widths = fanned["widths"]
            num_frames = fanned["num_frames"]
            if "input_video" in fanned:
                input_video = fanned["input_video"]
            if "target_intrinsics" in fanned:
                target_intrinsics = fanned["target_intrinsics"]
            if "target_extrinsics" in fanned:
                target_extrinsics = fanned["target_extrinsics"]
            if "target_fps" in fanned:
                target_fps = fanned["target_fps"]
            if "target_modality" in fanned:
                target_modality = fanned["target_modality"]


        inputs_posi = {
            "prompt": prompt,
            "drop_source": False,
            "drop_text": False,
        }


        drop_text_nega = not negative_prompt
        inputs_nega = {
            "negative_prompt": negative_prompt or "",
            "drop_source": cfg_drop_source,
            "drop_text": drop_text_nega,
        }

        inputs_shared = {
            "heights": heights,
            "widths": widths,
            "num_frames": num_frames,
            "seed": seed,
            "rand_device": rand_device,
            "cfg_scale": cfg_scale,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
            "input_video": input_video,
            "source_video": source_video,
            "target_fps": target_fps,
            "source_view_fps": source_view_fps,
            "use_gradient_checkpointing": use_gradient_checkpointing,
            "use_activation_offload": use_activation_offload,
            "target_modality": target_modality,
            "source_modality_list": source_modality_list,
            "noise_level_rgb": noise_level_rgb,
            "noise_level_event": noise_level_event,
            "target_intrinsics": target_intrinsics,
            "target_extrinsics": target_extrinsics,
            "source_intrinsics": source_intrinsics,
            "source_extrinsics": source_extrinsics,
            "source_view_T_video": source_view_T_video,
        }


        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(
                unit, self, inputs_shared, inputs_posi, inputs_nega
            )


        if use_ensemble:
            for key in ("noise", "latents", "input_latents"):
                ut = inputs_shared.get(key)
                if ut is not None:
                    inputs_shared[key] = _broadcast_slot0(ut, n_branches)


        self.load_models_to_device(self.in_iteration_models)

        for progress_id, timestep in enumerate(
            progress_bar_cmd(self.scheduler.timesteps)
        ):
            timestep = timestep.unsqueeze(0).to(
                dtype=self.torch_dtype, device=self.device
            )

            if use_ensemble:
                timestep = timestep.expand(n_branches)


            noise_pred_posi = self.model_fn(
                dit=self.dit,
                **inputs_shared,
                **inputs_posi,
                timestep=timestep,
            )

            if use_ensemble:


                if cfg_scale != 1.0:
                    noise_pred_nega = self.model_fn(
                        dit=self.dit,
                        **inputs_shared,
                        **inputs_nega,
                        timestep=timestep,
                    )
                    combined_per_slot = [
                        noise_pred_nega[i]
                        + cfg_scale * (noise_pred_posi[i] - noise_pred_nega[i])
                        for i in range(n_branches)
                    ]
                else:
                    combined_per_slot = [
                        noise_pred_posi[i] for i in range(n_branches)
                    ]
                avg_velocity = _average_uneven(combined_per_slot)


                slot0_latent = inputs_shared["latents"][0]
                stepped_slot0_ut = self.scheduler.step(
                    UnevenTensor([avg_velocity], channel_dim=0),
                    self.scheduler.timesteps[progress_id],
                    UnevenTensor([slot0_latent], channel_dim=0),
                )
                inputs_shared["latents"] = _broadcast_slot0(
                    stepped_slot0_ut, n_branches
                )
            else:

                if cfg_scale != 1.0:
                    noise_pred_nega = self.model_fn(
                        dit=self.dit,
                        **inputs_shared,
                        **inputs_nega,
                        timestep=timestep,
                    )
                    noise_pred = noise_pred_nega + cfg_scale * (
                        noise_pred_posi - noise_pred_nega
                    )
                else:
                    noise_pred = noise_pred_posi

                inputs_shared["latents"] = self.scheduler.step(
                    noise_pred,
                    self.scheduler.timesteps[progress_id],
                    inputs_shared["latents"],
                )


        self.load_models_to_device(["vae"])
        output_videos = []
        latents_to_decode = (
            [inputs_shared["latents"][0]]
            if use_ensemble
            else list(inputs_shared["latents"])
        )
        for lat in latents_to_decode:
            lat = lat.unsqueeze(0)
            frames = self.vae.decode(
                lat,
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
            frames = self.vae_output_to_video(frames)
            output_videos.append(frames)
        self.load_models_to_device([])

        return output_videos
