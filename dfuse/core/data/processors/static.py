import random
from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch

from .base import BaseProcessor
from .multiview_pair import MultiViewPairProcessor


@dataclass
class StaticProcessorConfig:
    """Configuration for two-frame static-scene sampling.

    Draws two distinct frames from a sequence where each frame is an
    independent camera viewpoint (e.g. Co3D).  The output schema mirrors
    MultiViewPairProcessor with T=1, one target view, and one source view.

    Attributes:
        camera_id: Canonical camera_id for static datasets.  Co3D stores
            per-frame camera params under "data".
        resize_scales: List of scale factors; one is chosen per sample.
        resolution_alignment: Round H, W to this multiple (VAE requirement).
        resize_mode: F.interpolate mode ("bilinear" or "bicubic").
        antialias: Anti-aliasing flag for bilinear/bicubic resize.
        fps: Constant FPS emitted in the output (T=1, so RoPE unaffected).
        scene_scale: Multiplies extrinsics translation to amplify PRoPE
            camera-position signal.
        normalize_to_target_frame: If True, transform both extrinsics so
            target_extrinsics[0] becomes the identity matrix.
    """

    camera_id: str = "data"
    resize_scales: List[float] = field(default_factory=lambda: [1.0])
    resolution_alignment: int = 16
    resize_mode: str = "bilinear"
    antialias: bool = True
    fps: float = 30.0
    scene_scale: float = 1.0
    normalize_to_target_frame: bool = True


class StaticProcessor(BaseProcessor):
    """Sample two frames from a static-scene sequence as a 2-view pair.

    Each call to ``process`` draws:
    - Two distinct frame indices sampled uniformly without replacement.
    - Their camera intrinsics and extrinsics from the configured camera_id.
    - RGB only (no event modality for static datasets).
    - The sequence prompt.

    Output matches MultiViewPairProcessor schema with one target view and
    one source view, each of length T=1, single modality (RGB, index 0).
    """

    def __init__(self, config: StaticProcessorConfig):
        self.config = config

    def process(self, context) -> Dict[str, Any]:
        if context.dataset_type != "static":
            raise ValueError(
                f"StaticProcessor requires static data, got {context.dataset_type}."
            )

        num_frames_total = context.seq_meta["num_frames"]
        if num_frames_total < 2:
            raise ValueError(
                f"StaticProcessor needs >= 2 frames, got {num_frames_total} "
                f"for sequence {context.seq_id}."
            )

        target_idx, source_idx = random.sample(range(num_frames_total), 2)


        lo = min(target_idx, source_idx)
        hi = max(target_idx, source_idx) + 1
        cam_params = context.get_camera_params(self.config.camera_id, lo, hi)
        ext_all = torch.from_numpy(cam_params["extrinsics"]).float()
        int_all = torch.from_numpy(cam_params["intrinsics"]).float()

        target_row = target_idx - lo
        source_row = source_idx - lo
        target_extrinsics = ext_all[target_row : target_row + 1]
        source_extrinsics = ext_all[source_row : source_row + 1]
        target_intrinsics = int_all[target_row : target_row + 1]
        source_intrinsics = int_all[source_row : source_row + 1]


        target_video = context.load_images_tensor(None, target_idx, target_idx + 1)
        source_video = context.load_images_tensor(None, source_idx, source_idx + 1)

        if self.config.scene_scale != 1.0:
            target_extrinsics = target_extrinsics.clone()
            source_extrinsics = source_extrinsics.clone()
            target_extrinsics[:, :3, 3] *= self.config.scene_scale
            source_extrinsics[:, :3, 3] *= self.config.scene_scale

        if self.config.normalize_to_target_frame:
            target_extrinsics, source_extrinsics = (
                MultiViewPairProcessor._normalize_to_target_frame(
                    target_extrinsics, source_extrinsics
                )
            )


        scale = random.choice(self.config.resize_scales)
        alignment = self.config.resolution_alignment
        _, _, raw_h, raw_w = target_video.shape
        out_h = max(alignment, round(raw_h * scale / alignment) * alignment)
        out_w = max(alignment, round(raw_w * scale / alignment) * alignment)

        if out_h != raw_h or out_w != raw_w:
            tgt_in_h, tgt_in_w = raw_h, raw_w
            _, _, src_in_h, src_in_w = source_video.shape

            target_video = MultiViewPairProcessor._resize_scale_and_crop(
                target_video, out_h, out_w,
                mode=self.config.resize_mode, antialias=self.config.antialias,
            )
            source_video = MultiViewPairProcessor._resize_scale_and_crop(
                source_video, out_h, out_w,
                mode=self.config.resize_mode, antialias=self.config.antialias,
            )
            target_intrinsics = MultiViewPairProcessor._adjust_intrinsics_for_resize(
                target_intrinsics, tgt_in_h, tgt_in_w, out_h, out_w,
            )
            source_intrinsics = MultiViewPairProcessor._adjust_intrinsics_for_resize(
                source_intrinsics, src_in_h, src_in_w, out_h, out_w,
            )

        _, _, H, W = target_video.shape
        target_video_packed = target_video.permute(0, 2, 3, 1).reshape(-1, 3)
        source_video_packed = source_video.permute(0, 2, 3, 1).reshape(-1, 3)

        prompt = context.get_prompt(self.config.camera_id) or ""

        return {
            "sequence_id": context.seq_id,
            "dataset_type": "static",
            "start_idx": target_idx,
            "end_idx": target_idx + 1,
            "num_frames": 1,
            "target_height": H,
            "target_width": W,
            "target_cam_id": self.config.camera_id,
            "source_cam_ids": [self.config.camera_id],
            "num_source_views": 1,
            "source_view_num_frames": [1],
            "source_view_pixel_counts": [H * W],
            "fps": self.config.fps,
            "target_video": target_video_packed,
            "source_videos": source_video_packed,
            "prompt": prompt,
            "target_fps": self.config.fps,
            "source_view_fps": [self.config.fps],
            "target_modality": 0,
            "source_modalities": [0],
            "target_extrinsics": target_extrinsics,
            "target_intrinsics": target_intrinsics,
            "source_extrinsics": source_extrinsics,
            "source_intrinsics": source_intrinsics,
        }
