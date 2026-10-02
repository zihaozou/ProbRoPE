import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .base import BaseProcessor


@dataclass
class MultiViewPairProcessorConfig:
    """Configuration for multi-view pair sampling.

    Attributes:
        num_frames_list: Candidate temporal window lengths; one is chosen randomly per sample.
        camera_ids: Explicit camera list to use. If None, all cameras in the sequence are used.
        num_source_views: Fixed (min, max) range for number of source views to sample.
        resize_scales: List of scale factors (e.g. [1.0, 0.5, 0.25]). One is
            chosen randomly per sample. The raw resolution is multiplied by the
            scale and rounded to the nearest multiple of ``resolution_alignment``.
        resolution_alignment: Round scaled H, W to this multiple (VAE requirement).
        resize_mode: Interpolation mode passed to F.interpolate ("bilinear" or "bicubic").
        antialias: Enable anti-aliasing for bilinear/bicubic resize.
        enable_frame_rate_reduction: Per-modality frame-rate reduction flags.
            A list of bools matching source_modalities order. True means the
            modality's source view is subsampled; False keeps full rate.
            Target view is always kept at full rate.
        fps: Fixed frames-per-second used for FPS-aware RoPE.
        normalize_to_target_frame: If True, transform all extrinsics so that
            target_extrinsics[0] becomes the identity matrix. All cameras
            are expressed relative to the target camera 0's coordinate frame.
    """

    num_frames_list: List[int] = field(default_factory=lambda: [17])
    camera_ids: Optional[List[str]] = None
    num_source_views: Tuple[int, int] = (2, 2)
    resize_scales: List[float] = field(default_factory=lambda: [1.0])
    resolution_alignment: int = 16
    resize_mode: str = "bilinear"
    antialias: bool = True
    enable_frame_rate_reduction: List[bool] = field(
        default_factory=lambda: [True, False]
    )
    fps: float = 30.0
    scene_scale: float = 1.0
    source_modalities: List[str] = field(default_factory=lambda: ["rgb", "events"])
    target_modalities: List[str] = field(default_factory=lambda: ["rgb"])
    fixed_cameras_only: bool = False
    sampling_mode: str = "single_view"
    normalize_to_target_frame: bool = True


class MultiViewPairProcessor(BaseProcessor):
    """Sample event+RGB source views for frame interpolation training.

    Each call to ``process`` draws:
    - A random contiguous temporal window of length ``T`` (from ``num_frames_list``).
    - A single camera.
    - Two source views from that camera:
        1. RGB at reduced frame rate (stride from ``_find_valid_strides``).
        2. Events at full frame rate.
    - Target: full-rate RGB from the same camera.

    Output tensors use variable-length concatenation along the temporal axis:
        source_videos:     (T_rgb + T_events, 3, H, W)  uint8
    """

    def __init__(self, config: MultiViewPairProcessorConfig):
        self.config = config


        self.modality_to_index: Dict[str, int] = {
            name: idx for idx, name in enumerate(config.source_modalities)
        }

        for tm in config.target_modalities:
            if tm not in self.modality_to_index:
                raise ValueError(
                    f"target_modality '{tm}' is not in source_modalities "
                    f"{config.source_modalities}"
                )

    @property
    def num_modalities(self) -> int:
        """Number of distinct modalities (= len(source_modalities))."""
        return len(self.config.source_modalities)

    @staticmethod
    def _load_modality_video(
        context, camera_id: str, start_idx: int, end_idx: int, modality_name: str
    ) -> torch.Tensor:
        """Load video tensor for a given modality.

        Args:
            context: ProcessorContext with data loading methods.
            camera_id: Camera identifier (e.g. 'cam_0').
            start_idx: Start frame index (inclusive).
            end_idx: End frame index (exclusive).
            modality_name: Modality string (e.g. 'rgb', 'depth', 'events').

        Returns:
            Tensor of shape (T, 3, H, W) uint8.
        """
        if modality_name == "rgb":
            return context.load_images_tensor(camera_id, start_idx, end_idx)
        elif modality_name == "depth":

            raise NotImplementedError(
                "Depth modality loading is not yet implemented. "
                "Override _load_modality_video to provide depth→RGB conversion."
            )
        elif modality_name == "events":
            events = context.load_events(
                camera_id, start_idx, end_idx
            )
            if events is None:
                raise FileNotFoundError(
                    f"Event video not found for camera {camera_id} in sequence."
                )

            return torch.from_numpy(events[:, [5, 3, 0], :, :])
        else:
            raise ValueError(f"Unknown modality name: {modality_name}")

    @staticmethod
    def _find_valid_strides(T: int) -> List[int]:
        """Find strides that produce VAE-compatible subsampled lengths.

        A stride ``s`` is valid when:
        - ``(T - 1) % s == 0`` so that both first and last frames are selected.
        - The resulting count ``n = (T - 1) // s + 1`` satisfies ``(n - 1) % 4 == 0``
          (VAE-compatible).

        Additionally, ``stride = T`` is always included (when ``T > 1``) to
        allow single-frame reduction (keep only frame 0).  ``n = 1`` satisfies
        the VAE constraint since ``(1 - 1) % 4 == 0``.

        Returns list of valid strides (excluding stride 1 which is always the
        full-rate fallback).
        """
        strides = []
        s = 2
        while s <= T - 1:
            if (T - 1) % s == 0:
                n = (T - 1) // s + 1
                if (n - 1) % 4 == 0:
                    strides.append(s)
            s *= 2

        if T > 1:
            strides.append(T)
        return strides

    @staticmethod
    def _intersect_intervals(
        intervals_a: List[Tuple[int, int]],
        intervals_b: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """Compute intersection of two sorted lists of [start, end) intervals.

        Both inputs must be sorted by start. Returns a sorted list of
        non-empty overlap intervals.
        """
        result: List[Tuple[int, int]] = []
        i, j = 0, 0
        while i < len(intervals_a) and j < len(intervals_b):
            lo = max(intervals_a[i][0], intervals_b[j][0])
            hi = min(intervals_a[i][1], intervals_b[j][1])
            if lo < hi:
                result.append((lo, hi))
            if intervals_a[i][1] < intervals_b[j][1]:
                i += 1
            else:
                j += 1
        return result

    @staticmethod
    def _resize_scale_and_crop(
        video: torch.Tensor,
        out_h: int,
        out_w: int,
        mode: str,
        antialias: bool,
    ) -> torch.Tensor:
        """Scale-to-fill then center-crop video tensor."""
        if video.ndim != 4:
            raise ValueError(
                f"Expected video tensor with shape (T, C, H, W), got {video.shape}"
            )

        _, _, in_h, in_w = video.shape
        if in_h == out_h and in_w == out_w:
            return video

        scale = max(out_h / in_h, out_w / in_w)
        resized_h = max(1, int(round(in_h * scale)))
        resized_w = max(1, int(round(in_w * scale)))

        resized = F.interpolate(
            video.float(),
            size=(resized_h, resized_w),
            mode=mode,
            align_corners=False if mode in ("bilinear", "bicubic") else None,
            antialias=antialias if mode in ("bilinear", "bicubic") else False,
        )

        top = max(0, (resized_h - out_h) // 2)
        left = max(0, (resized_w - out_w) // 2)
        cropped = resized[:, :, top : top + out_h, left : left + out_w]
        return cropped.to(video.dtype)

    @staticmethod
    def _adjust_intrinsics_for_resize(
        intrinsics: torch.Tensor,
        in_h: int,
        in_w: int,
        out_h: int,
        out_w: int,
    ) -> torch.Tensor:
        """Adjust intrinsics after scale-to-fill + center-crop.

        Applies the same transform as _resize_scale_and_crop: uniform scale
        then center-crop offset on the principal point.

        Args:
            intrinsics: (N, 3, 3) intrinsics matrices.
            in_h, in_w: Original video resolution.
            out_h, out_w: Target resolution after resize+crop.

        Returns:
            Adjusted (N, 3, 3) intrinsics.
        """
        if in_h == out_h and in_w == out_w:
            return intrinsics

        adjusted = intrinsics.clone()
        scale = max(out_h / in_h, out_w / in_w)
        resized_h = max(1, int(round(in_h * scale)))
        resized_w = max(1, int(round(in_w * scale)))
        top = max(0, (resized_h - out_h) // 2)
        left = max(0, (resized_w - out_w) // 2)


        adjusted[:, 0, :] *= scale
        adjusted[:, 1, :] *= scale

        adjusted[:, 0, 2] -= left
        adjusted[:, 1, 2] -= top
        return adjusted

    @staticmethod
    def _normalize_to_target_frame(
        target_ext: torch.Tensor,
        source_ext: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform extrinsics so target_ext[0] becomes the identity matrix.

        W2C convention: p_cam = E @ p_world (homogeneous 4x4).
        Normalizing to target camera 0's frame:
            E_new = E_old @ E0_inv
        makes E_target_new[0] = I, so the target camera sits at the origin
        with identity orientation.  All other cameras are expressed relative
        to this frame.  Intrinsics are invariant and are not passed in.
        """
        E0_inv = torch.linalg.inv(target_ext[0])
        return target_ext @ E0_inv, source_ext @ E0_inv

    def process(self, context) -> Dict[str, Any]:
        if context.dataset_type != "multiview":
            raise ValueError(
                f"MultiViewPairProcessor requires multiview data, got {context.dataset_type}."
            )

        camera_ids = self.config.camera_ids or context.get_camera_ids()
        if camera_ids is None or len(camera_ids) < 1:
            raise ValueError(f"Need at least one camera_id for {context.seq_id}.")


        if self.config.fixed_cameras_only:
            fixed_ids = context.get_fixed_camera_ids()
            if fixed_ids:
                camera_ids = [c for c in camera_ids if c in fixed_ids]
            if not camera_ids:
                raise ValueError(f"No fixed cameras available for {context.seq_id}.")


        fps = self.config.fps
        num_frames_total = context.seq_meta["num_frames"]
        target_num_frames = int(random.choice(self.config.num_frames_list))


        random.shuffle(camera_ids)
        cam = None
        start_idx = None
        source_cams = None

        if self.config.fixed_cameras_only:
            if self.config.sampling_mode == "multi_view":


                num_sources = len(self.config.source_modalities)
                max_attempts = max(len(camera_ids) ** 2 * 10, 100)
                for _ in range(max_attempts):
                    target_candidate = random.choice(camera_ids)
                    src_candidates = [
                        random.choice(camera_ids) for _ in range(num_sources)
                    ]
                    unique_cams = list(set([target_candidate] + src_candidates))


                    common = context.get_valid_frame_intervals(
                        unique_cams[0], min_length=1
                    )
                    common = sorted(common)
                    for uc in unique_cams[1:]:
                        other = context.get_valid_frame_intervals(uc, min_length=1)
                        other = sorted(other)
                        common = self._intersect_intervals(common, other)
                        if not common:
                            break


                    common = [(s, e) for s, e in common if e - s >= target_num_frames]
                    if common:
                        cam = target_candidate
                        source_cams = src_candidates
                        weights = [e - s - target_num_frames + 1 for s, e in common]
                        interval = random.choices(common, weights=weights, k=1)[0]
                        start_idx = random.randint(
                            interval[0], interval[1] - target_num_frames
                        )
                        break

                if cam is None:
                    raise ValueError(
                        f"No camera combination in {context.seq_id} has common "
                        f"valid intervals of length >= {target_num_frames}."
                    )
            else:

                for candidate in camera_ids:
                    intervals = context.get_valid_frame_intervals(
                        candidate, min_length=target_num_frames
                    )
                    if intervals:
                        cam = candidate
                        weights = [
                            end - start - target_num_frames + 1
                            for start, end in intervals
                        ]
                        interval = random.choices(intervals, weights=weights, k=1)[0]
                        start_idx = random.randint(
                            interval[0], interval[1] - target_num_frames
                        )
                        break
                if cam is None:
                    raise ValueError(
                        f"No fixed camera in {context.seq_id} has a valid interval "
                        f"of length >= {target_num_frames}."
                    )
        else:
            cam = camera_ids[0]
            if target_num_frames > num_frames_total:
                raise ValueError(
                    f"Sequence {context.seq_id} has {num_frames_total} frames, "
                    f"cannot sample {target_num_frames} frames."
                )
            start_idx = int(random.randint(0, num_frames_total - target_num_frames))

        end_idx = start_idx + target_num_frames


        T = target_num_frames
        valid_strides = self._find_valid_strides(T)


        if source_cams is None:
            if self.config.sampling_mode == "multi_view":
                source_cams = [
                    random.choice(camera_ids) for _ in self.config.source_modalities
                ]
            else:

                source_cams = [cam] * len(self.config.source_modalities)


        target_modality_name = self.config.target_modalities[0]
        target_video = self._load_modality_video(
            context, cam, start_idx, end_idx, target_modality_name
        )


        target_cam_params = context.get_camera_params(cam, start_idx, end_idx)
        target_extrinsics = torch.from_numpy(target_cam_params["extrinsics"]).float()
        target_intrinsics = torch.from_numpy(target_cam_params["intrinsics"]).float()


        if self.config.scene_scale != 1.0:
            target_extrinsics[:, :3, 3] *= self.config.scene_scale


        source_chunks = []
        source_view_num_frames = []
        source_view_fps_list = []
        source_modality_indices = []
        source_cam_ids = []
        source_ext_chunks = []
        source_int_chunks = []

        for mod_idx, mod_name in enumerate(self.config.source_modalities):
            src_cam = source_cams[mod_idx]


            do_reduce = (
                mod_idx < len(self.config.enable_frame_rate_reduction)
                and self.config.enable_frame_rate_reduction[mod_idx]
                and valid_strides
            )
            stride = random.choice(valid_strides) if do_reduce else 1

            src_video = self._load_modality_video(
                context, src_cam, start_idx, end_idx, mod_name
            )
            indices = list(range(0, T, stride))
            src_video = src_video[indices]


            src_cam_params = context.get_camera_params(src_cam, start_idx, end_idx)
            src_ext = src_cam_params["extrinsics"][indices]
            src_int = src_cam_params["intrinsics"][indices]

            source_chunks.append(src_video)
            source_view_num_frames.append(len(indices))
            source_view_fps_list.append(fps if len(indices) == 1 else fps / stride)
            source_modality_indices.append(self.modality_to_index[mod_name])
            source_cam_ids.append(src_cam)
            src_ext_tensor = torch.from_numpy(src_ext).float()
            if self.config.scene_scale != 1.0:
                src_ext_tensor[:, :3, 3] *= self.config.scene_scale
            source_ext_chunks.append(src_ext_tensor)
            source_int_chunks.append(torch.from_numpy(src_int).float())


        source_videos = torch.cat(source_chunks, dim=0)
        source_extrinsics = torch.cat(source_ext_chunks, dim=0)
        source_intrinsics = torch.cat(source_int_chunks, dim=0)


        if self.config.normalize_to_target_frame:
            target_extrinsics, source_extrinsics = (
                self._normalize_to_target_frame(target_extrinsics, source_extrinsics)
            )


        scale = random.choice(self.config.resize_scales)
        alignment = self.config.resolution_alignment

        _, _, raw_h, raw_w = target_video.shape
        out_h = max(alignment, round(raw_h * scale / alignment) * alignment)
        out_w = max(alignment, round(raw_w * scale / alignment) * alignment)

        if out_h != raw_h or out_w != raw_w:
            _, _, tgt_in_h, tgt_in_w = target_video.shape
            _, _, src_in_h, src_in_w = source_videos.shape

            target_video = self._resize_scale_and_crop(
                target_video,
                out_h,
                out_w,
                mode=self.config.resize_mode,
                antialias=self.config.antialias,
            )
            source_videos = self._resize_scale_and_crop(
                source_videos,
                out_h,
                out_w,
                mode=self.config.resize_mode,
                antialias=self.config.antialias,
            )
            target_intrinsics = self._adjust_intrinsics_for_resize(
                target_intrinsics,
                tgt_in_h,
                tgt_in_w,
                out_h,
                out_w,
            )
            source_intrinsics = self._adjust_intrinsics_for_resize(
                source_intrinsics,
                src_in_h,
                src_in_w,
                out_h,
                out_w,
            )


        T_tgt, _, H, W = target_video.shape
        target_video_packed = target_video.permute(0, 2, 3, 1).reshape(-1, 3)


        source_videos_packed = source_videos.permute(0, 2, 3, 1).reshape(-1, 3)


        source_view_pixel_counts = [nf * H * W for nf in source_view_num_frames]

        prompt = context.get_prompt(cam) or ""

        target_modality_idx = self.modality_to_index[target_modality_name]

        return {
            "sequence_id": context.seq_id,
            "dataset_type": "multiview",
            "start_idx": start_idx,
            "end_idx": end_idx,
            "num_frames": target_num_frames,
            "target_height": H,
            "target_width": W,
            "target_cam_id": cam,
            "source_cam_ids": source_cam_ids,
            "num_source_views": len(self.config.source_modalities),
            "source_view_num_frames": source_view_num_frames,
            "source_view_pixel_counts": source_view_pixel_counts,
            "fps": fps,
            "target_video": target_video_packed,
            "source_videos": source_videos_packed,
            "prompt": prompt,
            "target_fps": fps,
            "source_view_fps": source_view_fps_list,
            "target_modality": target_modality_idx,
            "source_modalities": source_modality_indices,
            "target_extrinsics": target_extrinsics,
            "target_intrinsics": target_intrinsics,
            "source_extrinsics": source_extrinsics,
            "source_intrinsics": source_intrinsics,
        }
