"""Shared DFuse validation infrastructure.

Single source of truth for:

- `DFuseValDataset`: SQLite-metadata-backed validation dataset (used by
  both the training-time validation loop and the standalone inference script).
- `run_validation`: per-rank validation function (gather + dedup
  happens upstream of the metric callback).
- `merge_validation_results`: rank-0 gather merge with dedup-by-`is_redundant`.
- `compute_dfuse_metrics`: rank-0 metric callback over the merged dict.

This module is consumed by:
- `scripts/train_dfuse.py` (training-time validation hook)
- `scripts/run_validation.py` (standalone distributed inference)
- `dfuse/diffusion/runner.py` (imports `merge_validation_results`)
"""

import glob
import json
import math
import os
import re

import numpy as np
import torch
from PIL import Image

from dfuse.utils.metadata_db import MetadataDB
from dfuse.core.data.processors import MultiViewPairProcessor
from dfuse.utils.data import save_video


def _tensor_to_pil_list(video: torch.Tensor) -> list:
    """Convert (T, 3, H, W) uint8 tensor to List[PIL.Image]."""
    return [
        Image.fromarray(video[t].permute(1, 2, 0).numpy(), mode="RGB")
        for t in range(video.shape[0])
    ]


_MODALITY_NAME_TO_INT = {"rgb": 0, "events": 1}


class DFuseValDataset(torch.utils.data.Dataset):
    """Validation dataset for DFuse using metadata.db format.

    Reads sequences from a metadata.db SQLite database and loads frames from
    per-camera mp4 files. The user picks a set of source cameras (by index
    into the sequence's ``camera_ids`` list) along with the modality each
    camera should be loaded as. The target is always RGB and iterates over
    every camera in the sequence — emitting one case per ``(seq, target_cam)``
    pair.

    Args:
        base_path: directory containing metadata.db.
        source_camera_indices: list of int indices into the per-sequence
            ``camera_ids`` list. May contain duplicates (same physical camera
            loaded as multiple modalities).
        source_camera_modalities: list of modality names ("rgb" / "events"),
            same length as ``source_camera_indices``.
        enable_frr: list of bool, same length as ``source_camera_indices``;
            whether to apply frame-rate reduction to that source camera.
    """

    def __init__(
        self,
        base_path,
        enable_frr,
        args,
        source_camera_indices=None,
        source_camera_modalities=None,
        source_modalities=None,
        val_num_frames=None,
        frr_stride=None,
        load_depth=False,
        all_target_cameras=True,
        max_scenes=None,
    ):


        if not all_target_cameras:
            raise ValueError(
                "all_target_cameras=False is no longer supported; "
                "use the new (source_camera_indices, source_camera_modalities) "
                "API to control source views."
            )


        if source_camera_indices is None or source_camera_modalities is None:
            if source_modalities is None:
                raise ValueError(
                    "must provide either (source_camera_indices, "
                    "source_camera_modalities) or legacy source_modalities"
                )
            _legacy_idx_for = {"rgb": 0, "events": 1}
            source_camera_indices = [
                _legacy_idx_for.get(m, 0) for m in source_modalities
            ]
            source_camera_modalities = list(source_modalities)

        if len(source_camera_indices) != len(source_camera_modalities):
            raise ValueError(
                f"source_camera_indices ({len(source_camera_indices)}) and "
                f"source_camera_modalities ({len(source_camera_modalities)}) "
                f"must have the same length"
            )
        for m in source_camera_modalities:
            if m not in _MODALITY_NAME_TO_INT:
                raise ValueError(
                    f"unknown source camera modality {m!r}; "
                    f"allowed: {sorted(_MODALITY_NAME_TO_INT)}"
                )

        self.base_path = base_path
        self.source_camera_indices = list(source_camera_indices)
        self.source_camera_modalities = list(source_camera_modalities)
        self.enable_frr = enable_frr
        self.args = args
        self.val_num_frames = val_num_frames
        self.frr_stride = frr_stride
        self.load_depth = load_depth

        self.db = MetadataDB(os.path.join(base_path, "metadata.db"), readonly=True)
        seq_ids = self.db.get_all_seq_ids()

        self.cases = []
        self.seq_info = {}


        self.per_seq_frr_stride: dict[str, int] = {}
        _stride_suffix_re = re.compile(r"_str(\d+)$")

        min_cams_required = (max(self.source_camera_indices) + 1) if self.source_camera_indices else 1

        kept_scenes = 0
        for seq_id in sorted(seq_ids):
            info = self.db.get_sequence_info(seq_id)
            if info is None:
                continue
            cam_ids = info.get("camera_ids") or []
            if len(cam_ids) < min_cams_required:
                continue
            if max_scenes is not None and kept_scenes >= max_scenes:
                break
            self.seq_info[seq_id] = info

            m = _stride_suffix_re.search(seq_id)
            if m is not None:
                self.per_seq_frr_stride[seq_id] = int(m.group(1))

            src_cams = [cam_ids[i] for i in self.source_camera_indices]
            for target_cam in cam_ids:
                self.cases.append((seq_id, target_cam, target_cam, src_cams))
            kept_scenes += 1

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        from torchcodec.decoders import VideoDecoder

        seq_id, case_tag, target_cam, src_cams = self.cases[idx]
        info = self.seq_info[seq_id]
        num_frames = info["num_frames"]
        if self.val_num_frames is not None:
            num_frames = min(num_frames, self.val_num_frames)
        frame_h = info["height"]
        frame_w = info["width"]
        seq_dir = os.path.join(self.base_path, seq_id)


        valid_strides = MultiViewPairProcessor._find_valid_strides(num_frames)

        valid_strides = [s for s in valid_strides if s < num_frames]
        per_seq_override = self.per_seq_frr_stride.get(seq_id)
        if per_seq_override is not None:
            if per_seq_override not in valid_strides:
                raise ValueError(
                    f"per-seq stride {per_seq_override} (from seq_id "
                    f"suffix on {seq_id!r}) is not valid for "
                    f"num_frames={num_frames}; valid strides are {valid_strides}"
                )
            frr_stride = per_seq_override
        elif self.frr_stride is not None:
            if self.frr_stride not in valid_strides:
                raise ValueError(
                    f"requested frame-rate reduction factor {self.frr_stride} is not "
                    f"valid for num_frames={num_frames}; valid strides are {valid_strides}"
                )
            frr_stride = self.frr_stride
        else:
            frr_stride = max(valid_strides) if valid_strides else 1


        gt_decoder = VideoDecoder(os.path.join(seq_dir, f"images_{target_cam}.mp4"))
        video_fps = gt_decoder.metadata.average_fps
        gt_frames = _tensor_to_pil_list(gt_decoder[0:num_frames])
        del gt_decoder

        gt_mp4_path = os.path.join(seq_dir, f"images_{target_cam}.mp4")


        src_views = []
        fps_list = []
        modality_indices = []
        view_camera_ids = []
        src_num_frames = []
        kept_view_origins = []

        for view_i, (cam, mod_name) in enumerate(
            zip(src_cams, self.source_camera_modalities)
        ):
            mod_int = _MODALITY_NAME_TO_INT[mod_name]
            do_reduce = view_i < len(self.enable_frr) and self.enable_frr[view_i]

            if mod_name == "rgb":
                mp4 = os.path.join(seq_dir, f"images_{cam}.mp4")
                if not os.path.exists(mp4):
                    continue
                decoder = VideoDecoder(mp4)
                if do_reduce:
                    indices = list(range(0, num_frames, frr_stride))
                    frames_tensor = decoder.get_frames_at(indices=indices).data
                    fps_list.append(video_fps / frr_stride)
                else:
                    frames_tensor = decoder[0:num_frames]
                    fps_list.append(video_fps)
                del decoder
                src_views.append(_tensor_to_pil_list(frames_tensor))
                src_num_frames.append(frames_tensor.shape[0])

            elif mod_name == "events":
                mp4 = os.path.join(seq_dir, f"events_{cam}.mp4")
                if not os.path.exists(mp4):
                    continue
                decoder = VideoDecoder(mp4)
                frames_tensor = decoder[0:num_frames]
                del decoder
                src_views.append(self._parse_event_grid_to_rgb(frames_tensor))
                fps_list.append(video_fps)
                src_num_frames.append(num_frames)

            else:
                continue

            modality_indices.append(mod_int)
            view_camera_ids.append(cam)
            kept_view_origins.append(view_i)


        source_depth = None
        if self.load_depth and view_camera_ids:
            first_cam = view_camera_ids[0]
            depth_indices = list(range(num_frames))
            depths_h5_path = os.path.join(seq_dir, "depths.h5")
            if os.path.exists(depths_h5_path):
                try:
                    import h5py
                    with h5py.File(depths_h5_path, "r") as hf:
                        if first_cam in hf:
                            dset = hf[first_cam]
                            source_depth = np.stack(
                                [np.array(dset[i], dtype=np.float32) for i in depth_indices]
                            )
                except Exception as _depth_exc:
                    print(f"  [warn] depth load failed for {first_cam} in {depths_h5_path}: {_depth_exc}")
                    source_depth = None


        cam_kwargs = {}
        tgt_params = self.db.get_camera_params(
            seq_id,
            camera_id=target_cam,
            frame_idx_start=0,
            frame_idx_end=num_frames,
        )
        if tgt_params:
            tgt_by_frame = {p["frame_idx"]: p for p in tgt_params}
            tgt_int = np.stack(
                [tgt_by_frame[fi]["intrinsics"] for fi in range(num_frames)], axis=0
            )
            tgt_ext = np.stack(
                [tgt_by_frame[fi]["extrinsics"] for fi in range(num_frames)], axis=0
            )

            src_int_parts = []
            src_ext_parts = []
            cam_T_video = []
            for kept_i, orig_view_i in enumerate(kept_view_origins):
                cam = view_camera_ids[kept_i]
                do_reduce = (
                    orig_view_i < len(self.enable_frr)
                    and self.enable_frr[orig_view_i]
                )
                n_src = src_num_frames[kept_i]

                src_params = self.db.get_camera_params(
                    seq_id,
                    camera_id=cam,
                    frame_idx_start=0,
                    frame_idx_end=num_frames,
                )
                if not src_params:
                    continue
                src_by_frame = {p["frame_idx"]: p for p in src_params}
                frame_indices = (
                    list(range(0, num_frames, frr_stride))
                    if do_reduce
                    else list(range(num_frames))
                )
                src_int_parts.append(
                    np.stack(
                        [
                            src_by_frame[fi]["intrinsics"]
                            for fi in frame_indices[:n_src]
                        ],
                        axis=0,
                    )
                )
                src_ext_parts.append(
                    np.stack(
                        [
                            src_by_frame[fi]["extrinsics"]
                            for fi in frame_indices[:n_src]
                        ],
                        axis=0,
                    )
                )
                cam_T_video.append(n_src)

            if src_int_parts:
                target_extrinsics = torch.from_numpy(tgt_ext)
                source_extrinsics = torch.from_numpy(
                    np.concatenate(src_ext_parts, axis=0)
                )


                scene_scale = getattr(self.args, "scene_scale", 1.0)
                if scene_scale != 1.0:
                    target_extrinsics[:, :3, 3] *= scene_scale
                    source_extrinsics[:, :3, 3] *= scene_scale


                if getattr(self.args, "normalize_to_target_frame", True):
                    target_extrinsics, source_extrinsics = (
                        MultiViewPairProcessor._normalize_to_target_frame(
                            target_extrinsics, source_extrinsics
                        )
                    )

                cam_kwargs = {
                    "target_intrinsics": torch.from_numpy(tgt_int),
                    "target_extrinsics": target_extrinsics,
                    "source_intrinsics": torch.from_numpy(
                        np.concatenate(src_int_parts, axis=0)
                    ),
                    "source_extrinsics": source_extrinsics,
                    "source_view_T_video": cam_T_video,
                }


        prompt = self.db.get_prompt(seq_id, target_cam) or ""

        return {
            "sample_name": seq_id,
            "case_tag": case_tag,
            "target_cam_id": target_cam,
            "source_views": src_views,
            "gt_frames": gt_frames,
            "gt_mp4_path": gt_mp4_path,
            "camera_data": cam_kwargs,
            "prompt": prompt,
            "target_fps": video_fps,
            "source_fps_list": fps_list,
            "num_frames": num_frames,
            "source_modality_indices": modality_indices,
            "source_view_camera_ids": view_camera_ids,
            "frame_h": frame_h,
            "frame_w": frame_w,
            "source_depth": source_depth,
        }

    @staticmethod
    def _parse_event_grid_to_rgb(frames_tensor):
        """Convert (T, C, H*2, W*3) event tensor to List[PIL.Image] (RGB).

        Event videos store 6 temporal bins in a 2×3 grid layout:
            [bin_0] [bin_1] [bin_2]
            [bin_3] [bin_4] [bin_5]

        Maps bin_5->R, bin_3->G, bin_0->B for display.
        """
        arr = frames_tensor[:, 0].numpy()
        bh, bw = arr.shape[1] // 2, arr.shape[2] // 3
        bin_0 = arr[:, 0:bh, 0:bw]
        bin_3 = arr[:, bh : 2 * bh, 0:bw]
        bin_5 = arr[:, bh : 2 * bh, 2 * bw : 3 * bw]
        rgb = np.stack([bin_5, bin_3, bin_0], axis=-1)
        return [Image.fromarray(rgb[t]) for t in range(rgb.shape[0])]


def _read_real_world_metadata(
    scene_dir: str,
    num_source_frames: int | None,
    frame_range: tuple[int, int] | None = None,
):
    """Read metadata.json from a real-world scene directory.

    Args:
        scene_dir: Directory containing metadata.json with cam0 / cam1
            calibration and a per-frame timestamp list.
        num_source_frames: Optional cap on the number of source RGB
            frames consumed (head of ``metadata["frames"]``). ``None``
            uses every frame. Mutually exclusive with ``frame_range``.
        frame_range: Optional ``(a, b)`` half-open RGB-frame range
            ``[a, b)``. Mutually exclusive with ``num_source_frames``.

    Returns:
        Dict with:
        - ``cam0``: dict with ``K_processed`` (3x3 list) and ``w2c``
          (4x4 list), the rgb camera.
        - ``cam1``: same shape, the events camera.
        - ``frames``: the (possibly cropped) per-frame metadata list.
        - ``T_src``: number of source RGB frames used (= len(frames)).
        - ``frame_h``, ``frame_w``: image height/width pulled from
          ``cam0.image_size`` (which is stored as ``[W, H]``).
        - ``native_fps``: float, derived from
          ``1.0 / (frames[1].timestamp_s - frames[0].timestamp_s)``.
        - ``start_time_s``: timestamp of the first kept frame (used
          to offset events when frame_range is set; 0.0 otherwise
          unless the dataset itself starts at non-zero t).

    Raises:
        FileNotFoundError: if metadata.json is missing.
        ValueError: if the metadata is malformed (missing cam0/cam1,
            <2 frames, non-uniform image_size, etc.).
    """
    metadata_path = os.path.join(scene_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.json not found at {metadata_path}")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    cameras = metadata.get("cameras", {})


    src_keys = metadata.get("source_modality_keys")
    if src_keys is not None:
        if len(src_keys) < 2:
            raise ValueError(
                f"source_modality_keys must list >= 2 cams (last = event), "
                f"got {src_keys}"
            )
        for k in src_keys:
            if k not in cameras:
                raise ValueError(f"cameras.{k} declared in source_modality_keys "
                                 f"but missing")
            if "K_processed" not in cameras[k] or "w2c" not in cameras[k]:
                raise ValueError(f"cameras.{k} missing K_processed or w2c")
        rgb_keys = list(src_keys[:-1])
        event_key = src_keys[-1]
    else:
        if "cam0" not in cameras or "cam1" not in cameras:
            raise ValueError(
                f"metadata.json must contain cameras.cam0 and cameras.cam1, "
                f"got keys: {sorted(cameras.keys())}"
            )
        if "K_processed" not in cameras["cam0"] or "w2c" not in cameras["cam0"]:
            raise ValueError("cam0 missing K_processed or w2c")
        if "K_processed" not in cameras["cam1"] or "w2c" not in cameras["cam1"]:
            raise ValueError("cam1 missing K_processed or w2c")
        rgb_keys = ["cam0"]
        event_key = "cam1"
    cam0 = cameras[rgb_keys[0]]
    cam1 = cameras[event_key]

    frames_all = metadata.get("frames", [])
    if len(frames_all) < 2:
        raise ValueError(
            f"metadata.json must contain at least 2 frames, got {len(frames_all)}"
        )

    if num_source_frames is not None and frame_range is not None:
        raise ValueError(
            "num_source_frames and frame_range are mutually exclusive"
        )

    if frame_range is not None:
        a, b = frame_range
        if not (0 <= a < b <= len(frames_all)):
            raise ValueError(
                f"frame_range {frame_range} is out of bounds for "
                f"{len(frames_all)} available frames; expected "
                f"0 <= a < b <= {len(frames_all)}"
            )
        if b - a < 2:
            raise ValueError(
                f"frame_range {frame_range} must span at least 2 frames"
            )
        frames = frames_all[a:b]
    elif num_source_frames is not None:
        if num_source_frames < 2:
            raise ValueError(
                f"num_source_frames must be >= 2, got {num_source_frames}"
            )
        if num_source_frames > len(frames_all):
            raise ValueError(
                f"num_source_frames={num_source_frames} exceeds available "
                f"frames {len(frames_all)}"
            )
        frames = frames_all[:num_source_frames]
    else:
        frames = frames_all

    T_src = len(frames)

    image_size = cam0.get("image_size")
    if image_size is None or len(image_size) != 2:
        raise ValueError(f"cam0.image_size missing or malformed: {image_size}")
    frame_w, frame_h = int(image_size[0]), int(image_size[1])

    dt = float(frames[1]["timestamp_s"]) - float(frames[0]["timestamp_s"])
    if dt <= 0:
        raise ValueError(
            f"non-monotonic timestamps: frame0={frames[0]['timestamp_s']}, "
            f"frame1={frames[1]['timestamp_s']}"
        )
    native_fps = 1.0 / dt


    max_drift = 0.0
    for i in range(2, len(frames)):
        dt_i = float(frames[i]["timestamp_s"]) - float(frames[i - 1]["timestamp_s"])
        max_drift = max(max_drift, abs(dt_i - dt))
    if max_drift > 1e-6:
        print(
            f"  [warn] inter-frame timestamp drift up to {max_drift:.6f}s "
            f"(dt0={dt:.6f}s) — assuming uniform native_fps={native_fps:.3f}"
        )

    start_time_s = float(frames[0]["timestamp_s"])

    return {
        "cam0": cam0,
        "cam1": cam1,
        "rgb_cams": [cameras[k] for k in rgb_keys],
        "rgb_keys": rgb_keys,
        "event_cam": cam1,
        "event_key": event_key,
        "frames": frames,
        "T_src": T_src,
        "frame_h": frame_h,
        "frame_w": frame_w,
        "native_fps": native_fps,
        "start_time_s": start_time_s,
    }


def _load_rgb_pil_list(
    scene_dir: str,
    frames: list,
    expected_w: int,
    expected_h: int,
) -> list:
    """Load source RGB frames listed in metadata.json into PIL Images.

    Each frame entry has ``rgb_path`` stored with Windows-flavored
    backslashes (``rgb_cam0\\0000.png``). We normalize to forward
    slashes and resolve relative to ``scene_dir``.

    Args:
        scene_dir: Scene root directory.
        frames: List of frame dicts, each with ``rgb_path`` key.
        expected_w: Expected image width (sanity check).
        expected_h: Expected image height (sanity check).

    Returns:
        List of ``len(frames)`` PIL.Image.Image (mode RGB).

    Raises:
        FileNotFoundError: if any PNG is missing.
        ValueError: if any PNG's size disagrees with the expected
            (``expected_w``, ``expected_h``) tuple.
    """
    rgb_list = []
    for i, frame in enumerate(frames):
        rgb_rel = frame.get("rgb_path")
        if rgb_rel is None:


            rgb_paths = frame.get("rgb_paths")
            if not rgb_paths:
                raise ValueError(f"frame {i} has no rgb_path / rgb_paths")
            rgb_rel = rgb_paths[0]
        rgb_rel = rgb_rel.replace("\\", "/")
        rgb_abs = os.path.join(scene_dir, rgb_rel)
        if not os.path.exists(rgb_abs):
            raise FileNotFoundError(f"rgb frame missing: {rgb_abs}")
        img = Image.open(rgb_abs).convert("RGB")
        if img.size != (expected_w, expected_h):
            raise ValueError(
                f"rgb frame {rgb_abs} has size {img.size}, expected "
                f"({expected_w}, {expected_h})"
            )
        rgb_list.append(img)
    return rgb_list


def _load_rgb_pil_lists(
    scene_dir: str,
    frames: list,
    n_rgb_cams: int,
    expected_w: int,
    expected_h: int,
) -> list[list]:
    """Multi-rgb variant: returns ``n_rgb_cams`` parallel PIL lists, one per
    rgb cam, each of length ``len(frames)``. Each frame entry must carry
    ``rgb_paths`` = list of length ``n_rgb_cams``.
    """
    out = [[] for _ in range(n_rgb_cams)]
    for i, frame in enumerate(frames):
        rgb_paths = frame.get("rgb_paths")
        if rgb_paths is None or len(rgb_paths) != n_rgb_cams:
            raise ValueError(
                f"frame {i}: expected rgb_paths of length {n_rgb_cams}, got "
                f"{None if rgb_paths is None else len(rgb_paths)}"
            )
        for j, rel in enumerate(rgb_paths):
            rel = rel.replace("\\", "/")
            abs_p = os.path.join(scene_dir, rel)
            if not os.path.exists(abs_p):
                raise FileNotFoundError(f"rgb frame missing: {abs_p}")
            img = Image.open(abs_p).convert("RGB")
            if img.size != (expected_w, expected_h):
                raise ValueError(
                    f"rgb frame {abs_p} has size {img.size}, expected "
                    f"({expected_w}, {expected_h})"
                )
            out[j].append(img)
    return out


def _load_event_pil_list(
    scene_dir: str,
    frame_w: int,
    frame_h: int,
    target_T: int,
    target_fps: float,
    *,
    start_time_s: float = 0.0,
    event_window_n: int = 1,
) -> list:
    """Build a list of ``target_T`` event-grid frames from raw events.txt.

    Reads ``<scene_dir>/events_cam1.txt`` (raw v2e-format ``t x y p``
    lines), bins it into ``target_T`` windows of width
    ``1 / target_fps``, packs each window's six temporal bins into a
    2x3 grid via ``events_to_grid_image``, then collapses the grid
    back to a single H x W RGB PIL image using the same
    bin-5 -> R, bin-3 -> G, bin-0 -> B mapping the existing dataset
    uses (see ``DFuseValDataset._parse_event_grid_to_rgb``).
    This way the model sees the same per-frame appearance the
    training-time pipeline produces.

    Args:
        scene_dir: Scene root directory containing events_cam1.txt.
        frame_w, frame_h: Per-bin dimensions (the grid will be
            ``2 * frame_h`` x ``3 * frame_w``).
        target_T: Number of output frames.
        target_fps: Frame rate at which the event windows are sliced.
        event_window_n: Accumulate only the last ``1/N`` portion of
            each per-frame event interval. ``N=1`` (default) keeps the
            full window; ``N=2`` keeps the later half; ``N=4`` keeps
            the last quarter, etc. Must be a positive integer.

    Returns:
        List of ``target_T`` PIL.Image.Image (mode RGB) at
        (``frame_w``, ``frame_h``).

    Raises:
        FileNotFoundError: if events_cam1.txt is missing.
    """
    import polars as pl
    from dfuse.utils.events.raw_event_io import (
        events_to_grid_image,
        iter_event_slices,
        load_events_synthetic,
    )

    if not isinstance(event_window_n, (int, np.integer)) or event_window_n < 1:
        raise ValueError(
            f"event_window_n must be a positive integer, got {event_window_n!r}"
        )
    event_window_n = int(event_window_n)

    events_path = os.path.join(scene_dir, "events_cam1.txt")
    if not os.path.exists(events_path):
        raise FileNotFoundError(f"events_cam1.txt not found at {events_path}")

    events_df = load_events_synthetic(events_path)


    if start_time_s > 0.0:
        end_time_s = start_time_s + target_T / target_fps
        events_df = events_df.filter(
            (pl.col("t") >= start_time_s) & (pl.col("t") < end_time_s)
        ).with_columns((pl.col("t") - start_time_s).alias("t"))

    bh, bw = frame_h, frame_w


    frame_dt = 1.0 / target_fps
    keep_threshold = frame_dt * (event_window_n - 1) / event_window_n
    pil_list = []
    empty_count = 0
    for slice_arr in iter_event_slices(events_df, target_T, target_fps):
        if event_window_n > 1 and len(slice_arr) > 0:
            slice_arr = slice_arr[slice_arr[:, 0] >= keep_threshold]
        if len(slice_arr) == 0:
            empty_count += 1
        grid = events_to_grid_image(slice_arr, bh, bw, num_bins=6)
        gray = grid[..., 0]
        bin_0 = gray[0:bh, 0:bw]
        bin_3 = gray[bh : 2 * bh, 0:bw]
        bin_5 = gray[bh : 2 * bh, 2 * bw : 3 * bw]
        rgb = np.stack([bin_5, bin_3, bin_0], axis=-1)
        pil_list.append(Image.fromarray(rgb.astype(np.uint8), mode="RGB"))

    if empty_count > target_T // 10:
        print(
            f"  [warn] {empty_count}/{target_T} event slices are empty — "
            f"target_T may exceed events.txt coverage"
        )
    return pil_list


def _build_real_world_camera_tensors(
    cam0: dict,
    cam1: dict,
    target_K: np.ndarray,
    target_E: np.ndarray,
    T_src: int,
    target_T: int,
    scene_scale: float,
    normalize_to_target_frame: bool,
    *,
    rgb_cams: list[dict] | None = None,
) -> dict:
    """Build per-frame camera tensors for the real-world inference pipeline.

    Both source cameras are static — their per-frame extrinsics are
    just the metadata.json ``w2c`` matrix replicated. The target is
    static too (per design Q2 (a)). Source intrinsics likewise
    replicated. The two source views have different lengths
    (``T_src`` for rgb, ``target_T`` for events) because they live at
    different effective frame rates.

    The ``scene_scale`` and ``normalize_to_target_frame`` operations
    mirror what ``DFuseValDataset.__getitem__`` does at
    lines 326-337 for synthetic data: scale translation columns by
    ``scene_scale``, then normalize so target_ext[0] = I.

    Args:
        cam0: dict with K_processed (3x3 list) + w2c (4x4 list) for
            the rgb source camera.
        cam1: dict with K_processed + w2c for the events source camera.
        target_K: numpy (3, 3) — user-supplied target intrinsics.
        target_E: numpy (4, 4) — user-supplied target w2c.
        T_src: number of source RGB frames (= rgb stream length).
        target_T: number of target frames (= events stream length).
        scene_scale: float, applied to all translations.
        normalize_to_target_frame: if True, multiply by inv(target[0])
            on the right so target[0] becomes the identity.

    Returns:
        Dict matching the ``camera_data`` block in
        ``DFuseValDataset.__getitem__``:
        - ``target_intrinsics``: torch.float32, shape (target_T, 3, 3)
        - ``target_extrinsics``: torch.float32, shape (target_T, 4, 4)
        - ``source_intrinsics``: torch.float32, shape (T_src + target_T, 3, 3)
        - ``source_extrinsics``: torch.float32, shape (T_src + target_T, 4, 4)
        - ``source_view_T_video``: ``[T_src, target_T]``
    """
    rgb_list = rgb_cams if rgb_cams else [cam0]
    target_K = np.asarray(target_K, dtype=np.float32)
    target_E = np.asarray(target_E, dtype=np.float32)
    target_int = np.broadcast_to(target_K, (target_T, 3, 3)).copy()
    target_ext = np.broadcast_to(target_E, (target_T, 4, 4)).copy()

    src_int_blocks = []
    src_ext_blocks = []
    src_view_T = []
    for cam in rgb_list:
        K_i = np.asarray(cam["K_processed"], dtype=np.float32)
        E_i = np.asarray(cam["w2c"], dtype=np.float32)
        src_int_blocks.append(np.broadcast_to(K_i, (T_src, 3, 3)).copy())
        src_ext_blocks.append(np.broadcast_to(E_i, (T_src, 4, 4)).copy())
        src_view_T.append(T_src)

    cam1_K = np.asarray(cam1["K_processed"], dtype=np.float32)
    cam1_E = np.asarray(cam1["w2c"], dtype=np.float32)
    src_int_blocks.append(np.broadcast_to(cam1_K, (target_T, 3, 3)).copy())
    src_ext_blocks.append(np.broadcast_to(cam1_E, (target_T, 4, 4)).copy())
    src_view_T.append(target_T)

    src_int = np.concatenate(src_int_blocks, axis=0)
    src_ext = np.concatenate(src_ext_blocks, axis=0)

    target_ext_t = torch.from_numpy(target_ext)
    src_ext_t = torch.from_numpy(src_ext)

    if scene_scale != 1.0:
        target_ext_t[:, :3, 3] *= scene_scale
        src_ext_t[:, :3, 3] *= scene_scale

    if normalize_to_target_frame:
        target_ext_t, src_ext_t = (
            MultiViewPairProcessor._normalize_to_target_frame(target_ext_t, src_ext_t)
        )

    return {
        "target_intrinsics": torch.from_numpy(target_int),
        "target_extrinsics": target_ext_t,
        "source_intrinsics": torch.from_numpy(src_int),
        "source_extrinsics": src_ext_t,
        "source_view_T_video": src_view_T,
    }


def load_real_world_scene(
    scene_dir: str,
    target_intrinsics: np.ndarray,
    target_extrinsics: np.ndarray,
    upsample_factor: int,
    prompt: str,
    *,
    num_source_frames: int | None = None,
    frame_range: tuple[int, int] | None = None,
    scene_scale: float = 1.0,
    normalize_to_target_frame: bool = True,
    event_window_n: int = 1,
) -> dict:
    """Load a real-world capture for single-scene DFuse inference.

    Returns a dict with the same keys as
    ``DFuseValDataset.__getitem__`` so the standalone inference
    script can drive the pipeline without adapting its call site.

    Args:
        scene_dir: Scene root containing ``metadata.json``,
            ``rgb_cam0/*.png``, and ``events_cam1.txt``.
        target_intrinsics: numpy (3, 3) — user-supplied target K.
        target_extrinsics: numpy (4, 4) — user-supplied target w2c
            in the same coordinate frame as ``cam0.w2c`` /
            ``cam1.w2c`` in metadata.json (post-scene-normalization).
        upsample_factor: integer >= 1. Target output is
            ``T_src * upsample_factor`` frames at
            ``native_fps * upsample_factor`` fps.
        prompt: caption fed into the diffusion pipeline.
        num_source_frames: optional cap on T_src (head of frames).
            Mutually exclusive with ``frame_range``.
        frame_range: optional ``(a, b)`` half-open RGB-frame range.
            Events are trimmed to the matching time window and
            rebased so the cropped clip starts at t=0.
        scene_scale: applied to translation columns of all extrinsics.
        normalize_to_target_frame: if True, normalize so
            target_extrinsics[0] = I.
        event_window_n: Accumulate only the last ``1/N`` of each
            per-frame event interval. ``N=1`` (default) keeps the full
            interval; ``N=2`` keeps the later half; ``N=4`` keeps the
            last quarter, etc. Must be a positive integer.

    Returns:
        Dict with keys: sample_name, case_tag, target_cam_id,
        source_views, gt_frames, gt_mp4_path, camera_data, prompt,
        target_fps, source_fps_list, num_frames,
        source_modality_indices, source_view_camera_ids, frame_h,
        frame_w, source_depth.

    Raises:
        FileNotFoundError, ValueError as described in the helpers.
    """
    if not isinstance(upsample_factor, (int, np.integer)) or upsample_factor < 1:
        raise ValueError(
            f"upsample_factor must be a positive integer, got {upsample_factor!r}"
        )
    upsample_factor = int(upsample_factor)

    target_K = np.asarray(target_intrinsics, dtype=np.float32)
    if target_K.shape != (3, 3):
        raise ValueError(
            f"target_intrinsics must have shape (3, 3), got {target_K.shape}"
        )
    target_E = np.asarray(target_extrinsics, dtype=np.float32)
    if target_E.shape != (4, 4):
        raise ValueError(
            f"target_extrinsics must have shape (4, 4), got {target_E.shape}"
        )
    if not np.allclose(target_E[3], np.array([0, 0, 0, 1], dtype=np.float32), atol=1e-5):
        raise ValueError(
            f"target_extrinsics last row must be [0, 0, 0, 1] (homogeneous), "
            f"got {target_E[3]}"
        )

    meta = _read_real_world_metadata(
        scene_dir, num_source_frames, frame_range=frame_range,
    )

    T_src = meta["T_src"]


    target_T = (T_src - 1) * upsample_factor + 1
    if (target_T - 1) % 4 != 0:
        raise ValueError(
            f"target_T={target_T} (from T_src={T_src}, upsample={upsample_factor}) "
            f"violates Wan VAE constraint (target_T - 1) % 4 == 0. Choose an "
            f"upsample_factor that's a multiple of 4, or pick T_src so that "
            f"(T_src - 1) * upsample_factor is a multiple of 4."
        )
    native_fps = meta["native_fps"]
    target_fps = native_fps * upsample_factor

    n_rgb = len(meta.get("rgb_cams", []) or [meta["cam0"]])
    if n_rgb > 1:
        rgb_pil_lists = _load_rgb_pil_lists(
            scene_dir, meta["frames"], n_rgb, meta["frame_w"], meta["frame_h"],
        )
    else:
        rgb_pil_lists = [_load_rgb_pil_list(
            scene_dir, meta["frames"], meta["frame_w"], meta["frame_h"],
        )]

    event_pil_list = _load_event_pil_list(
        scene_dir, meta["frame_w"], meta["frame_h"], target_T, target_fps,
        start_time_s=meta["start_time_s"],
        event_window_n=event_window_n,
    )

    cam_data = _build_real_world_camera_tensors(
        cam0=meta["cam0"], cam1=meta["cam1"],
        target_K=target_K, target_E=target_E,
        T_src=T_src, target_T=target_T,
        scene_scale=scene_scale,
        normalize_to_target_frame=normalize_to_target_frame,
        rgb_cams=meta.get("rgb_cams"),
    )

    source_views = list(rgb_pil_lists) + [event_pil_list]
    source_modality_indices = [0] * n_rgb + [1]
    source_fps_list = [native_fps] * n_rgb + [target_fps]
    source_view_camera_ids = list(meta.get("rgb_keys", ["cam0"])) + [meta.get("event_key", "cam1")]

    return {
        "sample_name": os.path.basename(os.path.normpath(scene_dir)),
        "case_tag": f"x{upsample_factor}",
        "target_cam_id": "target",
        "source_views": source_views,
        "gt_frames": [],
        "gt_mp4_path": None,
        "camera_data": cam_data,
        "prompt": prompt,
        "target_fps": target_fps,
        "source_fps_list": source_fps_list,
        "num_frames": target_T,
        "source_modality_indices": source_modality_indices,
        "source_view_camera_ids": source_view_camera_ids,
        "frame_h": meta["frame_h"],
        "frame_w": meta["frame_w"],
        "source_depth": None,
    }


def merge_validation_results(per_rank: list[dict]) -> dict:
    """Merge per-rank validation dicts; drop entries flagged `is_redundant=True`.

    Args:
        per_rank: List of per-rank dicts from `run_validation`.
            Each dict has `generated_videos`, `gt_videos`, `vis_video_paths`,
            `prompts`, `is_redundant`, and `counters`.

    Returns:
        Single merged dict with `generated_videos`, `gt_videos`,
        `vis_video_paths`, `prompts`, `counters` (all redundant entries
        removed; `None` entries in `vis_video_paths` also removed).
    """
    if not per_rank:
        return {"generated_videos": [], "gt_videos": [], "vis_video_paths": [], "prompts": [], "counters": {}}

    ref = per_rank[0]
    counter_keys = list(ref.get("counters", {}).keys())

    all_generated = []
    all_gt = []
    all_vis_videos = []
    all_prompts = []
    all_redundant = []
    counters_sum = {k: 0 for k in counter_keys}

    for rank_idx, r in enumerate(per_rank):
        is_red = r.get("is_redundant", [])
        n = len(is_red)

        gen_videos = r.get("generated_videos", [])
        gt_videos = r.get("gt_videos", [])
        vis_paths = r.get("vis_video_paths", [])
        prompts = r.get("prompts", [])

        if len(gen_videos) != n:
            raise ValueError(
                f"rank {rank_idx}: generated_videos has length {len(gen_videos)} "
                f"but is_redundant has length {n}"
            )
        if len(gt_videos) != n:
            raise ValueError(
                f"rank {rank_idx}: gt_videos has length {len(gt_videos)} "
                f"but is_redundant has length {n}"
            )
        if len(vis_paths) != n:
            raise ValueError(
                f"rank {rank_idx}: vis_video_paths has length {len(vis_paths)} "
                f"but is_redundant has length {n}"
            )
        if len(prompts) != n:
            raise ValueError(
                f"rank {rank_idx}: prompts has length {len(prompts)} "
                f"but is_redundant has length {n}"
            )

        all_generated.extend(gen_videos)
        all_gt.extend(gt_videos)
        all_vis_videos.extend(vis_paths)
        all_prompts.extend(prompts)
        all_redundant.extend(is_red)

        for k in counter_keys:
            v = r["counters"][k]
            if not isinstance(v, int) or isinstance(v, bool):
                raise ValueError(
                    f"rank {rank_idx}: counters[{k!r}] must be int, "
                    f"got {type(v).__name__}"
                )
            counters_sum[k] += v


    keep = [not red for red in all_redundant]
    merged_gen = [d for d, k in zip(all_generated, keep) if k]
    merged_gt = [d for d, k in zip(all_gt, keep) if k]
    merged_vis = [p for p, k in zip(all_vis_videos, keep) if k and p is not None]
    merged_prompts = [p for p, k in zip(all_prompts, keep) if k]

    return {
        "generated_videos": merged_gen,
        "gt_videos": merged_gt,
        "vis_video_paths": merged_vis,
        "prompts": merged_prompts,
        "counters": counters_sum,
    }


def _save_frames_lossless_mkv(frames, path, fps):
    """Save PIL frames as mathematically lossless FFV1 (.mkv) via PyAV.

    Uses `bgr0` (8-bit packed R/G/B with an ignored alpha byte) for the
    encoded stream. `rgb24 -> bgr0` via libswscale is a bit-exact byte
    rearrangement (same bit depth, same color space, no subsampling, no
    scaling), so the pipeline is lossless end-to-end: PIL RGB -> rgb24 ->
    bgr0 -> FFV1 -> decoded RGB -> numpy RGB. (Plain 8-bit `gbrp` is not
    available in every FFmpeg build's FFV1 encoder; `bgr0` is.)

    Args:
        frames: Iterable of PIL.Image (RGB, uint8) frames.
        path: Output .mkv path.
        fps: Frame rate for the output stream. Int, float, or Fraction; a
            float (e.g. from torchcodec's `average_fps`) is converted to a
            rational here because PyAV's `add_stream(rate=...)` does not
            accept float.
    """
    import av
    from fractions import Fraction

    first = np.array(frames[0])
    H, W = first.shape[:2]

    rate = fps if isinstance(fps, (int, Fraction)) else Fraction(fps).limit_denominator(1_000_000)

    container = av.open(path, mode="w")
    stream = container.add_stream("ffv1", rate=rate)
    stream.width = W
    stream.height = H
    stream.pix_fmt = "bgr0"

    for frame in frames:
        arr = np.array(frame)
        vf = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in stream.encode(vf):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _is_valid_video(path, min_frames):
    """Open `path` with torchcodec; True iff metadata reports >= min_frames frames."""
    if not path or not os.path.exists(path):
        return False
    try:
        from torchcodec.decoders import VideoDecoder

        dec = VideoDecoder(path)
        n = dec.metadata.num_frames
        del dec
        return n is not None and n >= min_frames
    except Exception:
        return False


def _find_existing_output(output_dir, subfolder, case_id, ext):
    """Search every rank_XX subdir under {output_dir}/{subfolder} for {case_id}.{ext}."""
    pattern = os.path.join(output_dir, subfolder, "rank_*", f"{case_id}.{ext}")
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def run_validation(
    pipe,
    accelerator,
    output_dir: str,
    args=None,
):
    """Validation for DFuse training.

    Generates videos from the validation dataset, saves generated frames and
    comparison videos to disk. Returns path lists for rank-0 metric computation.
    """
    from tqdm import tqdm


    enable_frr = [
        v.strip().lower() == "true" for v in args.enable_frame_rate_reduction.split(",")
    ]


    val_dataset = DFuseValDataset(
        base_path=args.val_dataset_base_path,
        source_camera_indices=getattr(args, "source_camera_indices", None),
        source_camera_modalities=getattr(args, "source_camera_modalities", None),
        source_modalities=getattr(args, "source_modalities", None),
        enable_frr=enable_frr,
        args=args,
        val_num_frames=getattr(args, "val_num_frames", None),
        frr_stride=getattr(args, "val_frr_stride", None),
        max_scenes=getattr(args, "val_max_scenes", None),
    )

    is_main = accelerator.is_main_process
    M = len(val_dataset)
    val_num_samples = getattr(args, "val_num_samples", -1)
    if val_num_samples >= 0 and val_num_samples < M:
        M = val_num_samples
    world_size = accelerator.num_processes
    rank = accelerator.process_index


    if M == 0:
        N_per_rank = 0
        local_indices = []
        is_redundant_flags = []
    else:
        N_per_rank = math.ceil(M / world_size)
        padded_len = N_per_rank * world_size
        all_indices = [i % M for i in range(padded_len)]
        local_indices = all_indices[rank * N_per_rank : (rank + 1) * N_per_rank]
        is_redundant_flags = [(rank * N_per_rank + i) >= M for i in range(N_per_rank)]

    if is_main:
        print(
            f"\n[Validation] {M} cases from {args.val_dataset_base_path}"
        )
        print(
            f"  sharded across {world_size} ranks: {N_per_rank} cases/rank "
            f"(padding: {N_per_rank * world_size - M} duplicates)"
        )


    training_module_names = {
        name for name, module in pipe.dit.named_modules() if module.training
    }
    pipe.dit.eval()

    generated_videos_list = []
    gt_videos_list = []
    vis_video_list = []
    prompts_list = []
    is_redundant_list = []
    num_errors = 0

    with torch.no_grad():
        for local_i, case_idx in enumerate(
            tqdm(
                local_indices,
                desc=f"Validation rank {rank}",
                disable=not is_main,
            )
        ):
            sample_is_redundant = is_redundant_flags[local_i]


            if getattr(args, "val_resume", False):
                seq_id_r, case_tag_r, target_cam_r, _ = val_dataset.cases[case_idx]
                case_id_r = f"{seq_id_r}_{case_tag_r}"
                info_r = val_dataset.seq_info[seq_id_r]
                T_r = info_r["num_frames"]
                if val_dataset.val_num_frames is not None:
                    T_r = min(T_r, val_dataset.val_num_frames)

                gen_existing = _find_existing_output(output_dir, "generated", case_id_r, "mkv")
                vis_existing = _find_existing_output(output_dir, "comparison", case_id_r, "mp4")

                if (
                    _is_valid_video(gen_existing, T_r)
                    and _is_valid_video(vis_existing, T_r)
                ):
                    gt_mp4_r = os.path.join(
                        val_dataset.base_path, seq_id_r, f"images_{target_cam_r}.mp4"
                    )
                    prompt_r = val_dataset.db.get_prompt(seq_id_r, target_cam_r) or ""
                    if sample_is_redundant:
                        generated_videos_list.append("")
                        gt_videos_list.append((None, 0))
                        vis_video_list.append(None)
                    else:
                        generated_videos_list.append(gen_existing)
                        gt_videos_list.append((gt_mp4_r, T_r))
                        vis_video_list.append(vis_existing)
                    prompts_list.append(prompt_r)
                    is_redundant_list.append(sample_is_redundant)
                    continue

            try:
                item = val_dataset[case_idx]
                sample_name = item["sample_name"]
                case_tag = item["case_tag"]
                case_id = f"{sample_name}_{case_tag}"
                src_views = item["source_views"]
                gt_frames = item["gt_frames"]
                gt_mp4_path = item["gt_mp4_path"]
                cam_kwargs = item["camera_data"]
                prompt = item["prompt"]
                fps = item["target_fps"]
                src_fps_list = item["source_fps_list"]
                T = item["num_frames"]
                src_mod_indices = item["source_modality_indices"]
                frame_h = item["frame_h"]
                frame_w = item["frame_w"]

                if not src_views:
                    if is_main:
                        print(f"    Warning: no source views for {case_id}, skipping")
                    generated_videos_list.append("")
                    gt_videos_list.append((gt_mp4_path, T))
                    vis_video_list.append(None)
                    prompts_list.append(prompt)
                    is_redundant_list.append(sample_is_redundant)
                    continue


                pipe_cam_kwargs = {}
                if cam_kwargs:
                    pipe_cam_kwargs = {
                        "target_intrinsics": [cam_kwargs["target_intrinsics"]],
                        "target_extrinsics": [cam_kwargs["target_extrinsics"]],
                        "source_intrinsics": [cam_kwargs["source_intrinsics"]],
                        "source_extrinsics": [cam_kwargs["source_extrinsics"]],
                        "source_view_T_video": [cam_kwargs["source_view_T_video"]],
                    }

                generated_frames = pipe(
                    prompt=prompt,
                    negative_prompt="",
                    source_video=[src_views],
                    heights=frame_h,
                    widths=frame_w,
                    num_frames=T,
                    seed=0,
                    cfg_scale=args.val_cfg_scale,
                    cfg_drop_source=args.val_cfg_drop_source,
                    tiled=False,
                    num_inference_steps=args.val_num_inference_steps,
                    target_fps=fps,
                    source_view_fps=[src_fps_list],
                    target_modality=[0],
                    source_modality_list=[src_mod_indices],
                    progress_bar_cmd=tqdm if is_main else (lambda x, **kw: x),
                    **pipe_cam_kwargs,
                )[0]


                gen_video_dir = os.path.join(output_dir, "generated", f"rank_{rank:02d}")
                gen_video_path = os.path.join(gen_video_dir, f"{case_id}.mkv")

                if sample_is_redundant:
                    gen_video_path = None
                    gt_mp4_path = None
                else:
                    os.makedirs(gen_video_dir, exist_ok=True)
                    _save_frames_lossless_mkv(generated_frames, gen_video_path, fps)


                    combined_frames = []
                    for frame_idx in range(T):
                        parts = []
                        for view_frames in src_views:
                            n_v = len(view_frames)
                            v_idx = min(
                                int(round(frame_idx * (n_v - 1) / max(T - 1, 1))),
                                n_v - 1,
                            )
                            parts.append(np.array(view_frames[v_idx]))
                        parts.append(np.array(generated_frames[frame_idx]))
                        parts.append(np.array(gt_frames[frame_idx]))
                        combined_frames.append(
                            Image.fromarray(np.concatenate(parts, axis=1))
                        )

                    video_dir = os.path.join(output_dir, "comparison", f"rank_{rank:02d}")
                    os.makedirs(video_dir, exist_ok=True)
                    video_path = os.path.join(video_dir, f"{case_id}.mp4")
                    save_video(combined_frames, video_path, fps=fps, quality=8)

                generated_videos_list.append(gen_video_path)
                gt_videos_list.append((gt_mp4_path, T) if gt_mp4_path is not None else (None, 0))
                vis_video_list.append(video_path if not sample_is_redundant else None)
                prompts_list.append(prompt)
                is_redundant_list.append(sample_is_redundant)

            except Exception as e:
                if is_main:
                    print(f"[Validation] Error on case {local_i}: {e}")
                    import traceback

                    traceback.print_exc()
                num_errors += 1
                generated_videos_list.append("")
                gt_videos_list.append(("", 0))
                vis_video_list.append(None)
                prompts_list.append("")
                is_redundant_list.append(sample_is_redundant)


    pipe.scheduler.set_timesteps(1000, training=True)
    for name, module in pipe.dit.named_modules():
        if name in training_module_names:
            module.train()

    if is_main:
        print(f"[Validation] rank {rank} - num_errors: {num_errors}")

    return {
        "generated_videos": generated_videos_list,
        "gt_videos": gt_videos_list,
        "vis_video_paths": vis_video_list,
        "prompts": prompts_list,
        "is_redundant": is_redundant_list,
        "counters": {"num_errors": int(num_errors)},
    }


def compute_dfuse_metrics(
    gathered_result,
    device,
    args=None,
    metrics: list[str] | None = None,
):
    """Rank-0 metric callback. Computes any subset of {psnr, fid, fvd, clip_t, clip_f}.

    Args:
        gathered_result: Already-deduped dict from `merge_validation_results`
            with keys `generated_videos`, `gt_videos`, `vis_video_paths`,
            `prompts`, `counters`.
        device: Torch device for FID/FVD/CLIP computation.
        args: Training/inference args object. CLIP reads optional
            `clip_model` (default "ViT-B/32") and `clip_batch_size`
            (default 32) attributes off it.
        metrics: Subset of `{"psnr", "fid", "fvd", "clip_t", "clip_f"}` to
            compute. None means compute all five.

    Returns:
        Dict with one entry per requested metric. Missing keys mean the
        metric was either not requested or could not be computed because
        no valid pairs were available.

    Raises:
        ValueError: if `metrics` contains an unknown name.
    """
    from dfuse.metrics import (
        compute_clip_f_from_videos,
        compute_clip_t_from_videos,
        compute_fid_from_videos,
        compute_fvd_from_videos,
        compute_psnr_from_videos,
    )

    allowed = {"psnr", "fid", "fvd", "clip_t", "clip_f"}
    if metrics is None:
        metrics = ["psnr", "fid", "fvd", "clip_t", "clip_f"]
    bad = [m for m in metrics if m not in allowed]
    if bad:
        raise ValueError(f"unknown metric(s): {bad}; allowed: {sorted(allowed)}")

    generated_videos = gathered_result["generated_videos"]
    gt_videos = gathered_result["gt_videos"]
    prompts = gathered_result.get("prompts", [""] * len(generated_videos))

    valid_gen = []
    valid_gt = []
    valid_prompts = []
    valid_num_frames = []
    psnr_values = []
    for gen_mp4, (gt_mp4, n_frames), prompt in zip(generated_videos, gt_videos, prompts):
        if not gen_mp4 or not gt_mp4:
            continue
        if "psnr" in metrics:
            psnr_values.append(compute_psnr_from_videos(gen_mp4, gt_mp4, n_frames))
        valid_gen.append(gen_mp4)
        valid_gt.append((gt_mp4, n_frames))
        valid_prompts.append(prompt)
        valid_num_frames.append(n_frames)

    scalars = {}

    if "psnr" in metrics and psnr_values:
        scalars["psnr"] = sum(psnr_values) / len(psnr_values)
        print(f"[Metrics] PSNR: {scalars['psnr']:.2f} dB over {len(psnr_values)} cases")

    if "fid" in metrics and valid_gen and valid_gt:
        print(f"[Metrics] Computing FID over {len(valid_gen)} pairs ...")
        try:
            scalars["fid"] = compute_fid_from_videos(
                valid_gen, valid_gt, device, batch_size=50, num_workers=4,
            )
            print(f"[Metrics] FID: {scalars['fid']:.2f}")
        except Exception as e:
            print(f"[Metrics] FID failed: {e}")

    if "fvd" in metrics and valid_gen and valid_gt:

        n_frames_min = min(t for (_, t) in valid_gt)
        if n_frames_min < 10:
            print(f"[Metrics] FVD skipped: min n_frames={n_frames_min} < 10")
        else:
            print(
                f"[Metrics] Computing FVD over {len(valid_gen)} pairs "
                f"at num_frames={n_frames_min} ..."
            )
            try:
                scalars["fvd"] = compute_fvd_from_videos(
                    valid_gen, valid_gt, device, num_frames=n_frames_min,
                )
                print(f"[Metrics] FVD: {scalars['fvd']:.2f}")
            except Exception as e:
                print(f"[Metrics] FVD failed: {e}")

    clip_model_name = getattr(args, "clip_model", "ViT-B/32") if args else "ViT-B/32"
    clip_batch_size = getattr(args, "clip_batch_size", 32) if args else 32

    if "clip_t" in metrics and valid_gen:
        print(f"[Metrics] Computing CLIP-T over {len(valid_gen)} cases ...")
        try:
            scalars["clip_t"] = compute_clip_t_from_videos(
                valid_gen, valid_prompts, device,
                num_frames=valid_num_frames,
                model_name=clip_model_name,
                batch_size=clip_batch_size,
            )
            print(f"[Metrics] CLIP-T: {scalars['clip_t']:.4f}")
        except Exception as e:
            print(f"[Metrics] CLIP-T failed: {e}")

    if "clip_f" in metrics and valid_gen:
        print(f"[Metrics] Computing CLIP-F over {len(valid_gen)} cases ...")
        try:
            scalars["clip_f"] = compute_clip_f_from_videos(
                valid_gen, device,
                num_frames=valid_num_frames,
                model_name=clip_model_name,
                batch_size=clip_batch_size,
            )
            print(f"[Metrics] CLIP-F: {scalars['clip_f']:.4f}")
        except Exception as e:
            print(f"[Metrics] CLIP-F failed: {e}")

    return scalars
