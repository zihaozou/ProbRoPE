"""
Video Dataset for loading stereo/monocular/multiview video data.

Supports the standardized video dataset format with:
- Metadata: SQLite database (metadata.db)
- Images: HDF5 (.h5) or MP4 format (auto-detected)
- Depths/Flows/Masks: HDF5 format
- Events: MP4 format with 2x3 grid layout (6 temporal bins)

Dataset types: monocular, stereo, multiview

Output format:
- Images: List[PIL.Image] (RGB)
- Events: np.ndarray (T, 6, H, W) uint8
- Other data: np.ndarray
"""

import os
import h5py
import torch
import numpy as np
from PIL import Image
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from torchcodec.decoders import VideoDecoder
from tqdm import tqdm
from accelerate.state import PartialState

from dfuse.core.data.processors.base import BaseProcessor
from dfuse.utils.metadata_db import MetadataDB


def _tensor_to_pil_list(tensor: torch.Tensor) -> List[Image.Image]:
    """Convert (N, C, H, W) uint8 tensor to list of PIL RGB images."""

    frames = tensor.permute(0, 2, 3, 1).numpy()
    return [Image.fromarray(frame) for frame in frames]


def _load_mp4_frames(
    decoder: VideoDecoder, start_idx: int, end_idx: int
) -> List[Image.Image]:
    """Load frame range from MP4 using torchcodec, return as PIL images."""
    frames_tensor = decoder[start_idx:end_idx]
    if torch.any(torch.isnan(frames_tensor)):
        raise ValueError(
            f"NaN values found in decoded frames from MP4 {decoder.path} for frames {start_idx}:{end_idx}"
        )
    return _tensor_to_pil_list(frames_tensor)


def _load_hdf5_images(
    h5_path: str, key: str, start_idx: int, end_idx: int
) -> List[Image.Image]:
    """Load frame range from HDF5, return as PIL images."""
    with h5py.File(h5_path, "r") as f:
        frames = f[key][start_idx:end_idx]
    return [Image.fromarray(frame) for frame in frames]


def _load_mp4_frames_tensor(
    decoder: VideoDecoder, start_idx: int, end_idx: int
) -> torch.Tensor:
    """Load frame range from MP4 using torchcodec, return as (N, C, H, W) uint8 tensor."""
    frames_tensor = decoder[start_idx:end_idx]
    if torch.any(torch.isnan(frames_tensor)):
        raise ValueError(
            f"NaN values found in decoded frames from MP4 for frames {start_idx}:{end_idx}"
        )
    return frames_tensor.contiguous()


def _load_hdf5_images_tensor(
    h5_path: str, key: str, start_idx: int, end_idx: int
) -> torch.Tensor:
    """Load frame range from HDF5, return as (N, C, H, W) uint8 tensor."""
    with h5py.File(h5_path, "r") as f:
        frames = f[key][start_idx:end_idx]
    return torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).contiguous()


def _load_hdf5_data(h5_path: str, key: str, start_idx: int, end_idx: int) -> np.ndarray:
    """Load frame range from HDF5, return as numpy array."""
    with h5py.File(h5_path, "r") as f:
        return f[key][start_idx:end_idx]


def _load_event_frames(
    decoder: VideoDecoder, start_idx: int, end_idx: int, height: int, width: int
) -> np.ndarray:
    """
    Load event video and rearrange from 2x3 grid to (T, 6, H, W).

    Event videos are stored with resolution (H*2, W*3) containing 6 temporal bins:
        [bin_0] [bin_1] [bin_2]
        [bin_3] [bin_4] [bin_5]

    Returns:
        np.ndarray: (T, 6, H, W) uint8 where 128=no events, 0=OFF, 255=ON
    """
    frames_tensor = decoder[start_idx:end_idx]

    if frames_tensor.shape[1] == 3:
        frames = frames_tensor[:, 0].numpy()
    else:
        frames = frames_tensor[:, 0].numpy()

    T = frames.shape[0]

    frames = frames.reshape(T, 2, height, 3, width)
    frames = frames.transpose(0, 1, 3, 2, 4)
    frames = frames.reshape(T, 6, height, width)
    return frames


def _normalize_extrinsics(
    extrinsics: np.ndarray,
    reference_extrinsic: np.ndarray,
) -> np.ndarray:
    """
    Normalize extrinsics so that the reference frame is at identity (origin, facing +x).

    The raw extrinsics are in OpenCV's W2C (world-to-camera) format.
    After normalization:
    - The reference frame has identity extrinsic (camera at origin)
    - Other frames maintain their relative positions to the reference
    - The camera convention is changed from OpenCV (looking -z) to facing +x

    Args:
        extrinsics: (T, 4, 4) array of W2C extrinsics for all frames
        reference_extrinsic: (4, 4) array of the reference frame's W2C extrinsic

    Returns:
        normalized_extrinsics: (T, 4, 4) array of normalized extrinsics
    """
    T = extrinsics.shape[0]
    normalized_first_frame_c2w = np.array(
        [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float32
    )

    normalized = np.zeros_like(extrinsics)
    for t in range(T):
        normalized[t] = (
            normalized_first_frame_c2w
            @ reference_extrinsic
            @ np.linalg.inv(extrinsics[t])
        )
    return np.linalg.inv(normalized)


def _normalize_extrinsics_stereo(
    extrinsics_left: np.ndarray,
    extrinsics_right: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize stereo extrinsics with left camera frame 0 as reference."""
    reference = extrinsics_left[0]
    return (
        _normalize_extrinsics(extrinsics_left, reference),
        _normalize_extrinsics(extrinsics_right, reference),
    )


def _normalize_extrinsics_multiview(
    extrinsics_per_camera: Dict[str, np.ndarray],
    reference_camera_id: str = "cam_0",
) -> Dict[str, np.ndarray]:
    """Normalize multiview extrinsics with reference camera frame 0 as reference."""

    ref_cam = reference_camera_id
    if ref_cam not in extrinsics_per_camera:
        for cam_id in sorted(extrinsics_per_camera.keys()):
            if cam_id.startswith("cam_0") or cam_id == "0":
                ref_cam = cam_id
                break
        else:
            ref_cam = sorted(extrinsics_per_camera.keys())[0]

    reference = extrinsics_per_camera[ref_cam][0]
    return {
        cam_id: _normalize_extrinsics(ext, reference)
        for cam_id, ext in extrinsics_per_camera.items()
    }


@dataclass
class VideoDatasetConfig:
    """Configuration for VideoDataset."""

    base_path: str
    metadata_path: str
    repeat: int = 1


class ProcessorContext:
    """Lightweight context passed into processors with convenience loaders."""

    def __init__(self, dataset: "VideoDataset", seq_id: str, seq_meta: Dict[str, Any]):
        self.dataset = dataset
        self.seq_id = seq_id
        self.seq_meta = seq_meta
        self.dataset_type = seq_meta["dataset_type"]
        resolution = seq_meta.get("resolution", [0, 0])
        self.height, self.width = resolution[0], resolution[1]
        self.seq_path = os.path.join(dataset.base_path, seq_id)

    def get_paths(self) -> Dict:
        return self.seq_meta.get("paths", {})

    def get_camera_ids(self) -> List[str]:
        return self.seq_meta.get("camera_ids", [])

    def load_images(
        self, camera_id: Optional[str], start_idx: int, end_idx: int
    ) -> List[Image.Image]:
        return self.dataset._load_images(
            self.seq_path, self.dataset_type, camera_id, start_idx, end_idx
        )

    def load_images_tensor(
        self, camera_id: Optional[str], start_idx: int, end_idx: int
    ) -> torch.Tensor:
        """Load images as (N, C, H, W) uint8 tensor for efficient IPC transfer."""
        return self.dataset._load_images_tensor(
            self.seq_path, self.dataset_type, camera_id, start_idx, end_idx
        )

    def load_events(
        self, camera_id: Optional[str], start_idx: int, end_idx: int
    ) -> Optional[np.ndarray]:
        return self.dataset._load_events(
            self.seq_path,
            self.dataset_type,
            camera_id,
            start_idx,
            end_idx,
            self.height,
            self.width,
        )

    def load_hdf5_slice(
        self, relative_path: str, key: str, start_idx: int, end_idx: int
    ) -> np.ndarray:
        h5_path = os.path.join(self.dataset.base_path, relative_path)
        return _load_hdf5_data(h5_path, key, start_idx, end_idx)

    def load_flow_slice(
        self,
        relative_path: str,
        key: str,
        start_idx: int,
        end_idx: int,
        is_forward: bool,
    ) -> np.ndarray:
        h5_path = os.path.join(self.dataset.base_path, relative_path)
        data = self.dataset._load_flow_data(
            h5_path, [key], start_idx, end_idx, is_forward
        )
        return data[key]

    def normalize_extrinsics(
        self, extrinsics: np.ndarray, reference: np.ndarray
    ) -> np.ndarray:
        return _normalize_extrinsics(extrinsics, reference)

    def normalize_extrinsics_multiview(
        self,
        extrinsics_per_camera: Dict[str, np.ndarray],
        reference_camera_id: str = "cam_0",
    ) -> Dict[str, np.ndarray]:
        return _normalize_extrinsics_multiview(
            extrinsics_per_camera, reference_camera_id
        )

    def get_camera_params(
        self, camera_id: str, start_idx: int, end_idx: int
    ) -> Dict[str, np.ndarray]:
        """Query camera parameters from database for specific camera and frame range.

        Args:
            camera_id: Camera identifier (e.g., 'data', 'left', 'right', 'cam_0')
            start_idx: Start frame index (inclusive)
            end_idx: End frame index (exclusive)

        Returns:
            Dictionary with:
                - 'intrinsics': (T, 3, 3) float32 array
                - 'extrinsics': (T, 4, 4) float32 array
        """
        params = self.dataset._metadata_db.get_camera_params(
            self.seq_id,
            camera_id=camera_id,
            frame_idx_start=start_idx,
            frame_idx_end=end_idx,
        )

        if not params:
            raise ValueError(
                f"No camera params found for seq_id={self.seq_id}, camera_id={camera_id}"
            )


        params_by_frame = {p["frame_idx"]: p for p in params}

        intrinsics_list = []
        extrinsics_list = []
        for frame_idx in range(start_idx, end_idx):
            frame_params = params_by_frame.get(frame_idx)
            if frame_params is None:
                raise ValueError(
                    f"Missing camera params for seq_id={self.seq_id}, "
                    f"camera_id={camera_id}, frame_idx={frame_idx}"
                )
            intrinsics_list.append(frame_params["intrinsics"])
            extrinsics_list.append(frame_params["extrinsics"])

        return {
            "intrinsics": np.stack(intrinsics_list, axis=0),
            "extrinsics": np.stack(extrinsics_list, axis=0),
        }

    def get_prompt(self, camera_id: str) -> Optional[str]:
        """Query prompt from database for specific camera.

        Args:
            camera_id: Camera identifier (e.g., 'data', 'left', 'right', 'cam_0')

        Returns:
            Prompt string, or None if no prompt exists
        """
        return self.dataset._metadata_db.get_prompt(self.seq_id, camera_id)

    def get_fixed_camera_ids(self) -> List[str]:
        """Return camera IDs classified as 'fixed' for this sequence.

        Queries the ``camera_types`` table populated by the blocking-detection
        script.  Returns an empty list if the table does not exist or has no
        entries for this sequence (caller should treat all cameras as fixed).
        """
        conn = self.dataset._metadata_db.conn
        try:
            rows = conn.execute(
                "SELECT camera_id FROM camera_types "
                "WHERE seq_id = ? AND camera_type = 'fixed'",
                (self.seq_id,),
            ).fetchall()
            return [r["camera_id"] if isinstance(r, dict) else r[0] for r in rows]
        except Exception:
            return []

    def get_valid_frame_intervals(
        self, camera_id: str, min_length: int = 1
    ) -> List[Tuple[int, int]]:
        """Return contiguous intervals of valid (unblocked) frames.

        Queries the ``fixed_camera_valid_frames`` table.  Each returned tuple
        is ``(start, end)`` where *start* is inclusive and *end* is exclusive,
        and the interval length is ``>= min_length``.

        Falls back to ``[(0, num_frames)]`` if the table does not exist or
        has no entries for this sequence/camera (i.e. camera types have not
        been annotated, so all frames are assumed valid).
        """
        num_frames = self.seq_meta["num_frames"]
        conn = self.dataset._metadata_db.conn
        try:
            rows = conn.execute(
                "SELECT frame_idx FROM fixed_camera_valid_frames "
                "WHERE seq_id = ? AND camera_id = ? AND is_valid = 1 "
                "ORDER BY frame_idx",
                (self.seq_id, camera_id),
            ).fetchall()
        except Exception:

            return [(0, num_frames)]

        if not rows:


            return [(0, num_frames)]

        valid_set = {r["frame_idx"] if isinstance(r, dict) else r[0] for r in rows}


        intervals: List[Tuple[int, int]] = []
        sorted_frames = sorted(valid_set)
        start = sorted_frames[0]
        prev = start
        for f in sorted_frames[1:]:
            if f == prev + 1:
                prev = f
            else:
                length = prev - start + 1
                if length >= min_length:
                    intervals.append((start, prev + 1))
                start = f
                prev = f
        length = prev - start + 1
        if length >= min_length:
            intervals.append((start, prev + 1))

        return intervals


class VideoDataset(torch.utils.data.Dataset):
    """
    Dataset for loading video data in the standardized format.

    Supports:
    - Metadata: SQLite database (metadata.db)
    - Images: HDF5 or MP4 (auto-detected)
    - Events: MP4 with 2x3 grid layout

    Output:
    - images: List[PIL.Image] (monocular) or images_left/images_right (stereo)
    - events: np.ndarray (T, 6, H, W) uint8
    - depths/flows/masks: np.ndarray
    """

    def __init__(self, config: VideoDatasetConfig, processor: BaseProcessor):
        self.config = config
        self.processor = processor
        self.base_path = config.base_path
        self.metadata_path = config.metadata_path
        self.repeat = config.repeat


        self._metadata_db = MetadataDB(self.metadata_path, readonly=True)


        self.seq_ids = self._metadata_db.get_all_seq_ids()
        if not self.seq_ids:
            raise ValueError(f"No sequences found in metadata: {self.metadata_path}")

    def close(self) -> None:
        """Close database connection and cleanup resources."""
        if hasattr(self, "_metadata_db") and self._metadata_db is not None:
            self._metadata_db.close()
            self._metadata_db = None

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.close()

    @staticmethod
    def _reinit_db(dataset: "VideoDataset") -> None:
        """Close and reopen DB connection for a single VideoDataset instance."""
        if hasattr(dataset, "_metadata_db") and dataset._metadata_db is not None:
            dataset._metadata_db.close()
        dataset._metadata_db = MetadataDB(dataset.metadata_path, readonly=True)

    @staticmethod
    def worker_init_fn(worker_id: int) -> None:
        """
        Worker initialization function for DataLoader.

        Opens per-worker readonly DB connection.
        Handles both plain VideoDataset and ConcatDataset of VideoDatasets.
        Usage: DataLoader(..., worker_init_fn=VideoDataset.worker_init_fn)
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            dataset = worker_info.dataset
            if isinstance(dataset, VideoDataset):
                VideoDataset._reinit_db(dataset)
            elif isinstance(dataset, torch.utils.data.ConcatDataset):
                for sub_dataset in dataset.datasets:
                    if isinstance(sub_dataset, VideoDataset):
                        VideoDataset._reinit_db(sub_dataset)

    @staticmethod
    def collate_fn(batch):
        """Collate function for variable-resolution video batching.

        Videos are expected in packed (P, 3) format from the processor.
        Tensors are concatenated along dim=0; non-tensors are kept as lists.
        Per-sample metadata is extracted for downstream unpacking.

        For batch_size=1, returns the single sample unchanged.
        """
        if len(batch) == 1:
            return batch[0]

        batched = {}
        batched["_batch_size"] = len(batch)


        batched["_num_frames_list"] = [
            sample["num_frames"] for sample in batch
            if "num_frames" in sample
        ]


        if "target_height" in batch[0]:
            batched["_target_heights"] = [s["target_height"] for s in batch]
            batched["_target_widths"] = [s["target_width"] for s in batch]


        if "target_video" in batch[0]:
            batched["_target_pixel_counts"] = [
                s["target_video"].shape[0] for s in batch
            ]
        if "source_videos" in batch[0]:
            batched["_source_pixel_counts"] = [
                s["source_videos"].shape[0] for s in batch
            ]


        if "source_view_num_frames" in batch[0]:
            batched["_source_view_num_frames"] = [
                s["source_view_num_frames"] for s in batch
            ]
        if "source_view_pixel_counts" in batch[0]:
            batched["_source_view_pixel_counts"] = [
                s["source_view_pixel_counts"] for s in batch
            ]


        for key in batch[0]:
            values = [sample[key] for sample in batch]
            if isinstance(values[0], torch.Tensor):
                batched[key] = torch.cat(values, dim=0)
            else:
                batched[key] = values

        return batched

    def _load_images(
        self,
        seq_path: str,
        dataset_type: str,
        camera_id: Optional[str],
        start_idx: int,
        end_idx: int,
    ) -> List[Image.Image]:
        """Load images from either HDF5 or MP4 format."""
        h5_path = os.path.join(seq_path, "images.h5")

        if dataset_type == "monocular":
            mp4_path = os.path.join(seq_path, "images.mp4")
            if os.path.exists(mp4_path):
                decoder = VideoDecoder(mp4_path)
                frames = _load_mp4_frames(decoder, start_idx, end_idx)
                del decoder
                return frames
            else:
                return _load_hdf5_images(h5_path, "data", start_idx, end_idx)

        elif dataset_type == "stereo":
            mp4_path = os.path.join(seq_path, f"images_{camera_id}.mp4")
            if os.path.exists(mp4_path):
                decoder = VideoDecoder(mp4_path)
                frames = _load_mp4_frames(decoder, start_idx, end_idx)
                del decoder
                return frames
            else:
                return _load_hdf5_images(h5_path, camera_id, start_idx, end_idx)

        elif dataset_type == "multiview":
            mp4_path = os.path.join(seq_path, f"images_{camera_id}.mp4")
            if os.path.exists(mp4_path):
                decoder = VideoDecoder(mp4_path)
                frames = _load_mp4_frames(decoder, start_idx, end_idx)
                del decoder
                return frames
            else:
                return _load_hdf5_images(h5_path, camera_id, start_idx, end_idx)

        elif dataset_type == "static":
            mp4_path = os.path.join(seq_path, "images.mp4")
            if os.path.exists(mp4_path):
                decoder = VideoDecoder(mp4_path)
                frames = _load_mp4_frames(decoder, start_idx, end_idx)
                del decoder
                return frames
            else:
                return _load_hdf5_images(h5_path, "data", start_idx, end_idx)

        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    def _load_images_tensor(
        self,
        seq_path: str,
        dataset_type: str,
        camera_id: Optional[str],
        start_idx: int,
        end_idx: int,
    ) -> torch.Tensor:
        """Load images as (N, C, H, W) uint8 tensor. Mirrors _load_images but avoids PIL."""
        h5_path = os.path.join(seq_path, "images.h5")

        if dataset_type == "monocular":
            mp4_path = os.path.join(seq_path, "images.mp4")
            if os.path.exists(mp4_path):
                decoder = VideoDecoder(mp4_path)
                frames = _load_mp4_frames_tensor(decoder, start_idx, end_idx)
                del decoder
                return frames
            else:
                return _load_hdf5_images_tensor(h5_path, "data", start_idx, end_idx)

        elif dataset_type == "stereo":
            mp4_path = os.path.join(seq_path, f"images_{camera_id}.mp4")
            if os.path.exists(mp4_path):
                decoder = VideoDecoder(mp4_path)
                frames = _load_mp4_frames_tensor(decoder, start_idx, end_idx)
                del decoder
                return frames
            else:
                return _load_hdf5_images_tensor(h5_path, camera_id, start_idx, end_idx)

        elif dataset_type == "multiview":
            mp4_path = os.path.join(seq_path, f"images_{camera_id}.mp4")
            if os.path.exists(mp4_path):
                decoder = VideoDecoder(mp4_path)
                frames = _load_mp4_frames_tensor(decoder, start_idx, end_idx)
                del decoder
                return frames
            else:
                return _load_hdf5_images_tensor(h5_path, camera_id, start_idx, end_idx)

        elif dataset_type == "static":
            mp4_path = os.path.join(seq_path, "images.mp4")
            if os.path.exists(mp4_path):
                decoder = VideoDecoder(mp4_path)
                frames = _load_mp4_frames_tensor(decoder, start_idx, end_idx)
                del decoder
                return frames
            else:
                return _load_hdf5_images_tensor(h5_path, "data", start_idx, end_idx)

        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    def _load_events(
        self,
        seq_path: str,
        dataset_type: str,
        camera_id: Optional[str],
        start_idx: int,
        end_idx: int,
        height: int,
        width: int,
    ) -> Optional[np.ndarray]:
        """Load event video and rearrange to (T, 6, H, W)."""
        if dataset_type == "monocular":
            event_path = os.path.join(seq_path, "events.mp4")
        elif dataset_type == "stereo":
            event_path = os.path.join(seq_path, f"events_{camera_id}.mp4")
        elif dataset_type == "multiview":
            event_path = os.path.join(seq_path, f"events_{camera_id}.mp4")
        else:
            return None

        if not os.path.exists(event_path):
            return None

        decoder = VideoDecoder(event_path)
        return _load_event_frames(decoder, start_idx, end_idx, height, width)

    def _load_hdf5_data_for_keys(
        self, h5_path: str, keys: List[str], start_idx: int, end_idx: int
    ) -> Dict[str, np.ndarray]:
        """Load data from HDF5 for multiple keys."""
        data = {}
        with h5py.File(h5_path, "r") as f:
            for key in keys:
                if key in f:
                    data[key] = f[key][start_idx:end_idx]
                else:
                    raise KeyError(f"Key '{key}' not found in {h5_path}")
        return data

    def _load_flow_data(
        self,
        h5_path: str,
        keys: List[str],
        start_idx: int,
        end_idx: int,
        is_forward: bool,
    ) -> Dict[str, np.ndarray]:
        """Load optical flow with proper indexing."""
        data = {}
        with h5py.File(h5_path, "r") as f:
            for key in keys:
                if key in f:
                    if is_forward:
                        data[key] = f[key][start_idx : end_idx - 1]
                    else:
                        data[key] = f[key][start_idx + 1 : end_idx]
                else:
                    raise KeyError(f"Key '{key}' not found in {h5_path}")
        return data

    def _get_hdf5_keys(
        self, dataset_type: str, camera_ids: Optional[List[str]] = None
    ) -> List[str]:
        """Get HDF5 dataset keys based on dataset type."""
        if dataset_type == "monocular":
            return ["data"]
        elif dataset_type == "stereo":
            return ["left", "right"]
        elif dataset_type == "multiview":
            if camera_ids is None:
                raise ValueError("camera_ids required for multiview")
            return camera_ids
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Select a sequence by index and delegate processing to the processor."""
        sequence_index = idx % len(self.seq_ids)
        seq_id = self.seq_ids[sequence_index]
        seq_info = self._metadata_db.get_sequence_info(seq_id)

        context = ProcessorContext(self, seq_id, seq_info)
        return self.processor.process(context)

    def __len__(self) -> int:
        return len(self.seq_ids) * self.repeat


def create_video_dataset(
    base_path: str,
    metadata_path: str,
    processor: BaseProcessor,
    repeat: int = 1,
    seed: Optional[int] = None,
) -> VideoDataset:
    """Factory function to create a VideoDataset with the given processor."""
    config = VideoDatasetConfig(
        base_path=base_path,
        metadata_path=metadata_path,
        repeat=repeat,
    )
    return VideoDataset(config, processor)
