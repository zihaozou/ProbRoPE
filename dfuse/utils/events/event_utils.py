from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterator, List, Union
import numpy as np
from PIL import Image
from pydantic import BaseModel
from natsort import natsorted
import torch

NUM_BINS = 6


def voxel_norm(voxel):
    """
    Normalize voxel grid to [-1, 1] based on 2% and 98% percentile for positive and negative values
    """
    voxel_pos = voxel[voxel > 0]

    voxel_neg = voxel[voxel < 0]


    if len(voxel_pos) == 0:
        return voxel

    if len(voxel_neg) == 0:
        return voxel

    pos_2 = np.percentile(voxel_pos, 2)

    pos_98 = np.percentile(voxel_pos, 98)

    neg_2 = np.percentile(voxel_neg, 2)

    neg_98 = np.percentile(voxel_neg, 98)

    voxel[voxel > 0] = np.clip(voxel[voxel > 0], pos_2, pos_98)

    voxel[voxel < 0] = np.clip(voxel[voxel < 0], neg_2, neg_98)

    if pos_98 == pos_2:
        pos_98 = np.max(voxel_pos)
        pos_2 = np.min(voxel_pos)

    if neg_98 == neg_2:
        neg_98 = np.max(voxel_neg)
        neg_2 = np.min(voxel_neg)

    if pos_98 == pos_2:
        voxel[voxel > 0] = np.where(voxel[voxel > 0] > 0, 1, 0)
    else:

        voxel[voxel > 0] = voxel[voxel > 0] / pos_98

    if neg_98 == neg_2:
        voxel[voxel < 0] = np.where(voxel[voxel < 0] < 0, -1, 0)
    else:
        voxel[voxel < 0] = -1 * (voxel[voxel < 0]) / (neg_98)

    return voxel


def visualize_voxel_grid(voxel_grid):
    """
    Visualize a voxel grid as RGB image.
    """


    voxel_grid = np.clip(voxel_grid, -1.0, 1.0)

    voxel_grid = (voxel_grid + 1.0) / 2

    voxel_grid = voxel_grid * 255
    voxel_grid = voxel_grid.astype(np.uint8)

    image_list = []

    for i in range(voxel_grid.shape[0]):
        img_i = voxel_grid[i, ...]

        img_save = Image.fromarray(img_i)

        image_list.append(img_save)

    return image_list


def get_event_stacks(events, num_stacks):


    max_ts = np.max(events[:, 0])

    events[:, 0] = max_ts - events[:, 0]

    events_reversed = events[np.argsort(events[:, 0])]

    event_stacks = []

    total_events = len(events_reversed)

    stack_idx_arr = np.linspace(0, num_stacks - 1, num_stacks).astype(np.int32)

    for i in range(num_stacks):
        stack_idx = 2 ** stack_idx_arr[i]

        stack_idx = total_events // stack_idx - 1

        event_stack = events_reversed[:stack_idx, ...]

        event_stacks.append(event_stack)

    return event_stacks


def events_to_voxel_grid(events, num_bins, width, height):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    """

    assert events.shape[1] == 4
    assert num_bins > 0
    assert width > 0
    assert height > 0

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    if len(events) < 5:
        voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))
        return voxel_grid

    events = events[np.argsort(events[:, 0])]


    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    new_ts = events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT


    ts = new_ts
    xs = events[:, 1].astype(np.int64)
    ys = events[:, 2].astype(np.int64)
    pols = events[:, 3]
    pols[pols == 0] = -1

    tis = ts.astype(np.int64)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins

    np.add.at(
        voxel_grid,
        xs[valid_indices]
        + ys[valid_indices] * width
        + tis[valid_indices] * width * height,
        vals_left[valid_indices],
    )

    valid_indices = (tis + 1) < num_bins
    np.add.at(
        voxel_grid,
        xs[valid_indices]
        + ys[valid_indices] * width
        + (tis[valid_indices] + 1) * width * height,
        vals_right[valid_indices],
    )

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    return voxel_grid


def process_event_image(events, h, w, num_bins=NUM_BINS):
    """
    Main function
    """

    if len(events) == 0:
        voxel_img = np.zeros((num_bins, h, w))

        save_img_list = visualize_voxel_grid(voxel_img)

        return save_img_list

    events_stacks = get_event_stacks(events, NUM_BINS)

    events_stack_voxels = []

    for i in range(NUM_BINS):
        event_voxels_i = events_to_voxel_grid(events_stacks[i], 1, w, h)

        """
        Normalize the voxel grid first!!!!
        """

        event_voxels_i = voxel_norm(event_voxels_i)

        events_stack_voxels.append(event_voxels_i)

    event_voxels = np.stack(events_stack_voxels, axis=0)

    event_voxels = event_voxels.squeeze(1)

    save_img_list = visualize_voxel_grid(event_voxels)

    return save_img_list


def convert_stack_to_tensor(stack):
    img_tensor = torch.from_numpy(np.array(stack)).float()

    img_tensor_1ch = img_tensor


    img_normalized = img_tensor_1ch / 127.0 - 1

    img_normalized = img_normalized.unsqueeze(0)

    return img_normalized


class EventDatasetMetadata(BaseModel):
    name: str
    path: Path
    video_path: Union[Path, List[Path]]
    event_path: Union[Path, List[Path]]
    fps: int
    video_timestamps: List[float]


class EventDatasetIterator(ABC):
    FRAME_WIDTH: int
    FRAME_HEIGHT: int

    @abstractmethod
    def __init__(self, dataset_path: Union[str, Path]):
        pass

    @abstractmethod
    def iterate_metadata(self) -> Iterator[EventDatasetMetadata]:
        """Iterate over metadata for each video-event pair in the dataset."""
        pass

    @abstractmethod
    def load_events(self, event_path: Union[Path, List[Path]]) -> np.ndarray:
        """Load events from the given path(s)."""
        pass

    @staticmethod
    def frame_index_from_timestamp(
        events: np.ndarray, timestamps: np.ndarray
    ) -> np.ndarray:
        """Get frame indices from timestamps."""
        if len(events) == 0:
            raise ValueError("No events found to process.")
        if len(timestamps) == 0:
            raise ValueError("No timestamps provided to process.")

        index = np.searchsorted(events[:, 0], timestamps, side="left")
        index = np.clip(index, 0, len(events) - 1)
        return index


def group_continuous_frames(timestamps: Dict[int, float]) -> List[Dict[int, float]]:

    groups = []
    current_group = {}
    sorted_keys = sorted(timestamps.keys())
    for k in sorted_keys:
        if not current_group:
            current_group[k] = timestamps[k]
            last_k = k
        elif k == last_k + 1:
            current_group[k] = timestamps[k]
            last_k = k
        else:
            groups.append(current_group)
            current_group = {k: timestamps[k]}
            last_k = k
    if current_group:
        groups.append(current_group)
    for group in groups:
        if len(group) < 16:
            groups.remove(group)
    return groups
