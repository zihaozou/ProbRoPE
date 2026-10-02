"""Read raw events and pack six temporal bins into video-frame grids."""

from typing import Iterator

import numpy as np
import polars as pl

from dfuse.utils.events.event_utils import process_event_image


def load_events_synthetic(path: str) -> pl.DataFrame:
    """Load synthetic events from v2e output text file.

    Two on-disk layouts are supported:

    - Headerless v2e output with ``# ...`` comments and rows in
      ``t x y p`` order (the original/synthetic case).
    - A plain header row naming the columns (e.g. ``x y t p``) followed
      by data rows in that same order — what the real-world capture
      pipeline writes.

    The first non-comment line is peeked: if all whitespace-separated
    tokens parse as floats, it's treated as data (no header); otherwise
    it's treated as a header row.

    Args:
        path: Path to events.txt file

    Returns:
        Polars DataFrame with columns [t, x, y, p]
        - t: timestamp in seconds (Float64)
        - x: pixel x coordinate (horizontal)
        - y: pixel y coordinate (vertical)
        - p: polarity (1 or -1, or 0/1 depending on v2e settings)
    """
    first_data_line = None
    with open(path, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            first_data_line = stripped
            break

    has_header_row = False
    if first_data_line is not None:
        tokens = first_data_line.split()
        try:
            for tok in tokens:
                float(tok)
        except ValueError:
            has_header_row = True

    if has_header_row:
        df = pl.read_csv(
            path,
            separator=" ",
            has_header=True,
            comment_prefix="#",
        )
        missing = [c for c in ("t", "x", "y", "p") if c not in df.columns]
        if missing:
            raise ValueError(
                f"events file {path} header missing required columns "
                f"{missing}; got {df.columns}"
            )
        df = df.select(["t", "x", "y", "p"])
    else:
        df = pl.read_csv(
            path,
            separator=" ",
            has_header=False,
            comment_prefix="#",
            new_columns=["t", "x", "y", "p"],
        )

    df = df.with_columns(pl.col("t").cast(pl.Float64))
    return df


def iter_event_slices(
    events_df: pl.DataFrame,
    num_frames: int,
    fps: float,
) -> Iterator[np.ndarray]:
    """Iterate over event slices for each frame.

    Yields events for each frame interval. The first yield is an empty array
    (for frame 0, which has no preceding events).

    Args:
        events_df: Polars DataFrame with columns [t, x, y, p]
        num_frames: Number of RGB frames
        fps: Frame rate of the video

    Yields:
        numpy arrays of shape (N, 4) with columns [t, x, y, p] for each frame.
        First frame yields empty array (no events before first frame).
    """
    frame_dt = 1.0 / fps


    timestamps = events_df["t"].to_numpy().astype(np.float64, copy=False)
    xs = events_df["x"].to_numpy().astype(np.float32, copy=False)
    ys = events_df["y"].to_numpy().astype(np.float32, copy=False)
    ps = events_df["p"].to_numpy().astype(np.float32, copy=False)
    events = np.stack([timestamps.astype(np.float32, copy=False), xs, ys, ps], axis=1)


    yield np.zeros((0, 4), dtype=np.float32)


    for i in range(1, num_frames):
        t_start = (i - 1) * frame_dt
        t_end = i * frame_dt

        mask = (timestamps >= t_start) & (timestamps < t_end)
        chunk_events = events[mask]


        if len(chunk_events) > 0:
            chunk_events = chunk_events.copy()
            chunk_events[:, 0] -= t_start

        yield chunk_events.astype(np.float32)


def events_to_grid_image(
    events: np.ndarray,
    height: int,
    width: int,
    num_bins: int = 6,
) -> np.ndarray:
    """Convert events to a grid image by rearranging 6 event images into 2x3 grid.

    Args:
        events: Event array of shape (N, 4) with columns [t, x, y, p]
        height: Frame height
        width: Frame width
        num_bins: Number of temporal bins (default: 6)

    Returns:
        Grid image of shape (H*2, W*3, 3) as uint8 (RGB for video encoding)
    """

    event_images = process_event_image(events, height, width, num_bins=num_bins)


    img_arrays = [np.array(img) for img in event_images]


    row0 = np.concatenate([img_arrays[0], img_arrays[1], img_arrays[2]], axis=1)
    row1 = np.concatenate([img_arrays[3], img_arrays[4], img_arrays[5]], axis=1)
    grid_image = np.concatenate([row0, row1], axis=0)


    grid_image_rgb = np.stack([grid_image, grid_image, grid_image], axis=-1)

    return grid_image_rgb
