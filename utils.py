"""Shared helpers for ARC visualisations."""

from __future__ import annotations

from typing import Sequence

import numpy as np


ARC_COLORS = np.array(
    [
        [0, 0, 0],
        [30, 147, 255],
        [249, 60, 49],
        [79, 204, 48],
        [255, 220, 0],
        [153, 153, 153],
        [229, 58, 163],
        [255, 133, 27],
        [135, 216, 241],
        [146, 18, 49],
    ],
    dtype=np.uint8,
)

GRID_COLOR = np.array([80, 80, 80], dtype=np.uint8)


def grid_to_image(grid: Sequence[Sequence[int]], max_cell_size: int = 32) -> np.ndarray | None:
    """Convert an ARC grid to a small RGB image as a NumPy array.

    The Streamlit app can feed this directly to ``st.image``. We upscale each
    cell using ``np.repeat`` which keeps the code simple (no Matplotlib).
    """

    if grid is None:
        return None

    arr = np.asarray(grid)
    if arr.ndim != 2:
        return None

    height, width = arr.shape
    if height == 0 or width == 0:
        return None

    max_dim = max(height, width)
    scale = max(1, min(max_cell_size, 256 // max_dim))

    rgb = ARC_COLORS[np.clip(arr, 0, len(ARC_COLORS) - 1)]

    line = 1  # thin grid line
    out_h = height * scale + (height + 1) * line
    out_w = width * scale + (width + 1) * line
    img = np.full((out_h, out_w, 3), GRID_COLOR, dtype=np.uint8)

    for r in range(height):
        y0 = r * (scale + line) + line
        y1 = y0 + scale
        for c in range(width):
            x0 = c * (scale + line) + line
            x1 = x0 + scale
            img[y0:y1, x0:x1] = rgb[r, c]

    return img


def grids_equal(a: Sequence[Sequence[int]] | None, b: Sequence[Sequence[int]] | None) -> bool:
    """Return True if two grids have identical shape and values."""

    if a is None or b is None:
        return False

    arr_a = np.asarray(a)
    arr_b = np.asarray(b)
    if arr_a.ndim != 2 or arr_b.ndim != 2:
        return False
    return arr_a.shape == arr_b.shape and np.array_equal(arr_a, arr_b)
