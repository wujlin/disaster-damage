"""RGB loading shared by the SAGIN and xBD datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_rgb_tensor(path: str | Path, image_size: int, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    mean_arr = np.asarray(mean, dtype=np.float32)
    std_arr = np.asarray(std, dtype=np.float32)
    array = (array - mean_arr) / std_arr
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def square_crop_box(bbox: tuple[float, float, float, float], padding: int, width: int, height: int) -> tuple[int, int, int, int]:
    """Return a square box around a polygon bbox, shifted so it stays inside the image."""
    x0, y0, x1, y1 = bbox
    x0 = int(np.floor(x0)) - padding
    y0 = int(np.floor(y0)) - padding
    x1 = int(np.ceil(x1)) + padding
    y1 = int(np.ceil(y1)) + padding
    side = max(x1 - x0, y1 - y0, 1)
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    right = left + side
    bottom = top + side
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > width:
        shift = right - width
        left -= shift
        right = width
    if bottom > height:
        shift = bottom - height
        top -= shift
        bottom = height
    left = max(0, left)
    top = max(0, top)
    right = max(left + 1, min(width, right))
    bottom = max(top + 1, min(height, bottom))
    return left, top, right, bottom


def crop_rgb_tensor(
    path: str | Path,
    box: tuple[int, int, int, int],
    image_size: int,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = image.convert("RGB").crop(box).resize((image_size, image_size), Image.BILINEAR)
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    mean_arr = np.asarray(mean, dtype=np.float32)
    std_arr = np.asarray(std, dtype=np.float32)
    array = (array - mean_arr) / std_arr
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()
