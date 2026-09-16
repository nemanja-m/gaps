from __future__ import annotations

from pathlib import Path

import cv2 as cv
import numpy as np

from gaps.domain import Image


class ImageIOError(RuntimeError):
    """Raised when an image cannot be read or written."""


def read_image(path: str | Path) -> Image:
    """Read a grayscale or color image or raise a descriptive I/O error."""
    image = cv.imread(str(path), cv.IMREAD_UNCHANGED)
    if image is None:
        raise ImageIOError(f"could not read image: {path}")
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv.cvtColor(image, cv.COLOR_BGRA2BGR)
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (1, 3)):
        raise ImageIOError("only grayscale and three-channel images are supported")
    return np.asarray(image, dtype=np.uint8)


def write_image(path: str | Path, image: Image) -> None:
    """Write an image or raise a descriptive I/O error."""
    if not cv.imwrite(str(path), image):
        raise ImageIOError(f"could not write image: {path}")
