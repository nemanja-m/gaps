from __future__ import annotations

from pathlib import Path

import cv2 as cv
import numpy as np

from gaps.domain import Image


class ImageIOError(RuntimeError):
    """Raised when an image cannot be read or written."""


def read_image(path: str | Path) -> Image:
    """Read a color image or raise a descriptive I/O error."""
    image = cv.imread(str(path), cv.IMREAD_COLOR)
    if image is None:
        raise ImageIOError(f"could not read image: {path}")
    return np.asarray(image, dtype=np.uint8)


def write_image(path: str | Path, image: Image) -> None:
    """Write an image or raise a descriptive I/O error."""
    if not cv.imwrite(str(path), image):
        raise ImageIOError(f"could not write image: {path}")
