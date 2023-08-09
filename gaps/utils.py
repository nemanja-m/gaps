from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray


def flatten_image(
    image: NDArray[np.uint8],
    piece_size: int,
) -> Tuple[List[NDArray[np.uint8]], int, int]:
    """Convert image into list of square pieces.

    Input image is divided into square pieces of specified size and than
    flattened into list. Each list element has shape of 'piece_size x piece_size x 3'.

    Args:
        image: Input image as numpy ndarray.
        piece_size: Size of single square piece.

    Examples:
        >>> from gaps.utils import flatten_image
        >>> flatten_image(image, piece_size=32)
        [np.ndarray(..., dtype=np.uint8), np.ndarray(..., dtype=np.uint8), ...]
    """

    rows, columns = image.shape[0] // piece_size, image.shape[1] // piece_size
    pieces: List[NDArray[np.uint8]] = []

    # Crop pieces from original image
    for y in range(rows):
        for x in range(columns):
            left, top, w, h = (
                x * piece_size,
                y * piece_size,
                (x + 1) * piece_size,
                (y + 1) * piece_size,
            )
            piece = np.empty((piece_size, piece_size, image.shape[2]), dtype=np.uint8)
            piece[:piece_size, :piece_size, :] = image[top:h, left:w, :]
            pieces.append(piece)

    return pieces, rows, columns


def stitch_image(
    pieces: List[NDArray[np.uint8]],
    rows: int,
    columns: int,
) -> NDArray[np.uint8]:
    """Stitch full puzzle image from pieces.

    Given an array of pieces and desired image dimensions, assembles
    image by stacking pieces.

    Args:
        pieces: Image pieces as an numpy array.
        rows: Number of rows in puzzle image.
        columns: Number of columns in puzzle image.

    Examples:
        >>> from gaps.utils import stitch_image, flatten_image
        >>> pieces, rows, cols = flatten_image(...)
        >>> stitch_image(pieces, rows, cols)
        np.ndarray(..., dtype=np.uint8)
    """
    vertical_stack = []
    for i in range(rows):
        horizontal_stack = []
        for j in range(columns):
            horizontal_stack.append(pieces[i * columns + j])
        vertical_stack.append(np.hstack(horizontal_stack))
    return np.vstack(vertical_stack).astype(np.uint8)
