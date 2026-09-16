from typing import Literal, overload

import numpy as np

from gaps.piece import Piece


@overload
def flatten_image(
    image: np.ndarray, piece_size: int, indexed: Literal[False] = False
) -> tuple[list[np.ndarray], int, int]: ...


@overload
def flatten_image(
    image: np.ndarray, piece_size: int, indexed: Literal[True]
) -> tuple[list[Piece], int, int]: ...


def flatten_image(
    image: np.ndarray, piece_size: int, indexed: bool = False
) -> tuple[list[np.ndarray] | list[Piece], int, int]:
    """Split an image into square pieces, cropping incomplete edges."""
    if piece_size <= 0:
        raise ValueError("piece_size must be positive")
    if image.ndim != 3:
        raise ValueError("image must be a color image with three dimensions")

    rows, columns = image.shape[0] // piece_size, image.shape[1] // piece_size
    if rows == 0 or columns == 0:
        raise ValueError("piece_size must fit within the image")

    pieces = [
        image[
            row * piece_size : (row + 1) * piece_size,
            column * piece_size : (column + 1) * piece_size,
        ].copy()
        for row in range(rows)
        for column in range(columns)
    ]

    if indexed:
        return (
            [Piece(image=piece, id=index) for index, piece in enumerate(pieces)],
            rows,
            columns,
        )

    return pieces, rows, columns


def assemble_image(
    pieces: list[np.ndarray] | list[Piece], rows: int, columns: int
) -> np.ndarray:
    """Assemble a sequence of pieces into an image."""
    if rows <= 0 or columns <= 0:
        raise ValueError("rows and columns must be positive")
    if len(pieces) != rows * columns:
        raise ValueError("piece count does not match the requested dimensions")

    images = [piece.image if isinstance(piece, Piece) else piece for piece in pieces]
    row_images = [
        np.hstack(images[row * columns : (row + 1) * columns]) for row in range(rows)
    ]
    return np.vstack(row_images).astype(np.uint8, copy=False)
