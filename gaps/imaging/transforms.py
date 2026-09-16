from __future__ import annotations

from typing import Literal, overload

import numpy as np

from gaps.domain import Image, Piece, PuzzleLayout


@overload
def flatten_image(
    image: Image, piece_size: int, indexed: Literal[False] = False
) -> tuple[list[Image], PuzzleLayout]: ...


@overload
def flatten_image(
    image: Image, piece_size: int, indexed: Literal[True]
) -> tuple[list[Piece], PuzzleLayout]: ...


def flatten_image(
    image: Image, piece_size: int, indexed: bool = False
) -> tuple[list[Image] | list[Piece], PuzzleLayout]:
    """Split an image into square pieces, cropping incomplete edges."""
    if piece_size <= 0:
        raise ValueError("piece_size must be positive")
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (1, 3)):
        raise ValueError("image must be grayscale or three-channel color")

    rows, columns = image.shape[0] // piece_size, image.shape[1] // piece_size
    if rows == 0 or columns == 0:
        raise ValueError("piece_size must fit within the image")

    layout = PuzzleLayout(rows, columns, piece_size)
    pieces = [
        image[
            row * piece_size : (row + 1) * piece_size,
            column * piece_size : (column + 1) * piece_size,
        ].copy()
        for row in range(rows)
        for column in range(columns)
    ]
    if indexed:
        return [Piece(piece, index) for index, piece in enumerate(pieces)], layout
    return pieces, layout


def assemble_image(pieces: list[Image] | list[Piece], layout: PuzzleLayout) -> Image:
    """Assemble a sequence of pieces into an image."""
    if len(pieces) != layout.piece_count:
        raise ValueError("piece count does not match the puzzle layout")

    images = [piece.image if isinstance(piece, Piece) else piece for piece in pieces]
    rows = [
        np.hstack(images[row * layout.columns : (row + 1) * layout.columns])
        for row in range(layout.rows)
    ]
    return np.vstack(rows).astype(np.uint8, copy=False)
