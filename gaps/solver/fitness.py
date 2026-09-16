from __future__ import annotations

import numpy as np

from gaps.domain import EdgeAxis, Piece


def dissimilarity_measure(
    first_piece: Piece,
    second_piece: Piece,
    axis: EdgeAxis = EdgeAxis.HORIZONTAL,
) -> float:
    """Calculate the color difference between two adjacent piece edges."""
    rows, columns, _ = first_piece.image.shape

    if axis is EdgeAxis.HORIZONTAL:
        first_edge = first_piece[:, columns - 1, :]
        second_edge = second_piece[:, 0, :]
    elif axis is EdgeAxis.VERTICAL:
        first_edge = first_piece[rows - 1, :, :]
        second_edge = second_piece[0, :, :]
    else:
        raise ValueError(f"Unsupported edge orientation: {axis}")

    try:
        color_difference = first_edge.astype(np.float32) - second_edge.astype(
            np.float32
        )
        return float(np.sqrt(np.sum(np.square(color_difference / 255.0))))
    except (TypeError, ValueError) as error:
        raise ValueError("pieces must contain compatible numeric image data") from error
