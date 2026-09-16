from typing import Literal

import numpy as np

from gaps.piece import Piece

EdgeOrientation = Literal["LR", "TD"]


def dissimilarity_measure(
    first_piece: Piece,
    second_piece: Piece,
    orientation: EdgeOrientation = "LR",
) -> float:
    """Calculate the color difference between two adjacent piece edges."""
    rows, columns, _ = first_piece.shape

    if orientation == "LR":
        color_difference = first_piece[:, columns - 1, :].astype(
            np.float32
        ) - second_piece[:, 0, :].astype(np.float32)
    elif orientation == "TD":
        color_difference = first_piece[rows - 1, :, :].astype(
            np.float32
        ) - second_piece[0, :, :].astype(np.float32)
    else:
        raise ValueError(f"Unsupported edge orientation: {orientation}")

    try:
        squared_difference = np.square(color_difference / 255.0)
        return float(np.sqrt(np.sum(squared_difference)))
    except (TypeError, ValueError) as error:
        raise ValueError("pieces must contain compatible numeric image data") from error
