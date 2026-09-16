from __future__ import annotations

import numpy as np

from gaps.domain import EdgeAxis, Piece

_HUBER_BETA = 0.01
_BOUNDARY_WEIGHT = 0.60
_NORMAL_GRADIENT_WEIGHT = 0.25
_TANGENT_GRADIENT_WEIGHT = 0.15


def _as_channels(image: np.ndarray) -> np.ndarray:
    """Return an image as float channels without changing its intensity."""
    if image.ndim == 2:
        return image.astype(np.float32, copy=False)[..., np.newaxis]
    if image.ndim == 3 and image.shape[2] in (1, 3):
        return image.astype(np.float32, copy=False)
    raise ValueError("images must be grayscale or three-channel")


def _huber_mean(difference: np.ndarray, scale: float) -> float:
    """Return a normalized Huber loss for an array of pixel differences."""
    error = np.abs(difference) / scale
    loss = np.where(
        error <= _HUBER_BETA,
        0.5 * np.square(error) / _HUBER_BETA,
        error - 0.5 * _HUBER_BETA,
    )
    try:
        return float(np.mean(loss))
    except (TypeError, ValueError) as error:
        raise ValueError("could not calculate edge loss") from error


def _edge_lines(
    first_piece: Piece, second_piece: Piece, axis: EdgeAxis
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None]:
    first = _as_channels(first_piece.image)
    second = _as_channels(second_piece.image)

    if axis is EdgeAxis.HORIZONTAL:
        return (
            first[:, -1, :],
            first[:, -2, :] if first.shape[1] > 1 else None,
            second[:, 0, :],
            second[:, 1, :] if second.shape[1] > 1 else None,
        )
    if axis is EdgeAxis.VERTICAL:
        return (
            first[-1, :, :],
            first[-2, :, :] if first.shape[0] > 1 else None,
            second[0, :, :],
            second[1, :, :] if second.shape[0] > 1 else None,
        )
    raise ValueError(f"Unsupported edge orientation: {axis}")


def dissimilarity_measure(
    first_piece: Piece,
    second_piece: Piece,
    axis: EdgeAxis = EdgeAxis.HORIZONTAL,
) -> float:
    """Calculate a robust, normalized compatibility cost for two edges.

    The cost combines boundary intensity, normal gradient, and tangential
    gradient differences.  Every term is averaged over pixels and channels,
    so grayscale images and color images have comparable score ranges.
    """
    if axis not in (EdgeAxis.HORIZONTAL, EdgeAxis.VERTICAL):
        raise ValueError(f"Unsupported edge orientation: {axis}")

    try:
        first_edge, first_inner, second_edge, second_inner = _edge_lines(
            first_piece, second_piece, axis
        )
        if first_edge.shape != second_edge.shape:
            raise ValueError("edge shapes do not match")

        boundary_cost = _huber_mean(first_edge - second_edge, scale=255.0)

        if first_inner is None or second_inner is None:
            normal_gradient_cost = 0.0
        else:
            normal_gradient_cost = _huber_mean(
                (first_edge - first_inner) - (second_inner - second_edge),
                scale=510.0,
            )

        if first_edge.shape[0] < 2:
            tangent_gradient_cost = 0.0
        else:
            tangent_gradient_cost = _huber_mean(
                np.diff(first_edge, axis=0) - np.diff(second_edge, axis=0),
                scale=510.0,
            )

        return float(
            _BOUNDARY_WEIGHT * boundary_cost
            + _NORMAL_GRADIENT_WEIGHT * normal_gradient_cost
            + _TANGENT_GRADIENT_WEIGHT * tangent_gradient_cost
        )
    except (TypeError, ValueError) as error:
        raise ValueError("pieces must contain compatible numeric image data") from error
