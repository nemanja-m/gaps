from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from gaps.domain import EdgeAxis, Piece

_HUBER_BETA = 0.01
_BOUNDARY_WEIGHT = 0.60
_NORMAL_GRADIENT_WEIGHT = 0.25
_TANGENT_GRADIENT_WEIGHT = 0.15
_PAIRWISE_BLOCK_SIZE = 32


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


def _huber_mean_batch(difference: np.ndarray, scale: float) -> np.ndarray:
    """Calculate Huber means for a batch of pairwise pixel differences."""
    error = np.abs(difference) / scale
    loss = np.where(
        error <= _HUBER_BETA,
        0.5 * np.square(error) / _HUBER_BETA,
        error - 0.5 * _HUBER_BETA,
    )
    return np.mean(loss, axis=(-2, -1))


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


def prepare_piece_channels(pieces: Sequence[Piece]) -> np.ndarray:
    """Convert all pieces to one contiguous float32 image array."""
    try:
        return np.stack([_as_channels(piece.image) for piece in pieces], axis=0)
    except (TypeError, ValueError) as error:
        raise ValueError("pieces must contain compatible numeric image data") from error


def pairwise_dissimilarity_from_channels(
    channels: np.ndarray,
    axis: EdgeAxis,
    block_size: int = _PAIRWISE_BLOCK_SIZE,
    progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Calculate directed pairwise edge costs in bounded vectorized blocks."""
    if axis not in (EdgeAxis.HORIZONTAL, EdgeAxis.VERTICAL):
        raise ValueError(f"Unsupported edge orientation: {axis}")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if channels.ndim != 4 or channels.shape[3] not in (1, 3):
        raise ValueError("pieces must contain compatible numeric image data")

    piece_count = channels.shape[0]
    if piece_count == 0:
        return np.empty((0, 0), dtype=np.float32)

    if axis is EdgeAxis.HORIZONTAL:
        first_edge = channels[:, :, -1, :]
        first_inner = channels[:, :, -2, :] if channels.shape[2] > 1 else None
        second_edge = channels[:, :, 0, :]
        second_inner = channels[:, :, 1, :] if channels.shape[2] > 1 else None
    else:
        first_edge = channels[:, -1, :, :]
        first_inner = channels[:, -2, :, :] if channels.shape[1] > 1 else None
        second_edge = channels[:, 0, :, :]
        second_inner = channels[:, 1, :, :] if channels.shape[1] > 1 else None

    first_normal = None if first_inner is None else first_edge - first_inner
    second_normal = None if second_inner is None else second_inner - second_edge
    first_tangent = np.diff(first_edge, axis=1) if first_edge.shape[1] > 1 else None
    second_tangent = np.diff(second_edge, axis=1) if second_edge.shape[1] > 1 else None
    costs = np.empty((piece_count, piece_count), dtype=np.float32)

    for start in range(0, piece_count, block_size):
        stop = min(start + block_size, piece_count)
        first_edge_block = first_edge[start:stop, np.newaxis, :, :]
        second_edge_block = second_edge[np.newaxis, :, :, :]

        boundary_cost = _huber_mean_batch(
            first_edge_block - second_edge_block,
            scale=255.0,
        )

        if first_normal is None or second_normal is None:
            normal_cost = 0.0
        else:
            first_normal_block = first_normal[start:stop, np.newaxis, :, :]
            second_normal_block = second_normal[np.newaxis, :, :, :]
            normal_cost = _huber_mean_batch(
                first_normal_block - second_normal_block,
                scale=510.0,
            )

        if first_tangent is None or second_tangent is None:
            tangent_cost = 0.0
        else:
            first_tangent_block = first_tangent[start:stop, np.newaxis, :, :]
            second_tangent_block = second_tangent[np.newaxis, :, :, :]
            tangent_cost = _huber_mean_batch(
                first_tangent_block - second_tangent_block,
                scale=510.0,
            )

        costs[start:stop] = (
            _BOUNDARY_WEIGHT * boundary_cost
            + _NORMAL_GRADIENT_WEIGHT * normal_cost
            + _TANGENT_GRADIENT_WEIGHT * tangent_cost
        )
        if progress is not None and piece_count > 1:
            progress(min(stop, piece_count - 1), piece_count - 1)

    np.fill_diagonal(costs, np.inf)
    return costs


def pairwise_dissimilarity(
    pieces: Sequence[Piece],
    axis: EdgeAxis,
    block_size: int = _PAIRWISE_BLOCK_SIZE,
) -> np.ndarray:
    """Calculate directed pairwise edge costs for a sequence of pieces."""
    return pairwise_dissimilarity_from_channels(
        prepare_piece_channels(pieces),
        axis,
        block_size=block_size,
    )


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
