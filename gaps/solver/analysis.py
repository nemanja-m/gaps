from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, SupportsFloat

import numpy as np

from gaps.domain import Direction, EdgeAxis, Piece
from gaps.solver.fitness import (
    pairwise_dissimilarity_from_channels,
    prepare_piece_channels,
)

if TYPE_CHECKING:
    from gaps.domain import Arrangement

_AXIS_INDEX = {
    EdgeAxis.HORIZONTAL: 0,
    EdgeAxis.VERTICAL: 1,
}


def _safe_float(value: SupportsFloat) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("cost table contains non-numeric values") from error


class EdgeCostTable:
    """Pairwise edge costs and best matches for one puzzle."""

    def __init__(self) -> None:
        self._costs = np.empty((0, 0, 2), dtype=np.float32)
        self._piece_ids = np.empty(0, dtype=np.intp)
        self._id_to_index: dict[int, int] = {}
        self._best_matches: dict[int, dict[Direction, list[tuple[int, float]]]] = {}

    def analyze(
        self,
        pieces: Sequence[Piece],
        progress: Callable[[str, int, int], None] | None = None,
    ) -> None:
        """Calculate and cache all pairwise edge comparisons."""
        piece_ids = [piece.identifier for piece in pieces]
        if len(set(piece_ids)) != len(piece_ids):
            raise ValueError("piece identifiers must be unique")

        self._piece_ids = np.asarray(piece_ids, dtype=np.intp)
        self._id_to_index = {
            piece_id: index for index, piece_id in enumerate(piece_ids)
        }
        self._best_matches = {
            piece_id: {direction: [] for direction in Direction}
            for piece_id in piece_ids
        }

        piece_count = len(pieces)
        if piece_count == 0:
            self._costs = np.empty((0, 0, 2), dtype=np.float32)
            return

        channels = prepare_piece_channels(pieces)
        costs = np.empty((piece_count, piece_count, 2), dtype=np.float32)
        costs[:, :, 0] = pairwise_dissimilarity_from_channels(
            channels,
            EdgeAxis.HORIZONTAL,
        )
        costs[:, :, 1] = pairwise_dissimilarity_from_channels(
            channels,
            EdgeAxis.VERTICAL,
            progress=(
                None
                if progress is None
                else lambda current, total: progress("analysis", current, total)
            ),
        )
        self._costs = costs

        ids = self._piece_ids.tolist()
        for index, piece_id in enumerate(ids):
            matches_by_direction = self._best_matches[piece_id]
            directional_costs = {
                Direction.RIGHT: costs[index, :, _AXIS_INDEX[EdgeAxis.HORIZONTAL]],
                Direction.LEFT: costs[:, index, _AXIS_INDEX[EdgeAxis.HORIZONTAL]],
                Direction.BOTTOM: costs[index, :, _AXIS_INDEX[EdgeAxis.VERTICAL]],
                Direction.TOP: costs[:, index, _AXIS_INDEX[EdgeAxis.VERTICAL]],
            }
            for direction, match_costs in directional_costs.items():
                order = np.argsort(match_costs, kind="stable")
                matches: list[tuple[int, float]] = []
                for candidate_index in order:
                    if candidate_index != index:
                        matches.append(
                            (
                                ids[candidate_index],
                                _safe_float(match_costs[candidate_index]),
                            )
                        )
                matches_by_direction[direction] = matches

    def __call__(self, ids: tuple[int, int], axis: EdgeAxis) -> float:
        return self.cost(ids, axis)

    def cost(self, ids: tuple[int, int], axis: EdgeAxis) -> float:
        try:
            first_index = self._id_to_index[ids[0]]
            second_index = self._id_to_index[ids[1]]
            axis_index = _AXIS_INDEX[axis]
        except (KeyError, IndexError) as error:
            raise KeyError(ids) from error
        if first_index == second_index:
            raise KeyError(ids)
        return _safe_float(self._costs[first_index, second_index, axis_index])

    def arrangement_cost(self, arrangement: Arrangement) -> float:
        """Return the total seam cost using direct dense-array indexing."""
        try:
            indices = np.fromiter(
                (self._id_to_index[piece.identifier] for piece in arrangement.pieces),
                dtype=np.intp,
                count=len(arrangement.pieces),
            )
        except KeyError as error:
            raise KeyError(error.args[0]) from error

        rows = arrangement.layout.rows
        columns = arrangement.layout.columns
        grid = indices.reshape(rows, columns)
        total_cost = 0.0
        try:
            if columns > 1:
                total_cost += _safe_float(
                    np.sum(
                        self._costs[grid[:, :-1], grid[:, 1:], 0],
                        dtype=np.float64,
                    )
                )
            if rows > 1:
                total_cost += _safe_float(
                    np.sum(
                        self._costs[grid[:-1, :], grid[1:, :], 1],
                        dtype=np.float64,
                    )
                )
        except (TypeError, ValueError) as error:
            raise ValueError("could not calculate arrangement cost") from error
        return total_cost

    def best_match(self, piece_id: int, direction: Direction) -> int:
        matches = self._best_matches[piece_id][direction]
        if not matches:
            raise ValueError(f"no best match available for piece {piece_id}")
        return matches[0][0]

    def matches(self, piece_id: int, direction: Direction) -> list[tuple[int, float]]:
        return self._best_matches[piece_id][direction]
