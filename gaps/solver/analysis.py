from __future__ import annotations

from collections.abc import Callable, Sequence

from gaps.domain import Direction, EdgeAxis, Piece
from gaps.solver.fitness import dissimilarity_measure

_MATCH_DIRECTIONS: dict[EdgeAxis, tuple[Direction, Direction]] = {
    EdgeAxis.HORIZONTAL: (Direction.LEFT, Direction.RIGHT),
    EdgeAxis.VERTICAL: (Direction.TOP, Direction.BOTTOM),
}


class EdgeCostTable:
    """Pairwise edge costs and best matches for one puzzle."""

    def __init__(self) -> None:
        self._costs: dict[tuple[int, int], dict[EdgeAxis, float]] = {}
        self._best_matches: dict[int, dict[Direction, list[tuple[int, float]]]] = {}

    def analyze(
        self,
        pieces: Sequence[Piece],
        progress: Callable[[str, int, int], None] | None = None,
    ) -> None:
        """Calculate and cache all pairwise edge comparisons."""
        self._costs.clear()
        self._best_matches.clear()
        for piece in pieces:
            self._best_matches[piece.identifier] = {
                direction: [] for direction in Direction
            }

        iterations = len(pieces) - 1
        for first_index in range(max(iterations, 0)):
            first_piece = pieces[first_index]
            if progress is not None:
                progress("analysis", first_index + 1, iterations)
            for second_piece in pieces[first_index + 1 :]:
                for axis in EdgeAxis:
                    self._record_match(first_piece, second_piece, axis)
                    self._record_match(second_piece, first_piece, axis)

        for matches_by_direction in self._best_matches.values():
            for matches in matches_by_direction.values():
                matches.sort(key=lambda match: match[1])

    def _record_match(self, first: Piece, second: Piece, axis: EdgeAxis) -> None:
        cost = dissimilarity_measure(first, second, axis)
        self._costs.setdefault((first.identifier, second.identifier), {})[axis] = cost
        first_direction, second_direction = _MATCH_DIRECTIONS[axis]
        self._best_matches[second.identifier][first_direction].append(
            (first.identifier, cost)
        )
        self._best_matches[first.identifier][second_direction].append(
            (second.identifier, cost)
        )

    def cost(self, ids: tuple[int, int], axis: EdgeAxis) -> float:
        return self._costs[ids][axis]

    def best_match(self, piece_id: int, direction: Direction) -> int:
        matches = self._best_matches[piece_id][direction]
        if not matches:
            raise ValueError(f"no best match available for piece {piece_id}")
        return matches[0][0]

    def matches(self, piece_id: int, direction: Direction) -> list[tuple[int, float]]:
        return self._best_matches[piece_id][direction]
