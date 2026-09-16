from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field
from typing import TypeGuard

from gaps.domain import Arrangement, Direction, Piece
from gaps.solver.analysis import EdgeCostTable

type Position = tuple[int, int]
type RelativePiece = tuple[int, Direction]


@dataclass(order=True, slots=True)
class Candidate:
    priority: float
    position: Position
    piece_id: int
    relative_piece: RelativePiece = field(compare=False)


SHARED_PIECE_PRIORITY = -10.0
BUDDY_PIECE_PRIORITY = -1.0


class Crossover:
    """Build a child arrangement from two parent arrangements."""

    def __init__(
        self,
        first_parent: Arrangement,
        second_parent: Arrangement,
        analysis: EdgeCostTable,
        rng: random.Random,
    ) -> None:
        self._parents = (first_parent, second_parent)
        self._analysis = analysis
        self._rng = rng
        self._pieces_length = len(first_parent.pieces)
        self._layout = first_parent.layout
        self._min_row = 0
        self._max_row = 0
        self._min_column = 0
        self._max_column = 0
        self._kernel: dict[int, Position] = {}
        self._taken_positions: set[Position] = set()
        self._candidate_pieces: list[Candidate] = []

    def child(self) -> Arrangement:
        """Return the child arrangement after :meth:`run` completes."""
        pieces: list[Piece | None] = [None] * self._pieces_length
        for piece_id, (row, column) in self._kernel.items():
            index = (row - self._min_row) * self._layout.columns + (
                column - self._min_column
            )
            pieces[index] = self._parents[0].piece_by_id(piece_id)

        if any(piece is None for piece in pieces):
            raise RuntimeError("crossover did not produce a complete child")
        return Arrangement(
            [piece for piece in pieces if piece is not None], self._layout
        )

    def run(self) -> None:
        """Populate the child kernel using parent and image matches."""
        self._initialize_kernel()
        while self._candidate_pieces:
            candidate = heapq.heappop(self._candidate_pieces)
            if candidate.position in self._taken_positions:
                continue
            if candidate.piece_id in self._kernel:
                self.add_piece_candidate(
                    candidate.relative_piece[0],
                    candidate.relative_piece[1],
                    candidate.position,
                )
                continue
            self._put_piece_to_kernel(candidate.piece_id, candidate.position)

    def _initialize_kernel(self) -> None:
        root_piece = self._rng.choice(self._parents[0].pieces)
        self._put_piece_to_kernel(root_piece.identifier, (0, 0))

    def _put_piece_to_kernel(self, piece_id: int, position: Position) -> None:
        self._kernel[piece_id] = position
        self._taken_positions.add(position)
        for direction, candidate_position in self._available_boundaries(position):
            self.add_piece_candidate(piece_id, direction, candidate_position)

    def add_piece_candidate(
        self, piece_id: int, direction: Direction, position: Position
    ) -> None:
        shared_piece = self._get_shared_piece(piece_id, direction)
        if self._is_valid_piece(shared_piece):
            self._push_candidate(
                SHARED_PIECE_PRIORITY, shared_piece, position, (piece_id, direction)
            )
            return

        buddy_piece = self._get_buddy_piece(piece_id, direction)
        if self._is_valid_piece(buddy_piece):
            self._push_candidate(
                BUDDY_PIECE_PRIORITY, buddy_piece, position, (piece_id, direction)
            )
            return

        best_piece, priority = self._get_best_match_piece(piece_id, direction)
        if self._is_valid_piece(best_piece) and priority is not None:
            self._push_candidate(priority, best_piece, position, (piece_id, direction))

    def _get_shared_piece(self, piece_id: int, direction: Direction) -> int | None:
        first_edge = self._parents[0].edge(piece_id, direction)
        second_edge = self._parents[1].edge(piece_id, direction)
        if first_edge is not None and first_edge == second_edge:
            return first_edge
        return None

    def _get_buddy_piece(self, piece_id: int, direction: Direction) -> int | None:
        try:
            first_buddy = self._analysis.best_match(piece_id, direction)
            second_buddy = self._analysis.best_match(first_buddy, direction.opposite)
        except ValueError:
            return None

        if second_buddy != piece_id:
            return None
        if any(
            parent.edge(piece_id, direction) == first_buddy for parent in self._parents
        ):
            return first_buddy
        return None

    def _get_best_match_piece(
        self, piece_id: int, direction: Direction
    ) -> tuple[int | None, float | None]:
        for candidate_piece, cost in self._analysis.matches(piece_id, direction):
            if self._is_valid_piece(candidate_piece):
                return candidate_piece, cost
        return None, None

    def _push_candidate(
        self,
        priority: float,
        piece_id: int,
        position: Position,
        relative_piece: RelativePiece,
    ) -> None:
        heapq.heappush(
            self._candidate_pieces,
            Candidate(priority, position, piece_id, relative_piece),
        )

    def _available_boundaries(
        self, position: Position
    ) -> list[tuple[Direction, Position]]:
        row, column = position
        if self._is_kernel_full():
            return []

        positions = {
            Direction.TOP: (row - 1, column),
            Direction.RIGHT: (row, column + 1),
            Direction.BOTTOM: (row + 1, column),
            Direction.LEFT: (row, column - 1),
        }
        boundaries = []
        for direction, candidate_position in positions.items():
            if candidate_position not in self._taken_positions and self._is_in_range(
                candidate_position
            ):
                self._update_kernel_boundaries(candidate_position)
                boundaries.append((direction, candidate_position))
        return boundaries

    def _is_kernel_full(self) -> bool:
        return len(self._kernel) == self._pieces_length

    def _is_in_range(self, position: Position) -> bool:
        row, column = position
        return self._is_row_in_range(row) and self._is_column_in_range(column)

    def _is_row_in_range(self, row: int) -> bool:
        current_rows = abs(min(self._min_row, row)) + abs(max(self._max_row, row))
        return current_rows < self._layout.rows

    def _is_column_in_range(self, column: int) -> bool:
        current_columns = abs(min(self._min_column, column)) + abs(
            max(self._max_column, column)
        )
        return current_columns < self._layout.columns

    def _update_kernel_boundaries(self, position: Position) -> None:
        row, column = position
        self._min_row = min(self._min_row, row)
        self._max_row = max(self._max_row, row)
        self._min_column = min(self._min_column, column)
        self._max_column = max(self._max_column, column)

    def _is_valid_piece(self, piece_id: int | None) -> TypeGuard[int]:
        return piece_id is not None and piece_id not in self._kernel
