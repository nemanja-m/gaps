from __future__ import annotations

import heapq
import random
from dataclasses import dataclass
from typing import TypeGuard

from gaps.domain import Arrangement, Direction, Piece
from gaps.solver.analysis import EdgeCostTable

type Position = tuple[int, int]
type RelativePiece = tuple[int, Direction]
type Candidate = tuple[float, int, int, int, int, int, Direction]


SHARED_PIECE_PRIORITY = -10.0
BUDDY_PIECE_PRIORITY = -1.0


@dataclass(slots=True)
class CrossoverStats:
    candidate_pushes: int = 0
    candidate_pops: int = 0
    discarded_taken_positions: int = 0
    reexpanded_placed_pieces: int = 0
    placed_pieces: int = 0
    shared_matches: int = 0
    buddy_matches: int = 0
    best_match_candidates: int = 0
    best_match_calls: int = 0
    best_match_scanned: int = 0
    validity_checks: int = 0


class Crossover:
    """Build a child arrangement from two parent arrangements."""

    def __init__(
        self,
        first_parent: Arrangement,
        second_parent: Arrangement,
        analysis: EdgeCostTable,
        rng: random.Random,
        stats: CrossoverStats | None = None,
    ) -> None:
        self._parents = (first_parent, second_parent)
        self._analysis = analysis
        self._rng = rng
        self._stats = stats
        self._pieces_length = len(first_parent.pieces)
        self._layout = first_parent.layout
        self._min_row = 0
        self._max_row = 0
        self._min_column = 0
        self._max_column = 0
        first_parent._build_edge_cache()
        second_parent._build_edge_cache()
        first_edges = first_parent._edge_cache
        second_edges = second_parent._edge_cache
        if first_edges is None or second_edges is None:
            raise RuntimeError("parent edge caches were not initialized")

        self._parent_edges = (first_edges, second_edges)
        self._kernel: dict[int, Position] = {}
        self._taken_positions: set[Position] = set()
        self._candidate_pieces: list[Candidate] = []
        self._candidate_sequence = 0
        self._match_cursors: dict[tuple[int, Direction], int] = {}

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
            (
                _priority,
                row,
                column,
                piece_id,
                _sequence,
                relative_piece_id,
                relative_direction,
            ) = heapq.heappop(self._candidate_pieces)
            position = (row, column)
            if self._stats is not None:
                self._stats.candidate_pops += 1
            if position in self._taken_positions:
                if self._stats is not None:
                    self._stats.discarded_taken_positions += 1
                continue
            if piece_id in self._kernel:
                if self._stats is not None:
                    self._stats.reexpanded_placed_pieces += 1
                self.add_piece_candidate(
                    relative_piece_id,
                    relative_direction,
                    position,
                )
                continue
            self._put_piece_to_kernel(piece_id, position)

    def _initialize_kernel(self) -> None:
        root_piece = self._rng.choice(self._parents[0].pieces)
        self._put_piece_to_kernel(root_piece.identifier, (0, 0))

    def _put_piece_to_kernel(self, piece_id: int, position: Position) -> None:
        if self._stats is not None:
            self._stats.placed_pieces += 1
        self._kernel[piece_id] = position
        self._taken_positions.add(position)
        for direction, candidate_position in self._available_boundaries(position):
            self.add_piece_candidate(piece_id, direction, candidate_position)

    def add_piece_candidate(
        self, piece_id: int, direction: Direction, position: Position
    ) -> None:
        shared_piece = self._get_shared_piece(piece_id, direction)
        if self._is_valid_piece(shared_piece):
            if self._stats is not None:
                self._stats.shared_matches += 1
            self._push_candidate(
                SHARED_PIECE_PRIORITY, shared_piece, position, (piece_id, direction)
            )
            return

        buddy_piece = self._get_buddy_piece(piece_id, direction)
        if self._is_valid_piece(buddy_piece):
            if self._stats is not None:
                self._stats.buddy_matches += 1
            self._push_candidate(
                BUDDY_PIECE_PRIORITY, buddy_piece, position, (piece_id, direction)
            )
            return

        best_piece, priority = self._get_best_match_piece(piece_id, direction)
        if self._stats is not None:
            self._stats.best_match_candidates += 1
        if self._is_valid_piece(best_piece) and priority is not None:
            self._push_candidate(priority, best_piece, position, (piece_id, direction))

    def _get_shared_piece(self, piece_id: int, direction: Direction) -> int | None:
        first_edge = self._parent_edges[0][direction][piece_id]
        second_edge = self._parent_edges[1][direction][piece_id]
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
        if (
            self._parent_edges[0][direction][piece_id] == first_buddy
            or self._parent_edges[1][direction][piece_id] == first_buddy
        ):
            return first_buddy
        return None

    def _get_best_match_piece(
        self, piece_id: int, direction: Direction
    ) -> tuple[int | None, float | None]:
        matches = self._analysis.matches(piece_id, direction)
        if self._stats is not None:
            self._stats.best_match_calls += 1
        key = (piece_id, direction)
        cursor = self._match_cursors.get(key, 0)
        match_count = len(matches)
        while cursor < match_count:
            candidate_piece, cost = matches[cursor]
            if self._stats is not None:
                self._stats.best_match_scanned += 1
            if self._is_valid_piece(candidate_piece):
                self._match_cursors[key] = cursor
                return candidate_piece, cost
            cursor += 1
        self._match_cursors[key] = cursor
        return None, None

    def _push_candidate(
        self,
        priority: float,
        piece_id: int,
        position: Position,
        relative_piece: RelativePiece,
    ) -> None:
        self._candidate_sequence += 1
        if self._stats is not None:
            self._stats.candidate_pushes += 1
        heapq.heappush(
            self._candidate_pieces,
            (
                priority,
                position[0],
                position[1],
                piece_id,
                self._candidate_sequence,
                relative_piece[0],
                relative_piece[1],
            ),
        )

    def _available_boundaries(
        self, position: Position
    ) -> list[tuple[Direction, Position]]:
        row, column = position
        if self._is_kernel_full():
            return []

        candidates = (
            (Direction.TOP, (row - 1, column)),
            (Direction.RIGHT, (row, column + 1)),
            (Direction.BOTTOM, (row + 1, column)),
            (Direction.LEFT, (row, column - 1)),
        )
        boundaries = []
        for direction, candidate_position in candidates:
            if candidate_position in self._taken_positions:
                continue
            if not self._is_in_range(candidate_position):
                continue
            self._update_kernel_boundaries(candidate_position)
            boundaries.append((direction, candidate_position))
        return boundaries

    def _is_kernel_full(self) -> bool:
        return len(self._kernel) == self._pieces_length

    def _is_in_range(self, position: Position) -> bool:
        row, column = position
        return self._is_row_in_range(row) and self._is_column_in_range(column)

    def _is_row_in_range(self, row: int) -> bool:
        if row < self._min_row:
            return self._max_row - row < self._layout.rows
        if row > self._max_row:
            return row - self._min_row < self._layout.rows
        return True

    def _is_column_in_range(self, column: int) -> bool:
        if column < self._min_column:
            return self._max_column - column < self._layout.columns
        if column > self._max_column:
            return column - self._min_column < self._layout.columns
        return True

    def _update_kernel_boundaries(self, position: Position) -> None:
        row, column = position
        if row < self._min_row:
            self._min_row = row
        elif row > self._max_row:
            self._max_row = row
        if column < self._min_column:
            self._min_column = column
        elif column > self._max_column:
            self._max_column = column

    def _is_valid_piece(self, piece_id: int | None) -> TypeGuard[int]:
        if self._stats is not None:
            self._stats.validity_checks += 1
        return piece_id is not None and piece_id not in self._kernel
