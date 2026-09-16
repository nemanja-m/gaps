import heapq
import random
from typing import TypeAlias, TypeGuard, cast

from gaps.image_analysis import ImageAnalysis, Orientation
from gaps.individual import Individual
from gaps.piece import Piece

SHARED_PIECE_PRIORITY = -10.0
BUDDY_PIECE_PRIORITY = -1.0

Position: TypeAlias = tuple[int, int]
RelativePiece: TypeAlias = tuple[int, Orientation]
Candidate: TypeAlias = tuple[float, tuple[Position, int], RelativePiece]


class Crossover:
    """Build a child arrangement from two parent arrangements."""

    def __init__(
        self,
        first_parent: Individual,
        second_parent: Individual,
        analysis: ImageAnalysis,
    ) -> None:
        self._parents = (first_parent, second_parent)
        self._analysis = analysis
        self._pieces_length = len(first_parent.pieces)
        self._child_rows = first_parent.rows
        self._child_columns = first_parent.columns

        self._min_row = 0
        self._max_row = 0
        self._min_column = 0
        self._max_column = 0
        self._kernel: dict[int, Position] = {}
        self._taken_positions: set[Position] = set()
        self._candidate_pieces: list[Candidate] = []

    def child(self) -> Individual:
        """Return the child arrangement after :meth:`run` completes."""
        pieces: list[Piece | None] = [None] * self._pieces_length
        for piece_id, (row, column) in self._kernel.items():
            index = (row - self._min_row) * self._child_columns + (
                column - self._min_column
            )
            pieces[index] = self._parents[0].piece_by_id(piece_id)

        if any(piece is None for piece in pieces):
            raise RuntimeError("crossover did not produce a complete child")

        return Individual(
            [piece for piece in pieces if piece is not None],
            self._child_rows,
            self._child_columns,
            self._analysis,
            shuffle=False,
        )

    def run(self) -> None:
        """Populate the child kernel using parent and image matches."""
        self._initialize_kernel()

        while self._candidate_pieces:
            _, (position, piece_id), relative_piece = heapq.heappop(
                self._candidate_pieces
            )

            if position in self._taken_positions:
                continue
            if piece_id in self._kernel:
                self.add_piece_candidate(relative_piece[0], relative_piece[1], position)
                continue

            self._put_piece_to_kernel(piece_id, position)

    def _initialize_kernel(self) -> None:
        root_piece = random.choice(self._parents[0].pieces)
        self._put_piece_to_kernel(root_piece.id, (0, 0))

    def _put_piece_to_kernel(self, piece_id: int, position: Position) -> None:
        self._kernel[piece_id] = position
        self._taken_positions.add(position)
        self._update_candidate_pieces(piece_id, position)

    def _update_candidate_pieces(self, piece_id: int, position: Position) -> None:
        for orientation, candidate_position in self._available_boundaries(position):
            self.add_piece_candidate(piece_id, orientation, candidate_position)

    def add_piece_candidate(
        self, piece_id: int, orientation: Orientation, position: Position
    ) -> None:
        shared_piece = self._get_shared_piece(piece_id, orientation)
        if self._is_valid_piece(shared_piece):
            self._add_shared_piece_candidate(
                shared_piece, position, (piece_id, orientation)
            )
            return

        buddy_piece = self._get_buddy_piece(piece_id, orientation)
        if self._is_valid_piece(buddy_piece):
            self._add_buddy_piece_candidate(
                buddy_piece, position, (piece_id, orientation)
            )
            return

        best_match_piece, priority = self._get_best_match_piece(piece_id, orientation)
        if self._is_valid_piece(best_match_piece) and priority is not None:
            self._add_best_match_piece_candidate(
                best_match_piece, position, priority, (piece_id, orientation)
            )

    def _get_shared_piece(self, piece_id: int, orientation: Orientation) -> int | None:
        first_edge = self._parents[0].edge(piece_id, orientation)
        second_edge = self._parents[1].edge(piece_id, orientation)
        if first_edge is not None and first_edge == second_edge:
            return first_edge
        return None

    def _get_buddy_piece(self, piece_id: int, orientation: Orientation) -> int | None:
        try:
            first_buddy = self._analysis.best_match(piece_id, orientation)
            second_buddy = self._analysis.best_match(
                first_buddy, complementary_orientation(orientation)
            )
        except ValueError:
            return None

        if second_buddy != piece_id:
            return None

        for parent in self._parents:
            if parent.edge(piece_id, orientation) == first_buddy:
                return first_buddy
        return None

    def _get_best_match_piece(
        self, piece_id: int, orientation: Orientation
    ) -> tuple[int | None, float | None]:
        for candidate_piece, dissimilarity in self._analysis.best_match_table[piece_id][
            orientation
        ]:
            if self._is_valid_piece(candidate_piece):
                return candidate_piece, dissimilarity
        return None, None

    def _add_shared_piece_candidate(
        self, piece_id: int, position: Position, relative_piece: RelativePiece
    ) -> None:
        self._push_candidate(SHARED_PIECE_PRIORITY, piece_id, position, relative_piece)

    def _add_buddy_piece_candidate(
        self, piece_id: int, position: Position, relative_piece: RelativePiece
    ) -> None:
        self._push_candidate(BUDDY_PIECE_PRIORITY, piece_id, position, relative_piece)

    def _add_best_match_piece_candidate(
        self,
        piece_id: int,
        position: Position,
        priority: float,
        relative_piece: RelativePiece,
    ) -> None:
        self._push_candidate(priority, piece_id, position, relative_piece)

    def _push_candidate(
        self,
        priority: float,
        piece_id: int,
        position: Position,
        relative_piece: RelativePiece,
    ) -> None:
        heapq.heappush(
            self._candidate_pieces, (priority, (position, piece_id), relative_piece)
        )

    def _available_boundaries(
        self, position: Position
    ) -> list[tuple[Orientation, Position]]:
        row, column = position
        if self._is_kernel_full():
            return []

        positions: dict[Orientation, Position] = {
            "T": (row - 1, column),
            "R": (row, column + 1),
            "D": (row + 1, column),
            "L": (row, column - 1),
        }
        boundaries = []
        for orientation, candidate_position in positions.items():
            if candidate_position not in self._taken_positions and self._is_in_range(
                candidate_position
            ):
                self._update_kernel_boundaries(candidate_position)
                boundaries.append((orientation, candidate_position))
        return boundaries

    def _is_kernel_full(self) -> bool:
        return len(self._kernel) == self._pieces_length

    def _is_in_range(self, position: Position) -> bool:
        row, column = position
        return self._is_row_in_range(row) and self._is_column_in_range(column)

    def _is_row_in_range(self, row: int) -> bool:
        current_rows = abs(min(self._min_row, row)) + abs(max(self._max_row, row))
        return current_rows < self._child_rows

    def _is_column_in_range(self, column: int) -> bool:
        current_columns = abs(min(self._min_column, column)) + abs(
            max(self._max_column, column)
        )
        return current_columns < self._child_columns

    def _update_kernel_boundaries(self, position: Position) -> None:
        row, column = position
        self._min_row = min(self._min_row, row)
        self._max_row = max(self._max_row, row)
        self._min_column = min(self._min_column, column)
        self._max_column = max(self._max_column, column)

    def _is_valid_piece(self, piece_id: int | None) -> TypeGuard[int]:
        return piece_id is not None and piece_id not in self._kernel


def complementary_orientation(orientation: Orientation) -> Orientation:
    """Return the opposite edge orientation."""
    return cast(Orientation, {"T": "D", "R": "L", "D": "T", "L": "R"}[orientation])
