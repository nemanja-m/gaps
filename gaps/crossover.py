import heapq
import random
from typing import Dict, List, Optional, Set, Tuple

from .core import Chromosome, Gene
from .image_analysis import ImageAnalysis

SHARED_PRIORITY = -10
BUDDY_PRIORITY = -1


def crossover(first_parent: Chromosome, second_parent: Chromosome) -> Chromosome:
    """Run crossover operation on two parents."""
    runner = _CrossoverRunner(first_parent, second_parent)
    return runner.result()


class _CrossoverRunner:
    def __init__(self, first_parent: Chromosome, second_parent: Chromosome) -> None:
        self._parents = (first_parent, second_parent)
        self._length = len(first_parent.genes)
        self._rows = first_parent.rows
        self._cols = first_parent.columns

        # Borders of growing kernel
        self._min_row = 0
        self._max_row = 0
        self._min_col = 0
        self._max_col = 0

        self._kernel: Dict[int, Tuple[int, int]] = {}
        self._taken_positions: Set[Tuple[int, int]] = set()

        # Priority queue
        self._candidates = []

        self._run()

    def result(self) -> Chromosome:
        genes = [None] * self._length

        for gene_index, (row, column) in self._kernel.items():
            index = (row - self._min_row) * self._cols + (column - self._min_col)
            genes[index] = self._parents[0].get_gene(gene_index)

        return Chromosome(genes, self._rows, self._cols)

    def _run(self) -> None:
        self._initialize_kernel()

        while len(self._candidates) > 0:
            _, (position, piece_id), relative_piece = heapq.heappop(self._candidates)

            if position in self._taken_positions:
                continue

            # If piece is already placed, find new piece candidate and put it back to
            # priority queue
            if piece_id in self._kernel:
                self._add_candidate(relative_piece[0], relative_piece[1], position)
                continue

            self._put_to_kernel(piece_id, position)

    def _initialize_kernel(self) -> None:
        root_piece = self._parents[0].genes[int(random.uniform(0, self._length))]
        self._put_to_kernel(root_piece.index, (0, 0))

    def _put_to_kernel(self, index: int, position: Tuple[int, int]) -> None:
        self._kernel[index] = position
        self._taken_positions.add(position)
        for orientation, position in self._available_boundaries(position):
            self._add_candidate(index, orientation, position)

    def _available_boundaries(
        self, position: Tuple[int, int]
    ) -> List[Tuple[str, Tuple[int, int]]]:
        (row, column) = position
        boundaries = []

        if not self._is_kernel_full():
            positions = {
                "T": (row - 1, column),
                "R": (row, column + 1),
                "D": (row + 1, column),
                "L": (row, column - 1),
            }

            for orientation, position in positions.items():
                taken = position in self._taken_positions
                in_range = self._is_in_range(position)
                if not taken and in_range:
                    self._update_kernel_boundaries(position)
                    boundaries.append((orientation, position))

        return boundaries

    def _is_kernel_full(self) -> bool:
        return len(self._kernel) == self._length

    def _is_in_range(self, position: Tuple[int, int]) -> bool:
        (row, column) = position
        return self._is_row_in_range(row) and self._is_column_in_range(column)

    def _is_row_in_range(self, row: int) -> bool:
        current_rows = abs(min(self._min_row, row)) + abs(max(self._max_row, row))
        return current_rows < self._rows

    def _is_column_in_range(self, column: int) -> bool:
        current_cols = abs(min(self._min_col, column)) + abs(max(self._max_col, column))
        return current_cols < self._cols

    def _update_kernel_boundaries(self, position: Tuple[int, int]) -> None:
        (row, column) = position
        self._min_row = min(self._min_row, row)
        self._max_row = max(self._max_row, row)
        self._min_col = min(self._min_col, column)
        self._max_col = max(self._max_col, column)

    def _add_candidate(
        self,
        index: int,
        orientation: str,
        position: Tuple[int, int],
    ) -> None:
        shared = self._get_shared(index, orientation)
        if self._is_valid(shared):
            self._add_shared_candidate(shared, position, (index, orientation))
            return

        buddy = self._get_buddy(index, orientation)
        if self._is_valid(buddy):
            self._add_buddy_candidate(buddy, position, (index, orientation))
            return

        best_match, priority = self._get_best_match(index, orientation)
        if self._is_valid(best_match):
            self._add_best_match_candidate(
                best_match,
                position,
                priority,
                (index, orientation),
            )
            return

    def _get_shared(self, index: int, orientation: str) -> Optional[Gene]:
        first_parent, second_parent = self._parents
        first_parent_edge = first_parent.gene_edge_index(index, orientation)
        second_parent_edge = second_parent.gene_edge_index(index, orientation)

        if first_parent_edge == second_parent_edge:
            return first_parent_edge

    def _add_shared_candidate(
        self,
        index: int,
        position: Tuple[int, int],
        relative: Tuple[int, str],
    ) -> None:
        candidate = (SHARED_PRIORITY, (position, index), relative)
        heapq.heappush(self._candidates, candidate)

    def _get_buddy(self, index: int, orientation: str) -> Optional[Gene]:
        first_buddy = ImageAnalysis.best_match(index, orientation)
        second_buddy = ImageAnalysis.best_match(
            first_buddy,
            _COMPLEMENTARY_ORIENTATION.get(orientation),
        )

        if second_buddy == index:
            if any(
                parent.gene_edge_index(index, orientation) == first_buddy
                for parent in self._parents
            ):
                return first_buddy

    def _add_buddy_candidate(
        self,
        index: int,
        position: Tuple[int, int],
        relative: Tuple[int, str],
    ):
        candidate = (BUDDY_PRIORITY, (position, index), relative)
        heapq.heappush(self._candidates, candidate)

    def _get_best_match(
        self,
        index: int,
        orientation: str,
    ) -> Optional[Tuple[int, float]]:
        dissimilarities = ImageAnalysis.best_match_table[index][orientation]
        for index, dissimilarity in dissimilarities:
            if self._is_valid(index):
                return index, dissimilarity

    def _add_best_match_candidate(
        self,
        index: int,
        position: Tuple[int, int],
        priority: int,
        relative: Tuple[int, str],
    ):
        piece_candidate = (priority, (position, index), relative)
        heapq.heappush(self._candidates, piece_candidate)

    def _is_valid(self, index: int) -> bool:
        return index is not None and index not in self._kernel


_COMPLEMENTARY_ORIENTATION: Dict[str, str] = {
    "T": "D",
    "R": "L",
    "D": "T",
    "L": "R",
}
