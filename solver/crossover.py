import random

from solver.cache import Cache
from solver.models import Individual

class Crossover(object):

    def __init__(self, first_parent, second_parent):
        self._parents = (first_parent, second_parent)
        self._pieces_length = len(first_parent.pieces)
        self._child_rows = first_parent.rows
        self._child_columns = first_parent.columns

        # Borders of growing kernel
        self._min_row = 0
        self._max_row = 0
        self._min_column = 0
        self._max_column = 0

        self._kernel = {}
        self._taken_positions = set()

        self._shared_piece_candidates = {}
        self._buddy_piece_candidates = {}
        self._best_match_piece_candidates = {}

    def child(self):
        pieces = [None] * self._pieces_length

        for piece, (row, column) in self._kernel.iteritems():
            index = (row - self._min_row) * self._child_columns + (column - self._min_column)
            pieces[index] = self._parents[0].piece_by_id(piece)

        return Individual(pieces, self._child_rows, self._child_columns, shuffle=False)

    def start(self):
        self._initialize_kernel()

        while not self._is_kernel_full():
            while len(self._shared_piece_candidates) > 0:
                position, piece_id = self._shared_piece_candidates.popitem()
                self._assert_position(position, piece_id)
                self._put_piece_to_kernel(piece_id, position)

            if len(self._buddy_piece_candidates) > 0:
                position, piece_id = self._buddy_piece_candidates.popitem()
                self._assert_position(position, piece_id)
                self._put_piece_to_kernel(piece_id, position)
                continue

            if len(self._best_match_piece_candidates) > 0:
                position, piece_id = self._best_match_piece_candidates.popitem()
                self._assert_position(position, piece_id)
                self._put_piece_to_kernel(piece_id, position)

        return self

    def _initialize_kernel(self):
        root_piece = self._parents[0].pieces[int(random.uniform(0, self._pieces_length))]
        self._put_piece_to_kernel(root_piece.id, (0, 0))

    def _put_piece_to_kernel(self, piece_id, position):
        self._kernel[piece_id] = position
        self._taken_positions.add(position)
        self._update_candidate_pieces(piece_id, position)

    def _update_candidate_pieces(self, piece_id, position):
        available_boundaries = self._available_boundaries(position)

        for orientation, position in available_boundaries:
            shared_piece = self._get_shared_piece(piece_id, orientation)
            if self._is_valid_piece(shared_piece):
                self._add_shared_piece_candidate(shared_piece, position)
                continue

            buddy_piece = self._get_buddy_piece(piece_id, orientation)
            if self._is_valid_piece(buddy_piece):
                self._add_buddy_piece_candidate(buddy_piece, position)
                continue

            best_match_piece = self._get_best_match_piece(piece_id, orientation)
            if self._is_valid_piece(best_match_piece):
                self._add_best_match_piece_candidate(best_match_piece, position)
                continue

    def _get_shared_piece(self, piece_id, orientation):
        first_parent, second_parent = self._parents
        first_parent_edge = first_parent.edge(piece_id, orientation)
        second_parent_edge = second_parent.edge(piece_id, orientation)

        if first_parent_edge == second_parent_edge:
            return first_parent_edge

    def _get_buddy_piece(self, piece_id, orientation):
        first_buddy = Cache.best_match(piece_id, orientation)
        second_buddy = Cache.best_match(first_buddy, complementary_orientation(orientation))

        if second_buddy == piece_id:
            for edge in [parent.edge(piece_id, orientation) for parent in self._parents]:
                if edge == first_buddy:
                    return edge

    def _get_best_match_piece(self, piece_id, orientation):
        for piece, _ in Cache.best_match_table[piece_id][orientation]:
            if self._is_valid_piece(piece):
                return piece

    def _add_shared_piece_candidate(self, piece_id, position):
        self._shared_piece_candidates[position] = piece_id

    def _add_buddy_piece_candidate(self, piece_id, position):
        self._buddy_piece_candidates[position] = piece_id

    def _add_best_match_piece_candidate(self, piece_id, position):
        self._best_match_piece_candidates[position] = piece_id

    def _available_boundaries(self, (row, column)):
        boundaries = []

        if not self._is_kernel_full():
            positions = {
                "T": (row - 1, column),
                "R": (row, column + 1),
                "D": (row + 1, column),
                "L": (row, column - 1)
            }

            for orientation, position in positions.iteritems():
                if position not in self._taken_positions and \
                   self._is_in_range(position) and \
                   position not in self._shared_piece_candidates and \
                   position not in self._buddy_piece_candidates and \
                   position not in self._best_match_piece_candidates:
                    self._update_kernel_boundaries(position)
                    boundaries.append((orientation, position))

        return boundaries

    def _is_kernel_full(self):
        return len(self._kernel) == self._pieces_length

    def _is_in_range(self, (row, column)):
        return self._is_row_in_range(row) and self._is_column_in_range(column)

    def _is_row_in_range(self, row):
        current_rows = abs(min(self._min_row, row)) + abs(max(self._max_row, row))
        return current_rows < self._child_rows

    def _is_column_in_range(self, column):
        current_columns = abs(min(self._min_column, column)) + abs(max(self._max_column, column))
        return current_columns < self._child_columns

    def _update_kernel_boundaries(self, (row, column)):
        self._min_row = min(self._min_row, row)
        self._max_row = max(self._max_row, row)
        self._min_column = min(self._min_column, column)
        self._max_column = max(self._max_column, column)

    def _is_valid_piece(self, piece_id):
        return piece_id is not None and \
               piece_id not in self._kernel and \
               piece_id not in self._shared_piece_candidates.values() and \
               piece_id not in self._buddy_piece_candidates.values() and \
               piece_id not in self._best_match_piece_candidates.values()

    def _assert_position(self, position, piece_id):
        if position in self._taken_positions:
            raise Exception("Position already in kernel.")
        if piece_id in self._kernel:
            raise Exception("Piece already in kernel.")

def complementary_orientation(orientation):
    return {
        "T": "D",
        "R": "L",
        "D": "T",
        "L": "R"
    }.get(orientation, None)

