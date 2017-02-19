import random
from solver.cache import Cache
from solver.models import Individual

class Crossover:

    def __init__(self, first_parent, second_parent):
        self.parents       = (first_parent, second_parent)
        self.pieces_length = len(first_parent.pieces)
        self.child_rows    = first_parent.rows
        self.child_columns = first_parent.columns

        self.child_pieces    = {}
        self.taken_positions = set()
        self._boundaries = set()

        # Borders of growing kernel
        self._min_row    = 0
        self._max_row    = 0
        self._min_column = 0
        self._max_column = 0

        # Dictionary is defined outside functions for speed.
        # This way it's created once, not each time on function call
        self._complementary_orientation = {
            "T": "D", "R": "L",
            "D": "T",
            "L": "R"
        }

        self._shared_candidates = set()
        self._buddy_candidates  = set()

        # Initialize with random piece
        root = first_parent.pieces[int(random.uniform(0, self.pieces_length))].id
        self.child_pieces[root] = (0, 0)
        self.taken_positions.add((0, 0))
        self._boundaries.update(self.__available_boundaries(root, (0, 0)))

        # Start phase one
        self.__select_shared_edges()

    def __select_shared_edges(self):
        if len(self.child_pieces) == self.pieces_length:
            return

        (piece_id, position), boundary = self.__find_shared_edge()

        if self.__valid_piece(piece_id):
            self.__put_piece(piece_id, position, boundary)
            self.__select_shared_edges()
        else:
            self.__select_buddy_edges()

    def __select_buddy_edges(self):
        if len(self.child_pieces) == self.pieces_length:
            return

        (piece_id, position), boundary = self.__find_buddy_edge()

        if self.__valid_piece(piece_id):
            self.__put_piece(piece_id, position, boundary)
            self.__select_shared_edges()
        else:
            self.__select_random_edges()

    def __select_random_edges(self):
        if len(self.child_pieces) == self.pieces_length:
            return

        (piece_id, position), boundary = self.__find_random_edge()

        self.__put_piece(piece_id, position, boundary)
        self.__select_shared_edges()

    def __find_shared_edge(self):
        for index, (piece, orientation, position) in enumerate(self._boundaries):
            shared = self.__get_shared_edge(piece, orientation)

            if self.__valid_piece(shared) and not position in self.taken_positions:
                return (shared, position), (piece, orientation, position)

        return (None, None), None

    def __find_buddy_edge(self):
        for index, (piece, orientation, position) in enumerate(self._boundaries):
            shared = self.__get_buddy_edge(piece, orientation)

            if self.__valid_piece(shared) and not position in self.taken_positions:
                return (shared, position), (piece, orientation, position)

        return (None, None), None

    def __find_random_edge(self):
        for piece, orientation, position in self._boundaries:
            if not position in self.taken_positions:
                return (self.__get_best_match(piece, orientation), position), (piece, orientation, position)

        raise KeyError

    def __get_shared_edge(self, piece, orientation):
        fp_edge = self.parents[0].edge(piece, orientation)
        sp_edge = self.parents[1].edge(piece, orientation)

        if sp_edge == fp_edge:
            return fp_edge

    def __get_buddy_edge(self, piece, orientation):
        first_buddy  = Cache.best_match(piece, orientation)
        second_buddy = Cache.best_match(first_buddy, self.__complementary_orientation(orientation))

        if second_buddy == piece:

            fp_edge = self.parents[0].edge(piece, orientation)
            if fp_edge == first_buddy:
                return fp_edge

            sp_edge = self.parents[1].edge(piece, orientation)
            if sp_edge == first_buddy:
                return sp_edge

    def __get_best_match(self, piece, orientation):
        best_match = Cache.best_match(piece, orientation)

        if best_match in self.child_pieces:
            for match, _ in Cache.best_match_table[piece][orientation]:
                if not match in self.child_pieces:
                    return match

        return best_match

    def __put_piece(self, piece, position, boundary):
        self.child_pieces[piece] = position
        self.taken_positions.add(position)
        self._boundaries.remove(boundary)
        self._boundaries.update(self.__available_boundaries(piece, position))

    def child(self):
        pieces = [None] * self.pieces_length

        for piece, (row, column) in self.child_pieces.iteritems():
            index = (row - self._min_row) * self.child_columns + (column - self._min_column)
            pieces[index] = self.parents[0].piece_by_id(piece)

        return Individual(pieces, self.child_rows, self.child_columns, shuffle=False)

    def __available_boundaries(self, piece, (row, column)):
        if len(self.taken_positions) == self.pieces_length:
            return []

        boundaries  = []

        positions = {
            "T": (row - 1, column),
            "R": (row, column + 1),
            "D": (row + 1, column),
            "L": (row, column - 1)
        }

        for orientation, position in positions.iteritems():
            if not position in self.taken_positions and self.__in_range(position):
                boundaries.append((piece, orientation, position))

        return boundaries

    def __valid_piece(self, piece):
        return piece != None and not piece in self.child_pieces

    def __in_range(self, (row, column)):
        if abs(min(self._min_row, row)) + abs(max(self._max_row, row)) >= self.child_rows:
            return False

        if abs(min(self._min_column, column)) + abs(max(self._max_column, column)) >= self.child_columns:
            return False

        # Update kernel boundaries
        self._min_row    = min(self._min_row, row)
        self._max_row    = max(self._max_row, row)
        self._min_column = min(self._min_column, column)
        self._max_column = max(self._max_column, column)

        return True

    def __complementary_orientation(self, orientation):
        return self._complementary_orientation.get(orientation, None)

