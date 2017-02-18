import numpy as np
import helpers

from solver.fitness import dissimilarity_measure

class Individual:
    """Class representing possible solution to puzzle.

    Individual object is one of the solutions to the problem (possible arrangement of the puzzle's pieces).
    It is created by random shuffling initial puzzle.

    :param pieces:  Array of pieces representing initial puzzle.
    :param rows:    Number of rows in input puzzle
    :param columns: Number of columns in input puzzle

    Usage::

        >>> from models import Individual
        >>> ind = Individual(pieces, 10, 15)

    """

    def __init__(self, pieces, rows, columns, shuffle=True):
        self.pieces  = pieces[:]
        self.rows    = rows
        self.columns = columns
        self.fitness = None

        if shuffle:
            np.random.shuffle(self.pieces)

        # Map piece ID to index in Individual's list
        self.piece_mapping = {piece.id: index for index, piece in enumerate(self.pieces)}

    def __getitem__(self, key):
        return self.pieces[key * self.columns : (key + 1) * self.columns]

    def piece_size(self):
        """Returns single piece size"""
        return self.pieces[0].size

    def piece_by_id(self, identifier):
        return self.pieces[self.piece_mapping[identifier]]

    def to_image(self):
        """Converts individual to showable image"""
        pieces = [piece.image for piece in self.pieces]
        return helpers.assemble_image(pieces, self.rows, self.columns)

    def edges(self, piece):
        """Returns edges as tuple for a given piece

        Return value is a tuple (top, right, down, left) where each value in a tuple
        is ID of a joint piece or None if there are no edges on that side.

        i.e. For a top left piece return value will be something like (None, 1, 2, None)

        :params piece: Piece's ID as source of the edges.

        Usage::

            >>> ind = Individual(...)
            >>> ind.edges(ind.pieces[0])
            >>> (None, 1, 2, None)

        """

        edge_index = self.piece_mapping[piece]
        edges = []

        # Top edge
        if edge_index >= self.columns:
            edge_piece = self.pieces[edge_index - self.columns].id
            weight     = dissimilarity_measure(self.piece_by_id(edge_piece), self.piece_by_id(piece), orientation="TD")

            edges.append((piece, edge_piece, "T", weight))

        # Right edge
        if edge_index % self.columns < self.columns - 1:
            edge_piece = self.pieces[edge_index + 1].id
            weight     = dissimilarity_measure(self.piece_by_id(piece), self.piece_by_id(edge_piece), orientation="LR")

            edges.append((piece, edge_piece, "R", weight))

        # Down edge
        if edge_index < (self.rows - 1) * self.columns:
            edge_piece = self.pieces[edge_index + self.columns].id
            weight     = dissimilarity_measure(self.piece_by_id(piece), self.piece_by_id(edge_piece), orientation="TD")

            edges.append((piece, edge_piece, "D", weight))

        # Left edge
        if edge_index % self.columns > 0:
            edge_piece = self.pieces[edge_index - 1].id
            weight     = dissimilarity_measure(self.piece_by_id(edge_piece), self.piece_by_id(piece), orientation="LR")

            edges.append((piece, edge_piece, "L", weight))

        return edges

    def contains_edge(self, src, dst, orientation):
        edges = [edge for edge in self.edges(src) if edge[1] == dst and edge[2] == orientation]
        return len(edges) > 0

    def best_buddies(self, first_piece, second_piece):
        """For two adjacent pieces returns if they are best buddies

        Two pieces are said to be best-buddies if each piece considers the other
        as its most compatible piece according to the compatibility measure defined

        :params first_piece:  First buddy.
        :params second_piece: Second buddy.

        Usage::

            >>> ind = Individual(...)
            >>> ind.best_buddies(1, 2)
            >>> True

        """

        fp_edges = self.edges(first_piece)
        sp_edges = self.edges(second_piece)

        fp_best_buddy = min(fp_edges, key=lambda edge: edge[3])[1]
        sp_best_buddy = min(sp_edges, key=lambda edge: edge[3])[1]

        return fp_best_buddy == second_piece and sp_best_buddy == first_piece

class Piece:
    """Represents single jigsaw puzzle piece.

    Each piece has identifier so it can be
    tracked accross different individuals

    :param value: ndarray representing piece's RGB values
    :param index: Unique id withing piece's parent image

    Usage::

        >>> from models import Piece
        >>> piece = Piece(image[:28, :28, :], 42)

    """

    def __init__(self, image, index):
        self.image = image[:]
        self.id    = index

    def __getitem__(self, index):
        return self.image.__getitem__(index)

    def size(self):
        """Returns piece size"""
        return self.image.shape[0]

    def shape(self):
        """Retursn shape of piece's image"""
        return self.image.shape
