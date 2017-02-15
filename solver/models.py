import numpy as np
import helpers

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

    def __init__(self, pieces, rows, columns):
        self.pieces  = pieces[:]
        self.rows    = rows
        self.columns = columns
        self.fitness = None

        np.random.shuffle(self.pieces)

        # Map piece ID to index in Individual's list
        self.piece_mapping = {piece.id: index for index, piece in enumerate(self.pieces)}

    def __getitem__(self, key):
        return self.pieces[key * self.columns : (key + 1) * self.columns]

    def piece_size(self):
        """Returns single piece size"""
        return self.pieces[0].size

    def to_image(self):
        """Converts individual to showable image"""
        pieces = [piece.image for piece in self.pieces]
        return helpers.assemble_image(pieces, self.rows, self.columns)

    def edges(self, piece):
        """Returns edges as tuple for a given piece

        Return value is a tuple (top, right, down, left) where each value in a tuple
        is ID of a joint piece or None if there are no edges on that side.

        i.e. For a top left piece return value will be something like (None, 1, 2, None)

        Usage::

            >>> ind = Individual(...)
            >>> ind.edges(ind.pieces[0])
            >>> (None, 1, 2, None)

        """

        edge_index = self.piece_mapping[piece.id]

        left  = self.pieces[edge_index - 1].id            if edge_index % self.columns > 0 else None
        right = self.pieces[edge_index + 1].id            if edge_index % self.columns < self.columns - 1 else None
        top   = self.pieces[edge_index - self.columns].id if edge_index >= self.columns else None
        down  = self.pieces[edge_index + self.columns].id if edge_index < (self.rows - 1) * self.columns else None

        return top, right, down, left

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
        self.image = image
        self.id    = index

    def __getitem__(self, index):
        return self.image.__getitem__(index)

    def size(self):
        """Returns piece size"""
        return self.image.shape[0]

    def shape(self):
        """Retursn shape of piece's image"""
        return self.image.shape
