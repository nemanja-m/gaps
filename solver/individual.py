import numpy as np
from solver import image_helpers

class Individual(object):
    """Class representing possible solution to puzzle.

    Individual object is one of the solutions to the problem
    (possible arrangement of the puzzle's pieces).
    It is created by random shuffling initial puzzle.

    :param pieces:  Array of pieces representing initial puzzle.
    :param rows:    Number of rows in input puzzle
    :param columns: Number of columns in input puzzle

    Usage::

        >>> from individual import Individual
        >>> ind = Individual(pieces, 10, 15)

    """

    def __init__(self, pieces, rows, columns, shuffle=True):
        self.pieces = pieces[:]
        self.rows = rows
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
        return image_helpers.assemble_image(pieces, self.rows, self.columns)

    def edge(self, piece, orientation):
        edge_index = self.piece_mapping[piece]

        if (orientation == "T") and (edge_index >= self.columns):
            return self.pieces[edge_index - self.columns].id

        if (orientation == "R") and (edge_index % self.columns < self.columns - 1):
            return self.pieces[edge_index + 1].id

        if (orientation == "D") and (edge_index < (self.rows - 1) * self.columns):
            return self.pieces[edge_index + self.columns].id

        if (orientation == "L") and (edge_index % self.columns > 0):
            return self.pieces[edge_index - 1].id
