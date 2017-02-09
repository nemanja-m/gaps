import numpy as np

class Individual:
    """Class representing possible solution to puzzle.

    Individual object is one of the solutions to the problem (possible arrangement of the puzzle's pieces).
    It is created by random shuffling initial puzzle.

    :param pieces:      Array of pieces representing initial puzzle.
    :param rows:        Number of rows in input puzzle
    :param columns:     Number of columns in input puzzle

    Usage::

        >>> from individual import Individual
        >>> ind = Individual(pieces, 10, 15)

    """

    def __init__(self, pieces, rows, columns):
        self.pieces     = pieces
        self.rows       = rows
        self.columns    = columns

        np.random.shuffle(self.pieces)

    def __getitem__(self, key):
        return self.pieces[key * self.columns : (key + 1) * self.columns]

    def piece_size(self):
        if len(self.pieces) > 0:
            return self.pieces[0].shape[0]
        else:
            return -1
