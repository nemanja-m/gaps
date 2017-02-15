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
        if len(self.pieces) > 0:
            return self.pieces[0].size
        else:
            return -1

    def to_image(self):
        """Converts individual to showable image"""
        pieces = [piece.image for piece in self.pieces]
        return helpers.assemble_image(pieces, self.rows, self.columns)

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
