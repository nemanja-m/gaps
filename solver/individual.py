import numpy as np

class Individual:
    """Class representing possible solution to puzzle.

    Individual object is one of the solutions to the problem (possible arrangement of the puzzle's pieces).
    It is created by random shuffling initial puzzle.

    :param puzzle: 2D matrix representing initial puzzle.

    Usage::

        >>> from individual import Individual
        >>> ind = Individual(puzzle)

    """

    def __init__(self, puzzle):
        permuted_puzzle = np.random.permutation(puzzle)

        for row in permuted_puzzle:
            np.random.shuffle(row)

        self.puzzle = permuted_puzzle
