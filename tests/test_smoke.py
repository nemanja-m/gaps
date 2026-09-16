import random

import numpy as np
import pytest

from gaps.imaging.io import read_image
from gaps.imaging.transforms import assemble_image, flatten_image
from gaps.solver.algorithm import GeneticAlgorithm

GENERATIONS = 3
POPULATION = 100
PIECE_SIZE = 128

image = read_image("images/baboon.jpg")


@pytest.fixture
def puzzle():
    pieces, layout = flatten_image(image, PIECE_SIZE)
    random.Random(0).shuffle(pieces)
    return assemble_image(pieces, layout)


def test_puzzle_solver(puzzle):
    algorithm = GeneticAlgorithm(
        puzzle,
        PIECE_SIZE,
        POPULATION,
        GENERATIONS,
        rng=random.Random(0),
    )
    result = algorithm.solve()

    solved_image = assemble_image(result.arrangement.pieces, result.arrangement.layout)
    assert np.array_equal(image, solved_image)
