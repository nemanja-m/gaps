import random

import cv2 as cv
import numpy as np
import pytest

from gaps import utils
from gaps.genetic_algorithm import GeneticAlgorithm


GENERATIONS = 3
POPULATION = 100
PIECE_SIZE = 128

image = cv.imread("images/baboon.jpg")


@pytest.fixture
def seeded_randomness():
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    random.seed(0)
    np.random.seed(0)
    yield
    random.setstate(random_state)
    np.random.set_state(numpy_state)


@pytest.fixture
def puzzle(seeded_randomness):
    pieces, rows, columns = utils.flatten_image(image, PIECE_SIZE)
    np.random.shuffle(pieces)
    return utils.assemble_image(pieces, rows, columns)


def test_puzzle_solver(puzzle):
    algorithm = GeneticAlgorithm(puzzle, PIECE_SIZE, POPULATION, GENERATIONS)
    solution = algorithm.start_evolution(verbose=False)

    assert np.array_equal(image, solution.to_image())
