import cv2 as cv
import numpy as np
import pytest

from gaps import utils
from gaps.size_detector import SizeDetector


sizes = [32, 48, 56, 64]

images = ["images/lena.jpg", "images/island.jpg", "images/pillars.jpg"]


def create_puzzle(image_path, piece_size):
    image = cv.imread(image_path)
    pieces, rows, columns = utils.flatten_image(image, piece_size)
    np.random.shuffle(pieces)
    return utils.assemble_image(pieces, rows, columns)


@pytest.mark.parametrize("image", images)
def test_size_detection(image):
    for piece_size in sizes:
        puzzle = create_puzzle(image, piece_size)
        detector = SizeDetector(puzzle)
        assert detector.detect() == piece_size
