import pytest
import numpy as np

from scipy import misc
from gaps import image_helpers
from gaps.size_detector import SizeDetector


testdata = [
    28, 32, 48, 56, 64
]

@pytest.mark.parametrize("size", testdata)
def test_size_detection(size):
    puzzle = create_puzzle(size)
    detector = SizeDetector(puzzle)
    assert detector.detect_piece_size() == size

def create_puzzle(piece_size):
    image = misc.imread("images/lena.jpg")
    pieces, rows, columns = image_helpers.flatten_image(image, piece_size)
    np.random.shuffle(pieces)
    return image_helpers.assemble_image(pieces, rows, columns)
