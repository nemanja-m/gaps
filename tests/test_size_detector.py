import pytest
import numpy as np
import cv2

from gaps import image_helpers
from gaps.size_detector import SizeDetector


sizes = [
    28, 32, 48, 56, 64
]

images = [
    "images/lena.jpg",
    "images/island.jpg",
    "images/pillars.jpg"
]

def create_puzzle(image_path, piece_size):
    image = cv2.imread(image_path)
    pieces, rows, columns = image_helpers.flatten_image(image, piece_size)
    np.random.shuffle(pieces)
    return image_helpers.assemble_image(pieces, rows, columns)

@pytest.mark.parametrize("image", images)
def test_size_detection(image):
    for piece_size in sizes:
        puzzle = create_puzzle(image, piece_size)
        detector = SizeDetector(puzzle)
        assert detector.detect_piece_size() == piece_size
