import random

import pytest

from gaps.imaging.io import read_image
from gaps.imaging.transforms import assemble_image, flatten_image
from gaps.imaging.detection import SizeDetector


sizes = [32, 48, 56, 64]
images = ["images/lena.jpg", "images/island.jpg", "images/pillars.jpg"]


def create_puzzle(image_path: str, piece_size: int):
    image = read_image(image_path)
    pieces, layout = flatten_image(image, piece_size)
    random.Random(piece_size).shuffle(pieces)
    return assemble_image(pieces, layout)


@pytest.mark.parametrize("image", images)
def test_size_detection(image: str):
    for piece_size in sizes:
        puzzle = create_puzzle(image, piece_size)
        detector = SizeDetector(puzzle)
        assert detector.detect() == piece_size
