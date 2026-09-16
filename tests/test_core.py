import numpy as np
import pytest

from gaps import utils
from gaps.fitness import dissimilarity_measure
from gaps.image_analysis import ImageAnalysis
from gaps.piece import Piece


def test_flatten_and_assemble_round_trip():
    image = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)

    pieces, rows, columns = utils.flatten_image(image, piece_size=2)

    assert (rows, columns) == (2, 3)
    assert np.array_equal(utils.assemble_image(pieces, rows, columns), image)


def test_image_analysis_is_scoped_to_one_puzzle():
    pieces = [
        Piece(np.zeros((2, 2, 3), dtype=np.uint8), 0),
        Piece(np.ones((2, 2, 3), dtype=np.uint8), 1),
    ]
    analysis = ImageAnalysis()

    analysis.analyze_image(pieces)

    assert analysis.best_match(0, "R") == 1
    assert analysis.best_match(1, "L") == 0
    assert analysis.get_dissimilarity((0, 1), "LR") == pytest.approx(np.sqrt(6) / 255)
    assert ImageAnalysis().best_match_table == {}


def test_dissimilarity_measure_rejects_unknown_orientation():
    piece = Piece(np.zeros((2, 2, 3), dtype=np.uint8), 0)

    with pytest.raises(ValueError, match="Unsupported edge orientation"):
        dissimilarity_measure(piece, piece, "diagonal")  # type: ignore[arg-type]
