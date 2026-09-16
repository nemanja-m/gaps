from typing import cast

import numpy as np
import pytest

from gaps.domain import Direction, EdgeAxis, Piece
from gaps.imaging.transforms import assemble_image, flatten_image
from gaps.solver.analysis import EdgeCostTable
from gaps.solver.fitness import dissimilarity_measure


def test_flatten_and_assemble_round_trip():
    image = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)

    pieces, layout = flatten_image(image, piece_size=2)

    assert (layout.rows, layout.columns) == (2, 3)
    assert np.array_equal(assemble_image(pieces, layout), image)


def test_edge_cost_table_is_scoped_to_one_puzzle():
    pieces = [
        Piece(np.zeros((2, 2, 3), dtype=np.uint8), 0),
        Piece(np.ones((2, 2, 3), dtype=np.uint8), 1),
    ]
    analysis = EdgeCostTable()

    analysis.analyze(pieces)

    assert analysis.best_match(0, Direction.RIGHT) == 1
    assert analysis.best_match(1, Direction.LEFT) == 0
    assert analysis.cost((0, 1), EdgeAxis.HORIZONTAL) == pytest.approx(np.sqrt(6) / 255)
    with pytest.raises(KeyError):
        EdgeCostTable().matches(0, Direction.RIGHT)


def test_dissimilarity_measure_rejects_unknown_axis():
    piece = Piece(np.zeros((2, 2, 3), dtype=np.uint8), 0)

    with pytest.raises(ValueError, match="Unsupported edge orientation"):
        dissimilarity_measure(piece, piece, cast(EdgeAxis, "diagonal"))
