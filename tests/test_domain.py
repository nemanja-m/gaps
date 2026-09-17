import random

import numpy as np

from gaps.domain import Arrangement, Direction, EdgeAxis, PuzzleLayout
from gaps.imaging.transforms import assemble_image, flatten_image
from gaps.solver.analysis import EdgeCostTable


def test_domain_types_model_layout_and_directions():
    layout = PuzzleLayout(rows=2, columns=2, piece_size=2)

    assert layout.piece_count == 4
    assert Direction.RIGHT.opposite is Direction.LEFT
    assert EdgeAxis.HORIZONTAL.value == "horizontal"


def test_arrangement_incremental_swap_matches_full_score():
    image = np.arange(6 * 6 * 3, dtype=np.uint8).reshape(6, 6, 3)
    pieces, layout = flatten_image(image, piece_size=2, indexed=True)
    analysis = EdgeCostTable()
    analysis.analyze(pieces)
    arrangement = Arrangement.random(pieces, layout, random.Random(5))
    arrangement.score(analysis)

    arrangement.swap(0, 8, cost_lookup=analysis)

    expected = Arrangement(list(arrangement.pieces), layout).score(analysis.cost)
    assert arrangement.score(analysis) == expected


def test_arrangement_preserves_indexed_pieces_and_layout():
    image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    pieces, layout = flatten_image(image, piece_size=2, indexed=True)
    arrangement = Arrangement(pieces, layout)

    assert arrangement.layout == layout
    assert arrangement.piece_by_id(0).identifier == pieces[0].identifier
    assert np.array_equal(assemble_image(arrangement.pieces, arrangement.layout), image)
