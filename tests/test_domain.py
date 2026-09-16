import numpy as np

from gaps.domain import Arrangement, Direction, EdgeAxis, PuzzleLayout
from gaps.imaging.transforms import assemble_image, flatten_image


def test_domain_types_model_layout_and_directions():
    layout = PuzzleLayout(rows=2, columns=2, piece_size=2)

    assert layout.piece_count == 4
    assert Direction.RIGHT.opposite is Direction.LEFT
    assert EdgeAxis.HORIZONTAL.value == "horizontal"


def test_arrangement_preserves_indexed_pieces_and_layout():
    image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    pieces, layout = flatten_image(image, piece_size=2, indexed=True)
    arrangement = Arrangement(pieces, layout)

    assert arrangement.layout == layout
    assert arrangement.piece_by_id(0).identifier == pieces[0].identifier
    assert np.array_equal(assemble_image(arrangement.pieces, arrangement.layout), image)
