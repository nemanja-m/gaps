from typing import cast

import numpy as np
import pytest

from gaps.domain import Direction, EdgeAxis, Piece
from gaps.imaging.io import read_image, write_image
from gaps.imaging.transforms import assemble_image, flatten_image
from gaps.solver.analysis import EdgeCostTable
from gaps.solver.fitness import dissimilarity_measure


def test_flatten_and_assemble_round_trip():
    image = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)

    pieces, layout = flatten_image(image, piece_size=2)

    assert (layout.rows, layout.columns) == (2, 3)
    assert np.array_equal(assemble_image(pieces, layout), image)


def test_grayscale_flatten_and_assemble_round_trip():
    image = np.arange(4 * 6, dtype=np.uint8).reshape(4, 6)

    pieces, layout = flatten_image(image, piece_size=2)

    assert (layout.rows, layout.columns) == (2, 3)
    assert pieces[0].shape == (2, 2)
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
    assert analysis.cost((0, 1), EdgeAxis.HORIZONTAL) == pytest.approx(
        dissimilarity_measure(pieces[0], pieces[1], EdgeAxis.HORIZONTAL)
    )
    with pytest.raises(KeyError):
        EdgeCostTable().matches(0, Direction.RIGHT)


def test_edge_cost_table_matches_scalar_fitness_for_both_axes():
    rng = np.random.default_rng(31)
    pieces = [
        Piece(rng.integers(0, 256, (4, 4, 3), dtype=np.uint8), identifier)
        for identifier in range(5)
    ]
    analysis = EdgeCostTable()
    analysis.analyze(pieces)

    for first_index, first_piece in enumerate(pieces):
        for second_index, second_piece in enumerate(pieces):
            if first_index == second_index:
                continue
            for axis in EdgeAxis:
                assert analysis.cost(
                    (first_piece.identifier, second_piece.identifier), axis
                ) == pytest.approx(
                    dissimilarity_measure(first_piece, second_piece, axis),
                    abs=1e-6,
                )


def test_grayscale_fitness_prefers_continuous_edges():
    first = np.zeros((4, 4), dtype=np.uint8)
    first[:, -2] = [5, 15, 25, 35]
    first[:, -1] = [10, 20, 30, 40]

    correct = np.zeros((4, 4), dtype=np.uint8)
    correct[:, 0] = [10, 20, 30, 40]
    correct[:, 1] = [15, 25, 35, 45]

    incorrect = np.zeros((4, 4), dtype=np.uint8)
    incorrect[:, 0] = [40, 30, 20, 10]
    incorrect[:, 1] = [35, 25, 15, 5]

    first_piece = Piece(first, 0)
    correct_piece = Piece(correct, 1)
    incorrect_piece = Piece(incorrect, 2)

    assert dissimilarity_measure(first_piece, correct_piece) == pytest.approx(0.0)
    assert dissimilarity_measure(first_piece, correct_piece) < dissimilarity_measure(
        first_piece, incorrect_piece
    )


def test_read_image_preserves_native_grayscale(tmp_path):
    path = tmp_path / "grayscale.png"
    image = np.arange(16, dtype=np.uint8).reshape(4, 4)

    write_image(path, image)

    loaded = read_image(path)
    assert loaded.ndim == 2
    assert np.array_equal(loaded, image)


def test_dissimilarity_measure_rejects_unknown_axis():
    piece = Piece(np.zeros((2, 2, 3), dtype=np.uint8), 0)

    with pytest.raises(ValueError, match="Unsupported edge orientation"):
        dissimilarity_measure(piece, piece, cast(EdgeAxis, "diagonal"))
