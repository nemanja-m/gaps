from pathlib import Path

import numpy as np
from click.testing import CliRunner

from gaps.cli import cli
from gaps.imaging.io import read_image

IMAGE = Path("images/baboon.jpg")
PIECE_SIZE = 128
GENERATIONS = 10
POPULATION = 100


def test_create_and_run_commands_round_trip(tmp_path):
    puzzle = tmp_path / "puzzle.png"
    solution = tmp_path / "solution.png"
    source_image = read_image(IMAGE)
    runner = CliRunner()

    create_result = runner.invoke(
        cli,
        [
            "create",
            str(IMAGE),
            str(puzzle),
            "--size",
            str(PIECE_SIZE),
            "--seed",
            "0",
        ],
    )
    assert create_result.exit_code == 0, create_result.output
    assert "Created puzzle with 16 pieces" in create_result.output
    assert puzzle.is_file()

    run_result = runner.invoke(
        cli,
        [
            "run",
            str(puzzle),
            str(solution),
            "--size",
            str(PIECE_SIZE),
            "--generations",
            str(GENERATIONS),
            "--population",
            str(POPULATION),
            "--seed",
            "0",
        ],
    )
    assert run_result.exit_code == 0, run_result.output
    assert "Piece size: 128" in run_result.output
    assert "Puzzle solved" in run_result.output
    assert solution.is_file()

    assert np.array_equal(source_image, read_image(solution))
