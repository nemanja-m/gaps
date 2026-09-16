import random
from pathlib import Path

import click
import cv2 as cv
import numpy as np

from gaps import utils
from gaps.genetic_algorithm import GeneticAlgorithm
from gaps.size_detector import SizeDetector

DEFAULT_GENERATIONS = 20
DEFAULT_POPULATION = 200
MIN_PIECE_SIZE = 32
MAX_PIECE_SIZE = 128


@click.group(
    context_settings={
        "help_option_names": ["-h", "--help"],
        "ignore_unknown_options": True,
    }
)
def cli() -> None:
    """Solve or create puzzles with square pieces."""


def _validate_piece_size(_context: click.Context, _param: str, value: int) -> int:
    if not MIN_PIECE_SIZE <= value <= MAX_PIECE_SIZE:
        raise click.BadParameter(
            f"Piece size must be between {MIN_PIECE_SIZE} and {MAX_PIECE_SIZE} pixels"
        )
    return value


def _validate_positive_integer(_context: click.Context, _param: str, value: int) -> int:
    if value <= 0:
        raise click.BadParameter("Should be a positive integer.")
    return value


def _read_image(path: str) -> np.ndarray:
    image = cv.imread(path)
    if image is None:
        raise click.ClickException(f"Could not read image: {path}")
    return image


def _write_image(path: str | Path, image: np.ndarray) -> None:
    if not cv.imwrite(str(path), image):
        raise click.ClickException(f"Could not write image: {path}")


@click.command()
@click.argument("puzzle", type=click.Path(exists=True, readable=True))
@click.argument("solution", type=click.Path(dir_okay=False, writable=True))
@click.option(
    "-s",
    "--size",
    type=int,
    help="Size of single square puzzle piece in pixels. Autodetected if not specified.",
)
@click.option(
    "-g",
    "--generations",
    type=int,
    show_default=True,
    default=DEFAULT_GENERATIONS,
    callback=_validate_positive_integer,
    help="The number of generations for genetic algorithm.",
)
@click.option(
    "-p",
    "--population",
    type=int,
    show_default=True,
    default=DEFAULT_POPULATION,
    callback=_validate_positive_integer,
    help="The size of the initial population for genetic algorithm.",
)
@click.option(
    "-d",
    "--debug",
    is_flag=True,
    help="If enabled, shows the best individual after each generation.",
)
def run(
    puzzle: str,
    solution: str,
    size: int | None,
    generations: int,
    population: int,
    debug: bool,
) -> None:
    """Solve PUZZLE and write the result to SOLUTION."""
    input_puzzle = _read_image(puzzle)
    if size is None:
        size = SizeDetector(input_puzzle).detect()

    click.echo(f"Population: {population}")
    click.echo(f"Generations: {generations}")
    click.echo(f"Piece size: {size}")

    algorithm = GeneticAlgorithm(
        image=input_puzzle,
        piece_size=size,
        population_size=population,
        generations=generations,
    )
    result = algorithm.start_evolution(verbose=debug)
    _write_image(solution, result.to_image())
    click.echo("Puzzle solved")


@click.command()
@click.argument("image", type=click.Path(exists=True, readable=True))
@click.argument("puzzle", type=click.Path(writable=True))
@click.option(
    "-s",
    "--size",
    type=int,
    show_default=True,
    default=MAX_PIECE_SIZE,
    callback=_validate_piece_size,
    help="Size of single square puzzle piece in pixels.",
)
def create(image: str, puzzle: str, size: int) -> None:
    """Create a jigsaw puzzle from IMAGE and write it to PUZZLE."""
    input_image = _read_image(image)
    pieces, rows, columns = utils.flatten_image(input_image, size)
    random.shuffle(pieces)
    _write_image(puzzle, utils.assemble_image(pieces, rows, columns))
    click.echo(f"\nCreated puzzle with {len(pieces)} pieces")


cli.add_command(run, name="run")
cli.add_command(create, name="create")

if __name__ == "__main__":
    cli()
