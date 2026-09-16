from __future__ import annotations

import random
from contextlib import ExitStack

import click

from gaps.display import OpenCVPreview, PreviewError, TerminalProgress
from gaps.domain import Arrangement
from gaps.imaging.detection import SizeDetector
from gaps.imaging.io import ImageIOError, read_image, write_image
from gaps.imaging.transforms import assemble_image, flatten_image
from gaps.solver.algorithm import GeneticAlgorithm

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


def _report_error(error: Exception) -> click.ClickException:
    return click.ClickException(str(error))


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
    "--seed",
    type=int,
    help="Seed for reproducible puzzle solving.",
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
    seed: int | None,
    debug: bool,
) -> None:
    """Solve PUZZLE and write the result to SOLUTION."""
    try:
        input_puzzle = read_image(puzzle)
        if size is None:
            size = SizeDetector(input_puzzle).detect()

        click.echo(f"Population: {population}")
        click.echo(f"Generations: {generations}")
        click.echo(f"Piece size: {size}")

        with ExitStack() as display:
            preview = display.enter_context(OpenCVPreview()) if debug else None
            progress = display.enter_context(TerminalProgress()) if debug else None
            if preview is not None:
                preview.show(input_puzzle, generation=0)

            def on_generation(generation: int, arrangement: Arrangement) -> None:
                if preview is not None:
                    preview.show(
                        assemble_image(arrangement.pieces, arrangement.layout),
                        generation,
                    )

            result = GeneticAlgorithm(
                image=input_puzzle,
                piece_size=size,
                population_size=population,
                generations=generations,
                rng=random.Random(seed),
            ).solve(
                progress=progress,
                on_generation=on_generation if debug else None,
            )
            write_image(
                solution,
                assemble_image(result.arrangement.pieces, result.arrangement.layout),
            )
    except (ImageIOError, PreviewError, ValueError) as error:
        raise _report_error(error) from error

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
@click.option(
    "--seed",
    type=int,
    help="Seed for reproducible piece shuffling.",
)
def create(image: str, puzzle: str, size: int, seed: int | None) -> None:
    """Create a jigsaw puzzle from IMAGE and write it to PUZZLE."""
    try:
        input_image = read_image(image)
        pieces, layout = flatten_image(input_image, size)
        random.Random(seed).shuffle(pieces)
        write_image(puzzle, assemble_image(pieces, layout))
    except (ImageIOError, ValueError) as error:
        raise _report_error(error) from error

    click.echo(f"\nCreated puzzle with {len(pieces)} pieces")


cli.add_command(run, name="run")
cli.add_command(create, name="create")

if __name__ == "__main__":
    cli()
