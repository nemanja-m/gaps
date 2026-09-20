import random

import numpy as np
import pytest

from gaps.domain import PuzzleLayout
from gaps.imaging.transforms import assemble_image, flatten_image
from gaps.solver.algorithm import GeneticAlgorithm, build_seed_arrangements
from gaps.solver.analysis import EdgeCostTable


def test_solver_returns_result_and_reports_progress():
    image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    events: list[tuple[str, int, int]] = []

    result = GeneticAlgorithm(
        image, piece_size=2, population_size=4, generations=2, rng=random.Random(7)
    ).solve(
        progress=lambda phase, current, total: events.append((phase, current, total))
    )

    assert result.generations_completed == 2
    assert result.arrangement.layout == PuzzleLayout(2, 2, 2)
    assert result.best_score > 0
    assert any(phase == "analysis" for phase, _, _ in events)
    assert any(phase == "evolution" for phase, _, _ in events)


def test_phase_one_search_keeps_global_best_valid_after_restart():
    image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    result = GeneticAlgorithm(
        image,
        piece_size=2,
        population_size=8,
        generations=8,
        rng=random.Random(31),
        mutation_rate=0.2,
        local_search_steps=3,
        local_search_candidates=6,
        repair_steps=2,
        max_restarts=1,
        restart_threshold=1,
    ).solve()

    pieces, _ = flatten_image(image, piece_size=2, indexed=True)
    analysis = EdgeCostTable()
    analysis.analyze(pieces)
    identifiers = [piece.identifier for piece in result.arrangement.pieces]

    assert result.best_score == pytest.approx(result.arrangement.score(analysis))
    assert sorted(identifiers) == list(range(16))


def test_phase_two_seeds_are_valid_and_reproducible():
    image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    pieces, layout = flatten_image(image, piece_size=2, indexed=True)
    analysis = EdgeCostTable()
    analysis.analyze(pieces)

    first = build_seed_arrangements(
        pieces,
        layout,
        analysis,
        random.Random(17),
        count=3,
        beam_width=3,
        candidate_width=5,
    )
    second = build_seed_arrangements(
        pieces,
        layout,
        analysis,
        random.Random(17),
        count=3,
        beam_width=3,
        candidate_width=5,
    )

    first_ids = [tuple(piece.identifier for piece in seed.pieces) for seed in first]
    second_ids = [tuple(piece.identifier for piece in seed.pieces) for seed in second]
    assert first_ids == second_ids
    assert all(sorted(ids) == list(range(16)) for ids in first_ids)


def test_solver_collects_optional_crossover_stats():
    image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    solver = GeneticAlgorithm(
        image,
        piece_size=2,
        population_size=8,
        generations=2,
        rng=random.Random(11),
        collect_stats=True,
    )

    solver.solve()

    assert solver.crossover_stats is not None
    assert solver.crossover_stats.placed_pieces == (8 - 2) * 2 * 16
    assert solver.crossover_stats.candidate_pushes > 0
    assert (
        solver.crossover_stats.candidate_pops == solver.crossover_stats.candidate_pushes
    )
    assert solver.crossover_stats.best_match_scanned > 0


def test_solver_can_generate_children_in_worker_processes():
    image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)

    first = GeneticAlgorithm(
        image,
        piece_size=2,
        population_size=8,
        generations=2,
        rng=random.Random(11),
        workers=2,
    ).solve()
    second = GeneticAlgorithm(
        image,
        piece_size=2,
        population_size=8,
        generations=2,
        rng=random.Random(11),
        workers=2,
    ).solve()

    first_ids = tuple(piece.identifier for piece in first.arrangement.pieces)
    second_ids = tuple(piece.identifier for piece in second.arrangement.pieces)
    assert first_ids == second_ids
    assert sorted(first_ids) == list(range(16))


def test_solver_reconstructs_a_grayscale_puzzle():
    base = np.array(
        [
            [12, 180, 45],
            [230, 72, 154],
            [91, 211, 28],
        ],
        dtype=np.uint8,
    )
    image = base[np.ix_([0, 1, 1, 2], [0, 1, 1, 2])]
    pieces, layout = flatten_image(image, piece_size=2)
    random.Random(23).shuffle(pieces)
    puzzle = assemble_image(pieces, layout)

    result = GeneticAlgorithm(
        puzzle,
        piece_size=2,
        population_size=30,
        generations=20,
        rng=random.Random(23),
    ).solve()

    solved_image = assemble_image(result.arrangement.pieces, result.arrangement.layout)
    assert np.array_equal(image, solved_image)
