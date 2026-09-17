from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
import pytest

from gaps.domain import EdgeAxis
from gaps.imaging.io import read_image
from gaps.imaging.transforms import flatten_image
from gaps.solver.algorithm import GeneticAlgorithm
from gaps.solver.analysis import EdgeCostTable

_IMAGE_PATH = Path(__file__).resolve().parents[2] / "images" / "demo_puzzle.jpg"
_PIECE_SIZE = 32
_PIECE_COUNT = 930
_POPULATION_SIZE = 16
_GENERATIONS = 1
_GENERATION_PIECE_SIZE = 64
_GENERATION_PIECE_COUNT = 225
_GENERATION_POPULATION_SIZE = 600
_GENERATION_COUNT = 5
_GENERATION_WORKERS = 2
_SEED = 1
_BENCHMARK_ROUNDS = 5
_END_TO_END_EXPECTED_SCORE = 0.417199844455363
_END_TO_END_EXPECTED_ARRANGEMENT_DIGEST = (
    "aa916936b45a31f0b012db0a1a38b2a9d8bafa3fab077885abcfa70aa72aad60"
)
_GENERATION_SERIAL_EXPECTED_SCORE = 0.4720038672779986
_GENERATION_SERIAL_EXPECTED_ARRANGEMENT_DIGEST = (
    "a29fdbe8e7a9d4f90b9d3c0760c582a2be68a9691899f78dd93489a0702576e6"
)
_GENERATION_PROCESS_EXPECTED_SCORE = 0.4642578469491072
_GENERATION_PROCESS_EXPECTED_ARRANGEMENT_DIGEST = (
    "da65368d2a0716c89c509c63252f2c9ac5ab740e296442afe65790bf39bdd6b5"
)


@pytest.fixture(scope="module")
def demo_puzzle():
    return read_image(_IMAGE_PATH)


@pytest.fixture(scope="module")
def indexed_pieces(demo_puzzle):
    pieces, layout = flatten_image(
        demo_puzzle,
        piece_size=_PIECE_SIZE,
        indexed=True,
    )
    assert layout.piece_count == _PIECE_COUNT
    return pieces


def _analyze_edge_costs(pieces):
    analysis = EdgeCostTable()
    analysis.analyze(pieces)
    return analysis


def _solve_end_to_end_serial(image):
    return GeneticAlgorithm(
        image=image,
        piece_size=_PIECE_SIZE,
        population_size=_POPULATION_SIZE,
        generations=_GENERATIONS,
        rng=random.Random(_SEED),
    ).solve()


def _solve_generation_heavy_serial(image):
    return GeneticAlgorithm(
        image=image,
        piece_size=_GENERATION_PIECE_SIZE,
        population_size=_GENERATION_POPULATION_SIZE,
        generations=_GENERATION_COUNT,
        rng=random.Random(_SEED),
    ).solve()


def _solve_generation_heavy_process_pool(image):
    return GeneticAlgorithm(
        image=image,
        piece_size=_GENERATION_PIECE_SIZE,
        population_size=_GENERATION_POPULATION_SIZE,
        generations=_GENERATION_COUNT,
        rng=random.Random(_SEED),
        workers=_GENERATION_WORKERS,
    ).solve()


def _arrangement_digest(result) -> str:
    identifiers = np.asarray(
        [piece.identifier for piece in result.arrangement.pieces],
        dtype=np.int32,
    )
    return hashlib.sha256(identifiers.tobytes()).hexdigest()


@pytest.mark.benchmark
def test_demo_puzzle_edge_analysis_930_pieces(benchmark, indexed_pieces):
    """Benchmark the one-time pairwise analysis for the 930-piece puzzle."""
    analysis = benchmark.pedantic(
        _analyze_edge_costs,
        args=(indexed_pieces,),
        rounds=_BENCHMARK_ROUNDS,
        warmup_rounds=0,
        iterations=1,
    )

    assert analysis.cost((0, 1), EdgeAxis.HORIZONTAL) >= 0.0


@pytest.mark.benchmark
def test_demo_puzzle_generation_heavy_serial(benchmark, demo_puzzle):
    """Benchmark repeated crossover and scoring work."""
    result = benchmark.pedantic(
        _solve_generation_heavy_serial,
        args=(demo_puzzle,),
        rounds=_BENCHMARK_ROUNDS,
        warmup_rounds=0,
        iterations=1,
    )

    identifiers = [piece.identifier for piece in result.arrangement.pieces]
    assert sorted(identifiers) == list(range(_GENERATION_PIECE_COUNT))
    assert result.generations_completed == _GENERATION_COUNT
    assert result.best_score == pytest.approx(
        _GENERATION_SERIAL_EXPECTED_SCORE,
        abs=1e-10,
    )
    assert _arrangement_digest(result) == _GENERATION_SERIAL_EXPECTED_ARRANGEMENT_DIGEST


@pytest.mark.benchmark
def test_demo_puzzle_generation_heavy_process_pool(benchmark, demo_puzzle):
    """Benchmark process-based child generation with compact tasks."""
    result = benchmark.pedantic(
        _solve_generation_heavy_process_pool,
        args=(demo_puzzle,),
        rounds=_BENCHMARK_ROUNDS,
        warmup_rounds=0,
        iterations=1,
    )

    identifiers = [piece.identifier for piece in result.arrangement.pieces]
    assert sorted(identifiers) == list(range(_GENERATION_PIECE_COUNT))
    assert result.generations_completed == _GENERATION_COUNT
    assert result.best_score == pytest.approx(
        _GENERATION_PROCESS_EXPECTED_SCORE,
        abs=1e-10,
    )
    assert (
        _arrangement_digest(result) == _GENERATION_PROCESS_EXPECTED_ARRANGEMENT_DIGEST
    )


@pytest.mark.benchmark
def test_demo_puzzle_end_to_end_serial(benchmark, demo_puzzle):
    """Benchmark a deterministic end-to-end solver run."""
    result = benchmark.pedantic(
        _solve_end_to_end_serial,
        args=(demo_puzzle,),
        rounds=_BENCHMARK_ROUNDS,
        warmup_rounds=0,
        iterations=1,
    )

    identifiers = [piece.identifier for piece in result.arrangement.pieces]
    assert sorted(identifiers) == list(range(_PIECE_COUNT))
    assert result.generations_completed == _GENERATIONS
    assert result.best_score == pytest.approx(
        _END_TO_END_EXPECTED_SCORE,
        abs=1e-10,
    )
    assert _arrangement_digest(result) == _END_TO_END_EXPECTED_ARRANGEMENT_DIGEST
