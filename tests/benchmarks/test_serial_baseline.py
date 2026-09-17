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
_SEED = 1
_BENCHMARK_ROUNDS = 5
_EXPECTED_SCORE = 0.41719985870711995
_EXPECTED_ARRANGEMENT_DIGEST = (
    "aa916936b45a31f0b012db0a1a38b2a9d8bafa3fab077885abcfa70aa72aad60"
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


def _analyze(pieces):
    analysis = EdgeCostTable()
    analysis.analyze(pieces)
    return analysis


def _solve(image):
    return GeneticAlgorithm(
        image=image,
        piece_size=_PIECE_SIZE,
        population_size=_POPULATION_SIZE,
        generations=_GENERATIONS,
        rng=random.Random(_SEED),
    ).solve()


def _arrangement_digest(result) -> str:
    identifiers = np.asarray(
        [piece.identifier for piece in result.arrangement.pieces],
        dtype=np.int32,
    )
    return hashlib.sha256(identifiers.tobytes()).hexdigest()


@pytest.mark.benchmark
def test_demo_puzzle_edge_analysis_baseline(benchmark, indexed_pieces):
    """Benchmark the one-time pairwise analysis for the 930-piece puzzle."""
    analysis = benchmark.pedantic(
        _analyze,
        args=(indexed_pieces,),
        rounds=_BENCHMARK_ROUNDS,
        warmup_rounds=0,
        iterations=1,
    )

    assert analysis.cost((0, 1), EdgeAxis.HORIZONTAL) >= 0.0


@pytest.mark.benchmark
def test_demo_puzzle_solver_baseline(benchmark, demo_puzzle):
    """Benchmark a deterministic end-to-end solver run."""
    result = benchmark.pedantic(
        _solve,
        args=(demo_puzzle,),
        rounds=_BENCHMARK_ROUNDS,
        warmup_rounds=0,
        iterations=1,
    )

    identifiers = [piece.identifier for piece in result.arrangement.pieces]
    assert sorted(identifiers) == list(range(_PIECE_COUNT))
    assert result.generations_completed == _GENERATIONS
    assert result.best_score == pytest.approx(_EXPECTED_SCORE, abs=1e-12)
    assert _arrangement_digest(result) == _EXPECTED_ARRANGEMENT_DIGEST
