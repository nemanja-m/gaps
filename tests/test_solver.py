import random

import numpy as np

from gaps.domain import PuzzleLayout
from gaps.solver.algorithm import GeneticAlgorithm


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
