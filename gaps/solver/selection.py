"""Selection strategies for the genetic algorithm."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence

from gaps.domain import Arrangement


def roulette_selection(
    population: Sequence[Arrangement],
    score: Callable[[Arrangement], float],
    rng: random.Random,
    elites: int = 4,
) -> list[tuple[Arrangement, Arrangement]]:
    """Select parent pairs with probability proportional to their score."""
    if not population:
        raise ValueError("population must not be empty")
    if not 0 <= elites < len(population):
        raise ValueError("elites must be between zero and population size")

    cumulative_scores: list[float] = []
    total_score = 0.0
    for individual in population:
        total_score += score(individual)
        cumulative_scores.append(total_score)

    def select_individual() -> Arrangement:
        random_value = rng.random() * total_score
        for index, upper_bound in enumerate(cumulative_scores):
            if random_value < upper_bound:
                return population[index]
        return population[-1]

    return [
        (select_individual(), select_individual())
        for _ in range(len(population) - elites)
    ]
