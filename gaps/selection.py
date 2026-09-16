"""Selection strategies for the genetic algorithm."""

import random
from collections.abc import Sequence

from gaps.individual import Individual


def roulette_selection(
    population: Sequence[Individual], elites: int = 4
) -> list[tuple[Individual, Individual]]:
    """Select parent pairs with probability proportional to fitness."""
    if not population:
        raise ValueError("population must not be empty")
    if not 0 <= elites < len(population):
        raise ValueError("elites must be between zero and population size")

    fitness_values = [individual.fitness for individual in population]
    cumulative_fitness = []
    total_fitness = 0.0
    for fitness in fitness_values:
        total_fitness += fitness
        cumulative_fitness.append(total_fitness)

    def select_individual() -> Individual:
        random_value = random.random() * total_fitness
        for index, upper_bound in enumerate(cumulative_fitness):
            if random_value < upper_bound:
                return population[index]
        return population[-1]

    return [
        (select_individual(), select_individual())
        for _ in range(len(population) - elites)
    ]
