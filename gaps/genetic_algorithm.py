from collections.abc import Sequence

import numpy as np

from gaps import utils
from gaps.crossover import Crossover
from gaps.image_analysis import ImageAnalysis
from gaps.individual import Individual
from gaps.progress_bar import print_progress
from gaps.selection import roulette_selection


class GeneticAlgorithm:
    """Orchestrate population evolution for a single puzzle image."""

    TERMINATION_THRESHOLD = 10

    def __init__(
        self,
        image: np.ndarray,
        piece_size: int,
        population_size: int,
        generations: int,
        elite_size: int = 2,
    ) -> None:
        if population_size <= 0:
            raise ValueError("population_size must be positive")
        if generations <= 0:
            raise ValueError("generations must be positive")
        if not 0 < elite_size < population_size:
            raise ValueError("elite_size must be between zero and population size")

        pieces, rows, columns = utils.flatten_image(image, piece_size, indexed=True)
        self._analysis = ImageAnalysis()
        self._pieces = pieces
        self._generations = generations
        self._elite_size = elite_size
        self._population = [
            Individual(pieces, rows, columns, self._analysis)
            for _ in range(population_size)
        ]

    def start_evolution(self, verbose: bool = False) -> Individual:
        """Run the configured number of generations and return the best result."""
        print(f"=== Pieces:      {len(self._pieces)}\n")
        plot = None
        if verbose:
            from gaps.plot import Plot

            plot = Plot(self._population[0].to_image())

        self._analysis.analyze_image(self._pieces)
        best_individual: Individual | None = None
        best_fitness: float | None = None
        stagnant_generations = 0

        for generation in range(self._generations):
            print_progress(
                generation,
                self._generations - 1,
                prefix="=== Solving puzzle: ",
            )
            next_population = list(self._get_elite_individuals())

            for first_parent, second_parent in roulette_selection(
                self._population, elites=self._elite_size
            ):
                crossover = Crossover(first_parent, second_parent, self._analysis)
                crossover.run()
                next_population.append(crossover.child())

            best_individual = self._best_individual()
            if best_fitness is not None and best_individual.fitness <= best_fitness:
                stagnant_generations += 1
            else:
                best_fitness = best_individual.fitness
                stagnant_generations = 0

            if stagnant_generations >= self.TERMINATION_THRESHOLD:
                print("\n\n=== GA terminated")
                print(
                    "=== There was no improvement for "
                    f"{self.TERMINATION_THRESHOLD} generations"
                )
                return best_individual

            self._population = next_population
            if plot is not None:
                plot.show_fittest(
                    best_individual.to_image(),
                    f"Generation: {generation + 1} / {self._generations}",
                )

        assert best_individual is not None
        return best_individual

    def _get_elite_individuals(self) -> Sequence[Individual]:
        """Return the fittest individuals preserved for the next generation."""
        return sorted(self._population, key=lambda individual: individual.fitness)[
            -self._elite_size :
        ]

    def _best_individual(self) -> Individual:
        """Return the fittest individual in the current population."""
        return max(self._population, key=lambda individual: individual.fitness)
