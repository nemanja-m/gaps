from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from gaps.domain import Arrangement, Image
from gaps.imaging.transforms import flatten_image
from gaps.solver.analysis import EdgeCostTable
from gaps.solver.crossover import Crossover
from gaps.solver.selection import roulette_selection

ProgressCallback = Callable[[str, int, int], None]
GenerationCallback = Callable[[int, Arrangement], None]


@dataclass(frozen=True, slots=True)
class SolveResult:
    arrangement: Arrangement
    generations_completed: int
    best_score: float
    terminated_early: bool


class GeneticAlgorithm:
    """Evolve candidate arrangements for one puzzle image."""

    TERMINATION_THRESHOLD = 10

    def __init__(
        self,
        image: Image,
        piece_size: int,
        population_size: int,
        generations: int,
        elite_size: int = 2,
        rng: random.Random | None = None,
    ) -> None:
        if population_size <= 0:
            raise ValueError("population_size must be positive")
        if generations <= 0:
            raise ValueError("generations must be positive")
        if not 0 < elite_size < population_size:
            raise ValueError("elite_size must be between zero and population size")

        pieces, layout = flatten_image(image, piece_size, indexed=True)
        self._analysis = EdgeCostTable()
        self._pieces = pieces
        self._generations = generations
        self._elite_size = elite_size
        self._rng = rng or random.Random()
        self._population = [
            Arrangement.random(pieces, layout, self._rng)
            for _ in range(population_size)
        ]

    def solve(
        self,
        progress: ProgressCallback | None = None,
        on_generation: GenerationCallback | None = None,
    ) -> SolveResult:
        """Run evolution and return the best arrangement and metadata."""
        self._analysis.analyze(self._pieces, progress=progress)
        best_arrangement: Arrangement | None = None
        best_score: float | None = None
        stagnant_generations = 0

        for generation in range(self._generations):
            next_population = list(self._elite_individuals())
            for first_parent, second_parent in roulette_selection(
                self._population,
                score=self._score,
                rng=self._rng,
                elites=self._elite_size,
            ):
                crossover = Crossover(
                    first_parent, second_parent, self._analysis, self._rng
                )
                crossover.run()
                next_population.append(crossover.child())

            self._population = next_population
            best_arrangement = self._best_arrangement()
            current_score = self._score(best_arrangement)
            if best_score is not None and current_score <= best_score:
                stagnant_generations += 1
            else:
                best_score = current_score
                stagnant_generations = 0

            if progress is not None:
                progress("evolution", generation + 1, self._generations)
            if on_generation is not None:
                on_generation(generation + 1, best_arrangement)

            if stagnant_generations >= self.TERMINATION_THRESHOLD:
                return SolveResult(
                    best_arrangement,
                    generation + 1,
                    current_score,
                    True,
                )

        assert best_arrangement is not None and best_score is not None
        return SolveResult(
            best_arrangement,
            self._generations,
            best_score,
            False,
        )

    def _score(self, arrangement: Arrangement) -> float:
        return arrangement.score(self._analysis.cost)

    def _elite_individuals(self) -> Sequence[Arrangement]:
        return sorted(self._population, key=self._score)[-self._elite_size :]

    def _best_arrangement(self) -> Arrangement:
        return max(self._population, key=self._score)
