import bisect
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from . import utils
from .callback import Callback
from .crossover import Crossover
from .evaluation import Evaluator
from .exception import FitnessException


@dataclass(frozen=True)
class Gene:
    """Represents a single jigsaw puzzle piece.

    Multiple genes form a chromosome, i.e. a jigsaw puzzle. Each gene has an
    identifier so it can be tracked across different chromosomes.

    Attributes:
        image: ndarray representing piece's BGR values.
        index: Index withing jigsaw puzzle.
    """

    index: int
    image: NDArray[np.uint8]

    def __getitem__(self, index: int):
        return self.image.__getitem__(index)

    def size(self) -> int:
        """Return puzzle piece size in pixels."""
        return self.image.shape[0]


class Chromosome:
    """Represents one possible solution to the jigsaw puzzle.

    Chromosome object is one of the solutions to the problem (possible
    arrangement of the puzzle's pieces). Initially, it is created by randomly
    shuffling puzzle pieces (genes). Over generations, chromosomes will be
    closer to the puzzle solution due to crossover, mutations, and fitness
    evaluation.

    Args:
        pieces: Array of pieces representing initial puzzle.
        rows: Number of rows in input puzzle
        columns: Number of columns in input puzzle

    Examples:
        >>> from gaps.core import Chromosome
        >>> from gaps.utils import flatten_image
        >>> pieces, rows, columns = flatten_image(...)
        >>> chromosome = Chromosome(pieces, rows, columns)
    """

    def __init__(
        self,
        genes: List[Gene],
        rows: int,
        columns: int,
        shuffle: bool = False,
    ) -> None:
        self._genes = random.sample(genes, len(genes)) if shuffle else genes
        self._rows = rows
        self._columns = columns
        self._fitness: Optional[float] = None

    @property
    def fitness(self) -> float:
        if self._fitness is None:
            raise FitnessException("Trying to return empty fitness value.")
        return self._fitness

    @fitness.setter
    def fitness(self, value: float) -> None:
        self._fitness = value

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def columns(self) -> int:
        return self._columns

    def to_image(self) -> NDArray[np.uint8]:
        """Convert list of puzzle pieces to full puzzle image."""
        pieces = [gene.image for gene in self._genes]
        return utils.stitch_image(pieces, rows=self._rows, columns=self._columns)

    def __getitem__(self, key: int) -> List[Gene]:
        return self._genes[key * self.columns : (key + 1) * self.columns]


class Population:
    def __init__(self, chromosomes: List[Chromosome]) -> None:
        self._chromosomes = chromosomes

    def set_fitness(self, fitness_scores: List[float]) -> None:
        """Set fitness score for each chromosome.

        Positions in the input list match positions in the chromosome list.

        Args:
            fitness_scores: A list of fitness score for each chromosome.

        Raises:
            FitnessException: when size of fitness scores list
                does not match population size.
        """
        if len(fitness_scores) != len(self._chromosomes):
            raise FitnessException("Fitness scores size must match population size")

        for score, chromosome in zip(fitness_scores, self._chromosomes):
            chromosome.fitness = score

    def elites(self, elite_size: int) -> List[Chromosome]:
        """Return the best chromosomes based on the fitness values.

        Elite chromosomes are transferred directly in the next generation.

        Args:
            elite_size: Size of the elite population.

        Returns:
            A list of elite chromosomes.
        """
        return sorted(
            self._chromosomes,
            key=lambda c: c.fitness,
            reverse=True,
        )[:elite_size]

    def select_parent_pairs(
        self, elite_size: int
    ) -> List[Tuple[Chromosome, Chromosome]]:
        """Select parent pairs for crossover.

        Parent selection considers the number of elites and returns
        `population_size - elite_size` pairs.

        Args:
            elite_size: The number of elite chromosomes.

        Returns:
            A list of parent tuples.
        """
        fitness_values = [chromosome.fitness for chromosome in self._chromosomes]
        intervals = [sum(fitness_values[: i + 1]) for i in range(len(fitness_values))]
        pairs = [
            (self._select_parent(intervals), self._select_parent(intervals))
            for _ in range(len(self._chromosomes) - elite_size)
        ]
        return pairs

    def _select_parent(self, probability_intervals: List[float]) -> Chromosome:
        random_select = random.uniform(0, probability_intervals[-1])
        selected_index = bisect.bisect_left(probability_intervals, random_select)
        return self._chromosomes[selected_index]


class Evolution:
    DEFAULT_TERMINATION_THRESHOLD = 10

    def __init__(
        self,
        initial_population: Population,
        fitness_evaluator: Evaluator,
        callbacks: Optional[List[Callback]] = None,
    ) -> None:
        self._initial_population = initial_population
        self._fitness_evaluator = fitness_evaluator
        self._callbacks = callbacks or []

    def run(
        self,
        num_generations: int,
        elite_size: int,
        termination_threshold: int = DEFAULT_TERMINATION_THRESHOLD,
    ) -> Optional[Chromosome]:
        """Evolve initial population and return the best chromosome.

        Args:
            num_generations: How many iterations does evolution run.
            elite_size: Number of elite chromosomes passed to the next generation.
        """
        fittest = None
        population = self._initial_population

        termination_counter = 0
        best_fitness_score = float("-inf")

        for generation in range(num_generations):
            self._notify_generation_started(generation)
            new_chromosomes: List[Chromosome] = []

            self._fitness_evaluator.evaluate(population)
            elites = population.elites(elite_size)
            new_chromosomes.extend(elites)
            fittest = elites[0]
            self._notify_evaluation_completed(generation, population, fittest)

            if fittest.fitness <= best_fitness_score:
                termination_counter += 1
            else:
                best_fitness_score = fittest.fitness

            if termination_counter == termination_threshold:
                self._notify_early_stopping(generation, fittest)
                return fittest

            parents = population.select_parent_pairs(elite_size)
            for first_parent, second_parent in parents:
                crossover = Crossover(first_parent, second_parent)
                crossover.run()
                child = crossover.child()
                new_chromosomes.append(child)

            population = Population(new_chromosomes)
            self._notify_generation_completed(generation)

        self._notify_solution_found(generation, fittest)
        return fittest

    def _notify_generation_started(self, generation: int) -> None:
        for callback in self._callbacks:
            callback.generation_started(generation)

    def _notify_generation_completed(self, generation: int) -> None:
        for callback in self._callbacks:
            callback.generation_completed(generation)

    def _notify_evaluation_completed(
        self,
        generation: int,
        population: Population,
        fittest: Chromosome,
    ) -> None:
        for callback in self._callbacks:
            callback.evaluation_completed(generation, population, fittest)

    def _notify_early_stopping(self, generation: int, fittest: Chromosome) -> None:
        for callback in self._callbacks:
            callback.early_stopping(generation, fittest)

    def _notify_solution_found(self, generation: int, fittest: Chromosome) -> None:
        for callback in self._callbacks:
            callback.solution_found(generation, fittest)
