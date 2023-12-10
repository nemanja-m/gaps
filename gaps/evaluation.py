from abc import ABC, abstractmethod
from typing import Callable, List

from .core import Chromosome, Population


class Evaluator(ABC):
    """Base class for fitness evaluator.

    Specializations can be either serial or parallel.

    """

    def __init__(self, fitness_fn: Callable[[Chromosome], float]) -> None:
        self._fitness_fn = fitness_fn

    def evaluate(self, population: Population) -> None:
        scores = self._evaluate_fitness_scores(population._chromosomes)
        population.set_fitness(scores)

    @abstractmethod
    def _evaluate_fitness_scores(
        self,
        chromosomes: List[Chromosome],
        fitness_fn: Callable[[Chromosome], float],
    ) -> List[float]:
        """Return fitness scores for chromosomes."""


class SerialEvaluator(Evaluator):
    """Serial implementation of fitness evaluator."""

    def _evaluate_fitness_scores(self, chromosomes: List[Chromosome]) -> List[float]:
        scores = [self._fitness_fn(chromosome) for chromosome in chromosomes]
        return scores
