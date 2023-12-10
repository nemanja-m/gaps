from abc import ABC

from gaps.core import Chromosome, Population


class Callback(ABC):
    def generation_started(self, generation: int) -> None:
        pass

    def generation_completed(self, generation: int) -> None:
        pass

    def evaluation_completed(
        self,
        generation: int,
        population: Population,
        fittest: Chromosome,
    ) -> None:
        pass

    def early_stopping(self, generation: int, fittest: Chromosome) -> None:
        pass

    def solution_found(self, generation: int, fittest: Chromosome) -> None:
        pass
