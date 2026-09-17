from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from gaps.domain import Arrangement, Image, Piece, PuzzleLayout
from gaps.imaging.transforms import flatten_image
from gaps.solver.analysis import EdgeCostTable
from gaps.solver.crossover import Crossover, CrossoverStats
from gaps.solver.selection import roulette_selection

ProgressCallback = Callable[[str, int, int], None]
GenerationCallback = Callable[[int, Arrangement], None]
type ChildTask = tuple[tuple[int, ...], tuple[int, ...], int]


@dataclass(slots=True)
class _WorkerContext:
    pieces_by_id: dict[int, Piece]
    layout: PuzzleLayout
    analysis: EdgeCostTable
    mutation_rate: float


_WORKER_CONTEXT: _WorkerContext | None = None


def initialize_worker(
    pieces: tuple[Piece, ...],
    layout: PuzzleLayout,
    analysis: EdgeCostTable,
    mutation_rate: float,
) -> None:
    """Initialize read-only puzzle state once in a process-pool worker."""
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = _WorkerContext(
        pieces_by_id={piece.identifier: piece for piece in pieces},
        layout=layout,
        analysis=analysis,
        mutation_rate=mutation_rate,
    )


def _require_worker_context() -> _WorkerContext:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("child worker was not initialized")
    return _WORKER_CONTEXT


def _arrangement_from_ids(
    identifiers: tuple[int, ...], context: _WorkerContext
) -> Arrangement:
    try:
        pieces = [context.pieces_by_id[identifier] for identifier in identifiers]
    except KeyError as error:
        raise ValueError("child arrangement contains an unknown piece") from error
    return Arrangement(pieces, context.layout)


def _mutate_child(
    arrangement: Arrangement,
    analysis: EdgeCostTable,
    rng: random.Random,
    mutation_rate: float,
) -> None:
    if mutation_rate == 0.0 or rng.random() >= mutation_rate:
        return
    if len(arrangement.pieces) < 2:
        return

    first_index, second_index = rng.sample(range(len(arrangement.pieces)), 2)
    original_score = arrangement.score(analysis)
    arrangement.swap(first_index, second_index, cost_lookup=analysis)
    if arrangement.score(analysis) < original_score:
        arrangement.swap(first_index, second_index, cost_lookup=analysis)


def build_child(task: ChildTask) -> tuple[int, ...]:
    """Build one child from compact parent permutations in a worker."""
    context = _require_worker_context()
    first_ids, second_ids, seed = task
    rng = random.Random(seed)
    first_parent = _arrangement_from_ids(first_ids, context)
    second_parent = _arrangement_from_ids(second_ids, context)
    crossover = Crossover(first_parent, second_parent, context.analysis, rng)
    crossover.run()
    child = crossover.child()
    _mutate_child(child, context.analysis, rng, context.mutation_rate)
    return tuple(piece.identifier for piece in child.pieces)


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
        mutation_rate: float = 0.05,
        collect_stats: bool = False,
        workers: int = 1,
    ) -> None:
        if population_size <= 0:
            raise ValueError("population_size must be positive")
        if generations <= 0:
            raise ValueError("generations must be positive")
        if not 0 < elite_size < population_size:
            raise ValueError("elite_size must be between zero and population size")
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be between zero and one")
        if workers <= 0:
            raise ValueError("workers must be positive")
        if collect_stats and workers > 1:
            raise ValueError("collect_stats is only supported with one worker")

        pieces, layout = flatten_image(image, piece_size, indexed=True)
        self._analysis = EdgeCostTable()
        self._pieces = pieces
        self._pieces_by_id = {piece.identifier: piece for piece in pieces}
        self._layout = layout
        self._generations = generations
        self._elite_size = elite_size
        self._rng = rng or random.Random()
        self._mutation_rate = mutation_rate
        self._workers = workers
        self._crossover_stats = CrossoverStats() if collect_stats else None
        self._population = [
            Arrangement.random(pieces, layout, self._rng)
            for _ in range(population_size)
        ]

    @property
    def crossover_stats(self) -> CrossoverStats | None:
        """Return optional crossover counters collected during the last solve."""
        return self._crossover_stats

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
        executor = self._create_executor()

        try:
            for generation in range(self._generations):
                next_population = (
                    self._next_population_parallel(executor)
                    if executor is not None
                    else self._next_population_serial()
                )
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
        finally:
            if executor is not None:
                executor.shutdown()

        assert best_arrangement is not None and best_score is not None
        return SolveResult(
            best_arrangement,
            self._generations,
            best_score,
            False,
        )

    def _create_executor(self) -> ProcessPoolExecutor | None:
        if self._workers == 1:
            return None
        return ProcessPoolExecutor(
            max_workers=self._workers,
            initializer=initialize_worker,
            initargs=(
                tuple(self._pieces),
                self._layout,
                self._analysis,
                self._mutation_rate,
            ),
        )

    def _next_population_serial(self) -> list[Arrangement]:
        next_population = list(self._elite_individuals())
        for first_parent, second_parent in roulette_selection(
            self._population,
            score=self._score,
            rng=self._rng,
            elites=self._elite_size,
        ):
            crossover = Crossover(
                first_parent,
                second_parent,
                self._analysis,
                self._rng,
                stats=self._crossover_stats,
            )
            crossover.run()
            child = crossover.child()
            self._mutate(child)
            next_population.append(child)
        return next_population

    def _next_population_parallel(
        self, executor: ProcessPoolExecutor
    ) -> list[Arrangement]:
        next_population = list(self._elite_individuals())
        tasks: list[ChildTask] = []
        for first_parent, second_parent in roulette_selection(
            self._population,
            score=self._score,
            rng=self._rng,
            elites=self._elite_size,
        ):
            tasks.append(
                (
                    self._identifiers(first_parent),
                    self._identifiers(second_parent),
                    self._rng.getrandbits(64),
                )
            )

        chunksize = max(1, len(tasks) // (self._workers * 4))
        for child_identifiers in executor.map(build_child, tasks, chunksize=chunksize):
            next_population.append(
                self._arrangement_from_identifiers(child_identifiers)
            )
        return next_population

    @staticmethod
    def _identifiers(arrangement: Arrangement) -> tuple[int, ...]:
        return tuple(piece.identifier for piece in arrangement.pieces)

    def _arrangement_from_identifiers(
        self, identifiers: tuple[int, ...]
    ) -> Arrangement:
        try:
            pieces = [self._pieces_by_id[identifier] for identifier in identifiers]
        except KeyError as error:
            raise ValueError("child arrangement contains an unknown piece") from error
        return Arrangement(pieces, self._layout)

    def _mutate(self, arrangement: Arrangement) -> None:
        """Try one improving random swap to preserve population diversity."""
        if self._mutation_rate == 0.0 or self._rng.random() >= self._mutation_rate:
            return
        if len(arrangement.pieces) < 2:
            return

        first_index, second_index = self._rng.sample(range(len(arrangement.pieces)), 2)
        original_score = self._score(arrangement)
        arrangement.swap(first_index, second_index, cost_lookup=self._analysis)
        if self._score(arrangement) < original_score:
            arrangement.swap(
                first_index,
                second_index,
                cost_lookup=self._analysis,
            )

    def _score(self, arrangement: Arrangement) -> float:
        return arrangement.score(self._analysis)

    def _elite_individuals(self) -> Sequence[Arrangement]:
        return sorted(self._population, key=self._score)[-self._elite_size :]

    def _best_arrangement(self) -> Arrangement:
        return max(self._population, key=self._score)
