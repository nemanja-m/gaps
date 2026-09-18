from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from gaps.domain import Arrangement, Image, Piece, PuzzleLayout
from gaps.imaging.transforms import flatten_image
from gaps.solver.analysis import EdgeCostTable
from gaps.solver.crossover import Crossover, CrossoverStats
from gaps.solver.selection import tournament_selection

ProgressCallback = Callable[[str, int, int], None]
GenerationCallback = Callable[[int, Arrangement], None]
type Move = tuple[str, int, int, int]
type ChildTask = tuple[tuple[int, ...], tuple[int, ...], int, float]


@dataclass(slots=True)
class _WorkerContext:
    pieces_by_id: dict[int, Piece]
    layout: PuzzleLayout
    analysis: EdgeCostTable
    mutation_rate: float
    local_search_steps: int
    local_search_candidates: int


_WORKER_CONTEXT: _WorkerContext | None = None


def initialize_worker(
    pieces: tuple[Piece, ...],
    layout: PuzzleLayout,
    analysis: EdgeCostTable,
    mutation_rate: float,
    local_search_steps: int = 1,
    local_search_candidates: int = 4,
) -> None:
    """Initialize read-only puzzle state once in a process-pool worker."""
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = _WorkerContext(
        pieces_by_id={piece.identifier: piece for piece in pieces},
        layout=layout,
        analysis=analysis,
        mutation_rate=mutation_rate,
        local_search_steps=local_search_steps,
        local_search_candidates=local_search_candidates,
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


def _random_move(piece_count: int, rng: random.Random) -> Move:
    """Generate a permutation-preserving local-search move."""
    if piece_count < 2:
        raise ValueError("at least two pieces are required for a move")

    move_kind = rng.randrange(3) if piece_count >= 4 else rng.randrange(2)
    if move_kind == 0:
        first, second = rng.sample(range(piece_count), 2)
        return ("swap", first, second, 1)
    if move_kind == 1:
        source, target = rng.sample(range(piece_count), 2)
        return ("relocate", source, target, 1)

    block_size = rng.randint(1, min(4, piece_count // 2))
    first = rng.randrange(piece_count - 2 * block_size + 1)
    second = rng.randrange(first + block_size, piece_count - block_size + 1)
    return ("block", first, second, block_size)


def _apply_move(
    arrangement: Arrangement,
    move: Move,
    analysis: EdgeCostTable,
) -> None:
    operation, first, second, block_size = move
    if operation == "swap":
        arrangement.swap(first, second, cost_lookup=analysis)
    elif operation == "relocate":
        arrangement.relocate(first, second, cost_lookup=analysis)
    elif operation == "block":
        arrangement.swap_blocks(first, second, block_size, cost_lookup=analysis)
    else:
        raise ValueError(f"unknown local-search move: {operation}")


def _inverse_move(move: Move) -> Move:
    operation, first, second, block_size = move
    if operation == "relocate":
        return (operation, second, first, block_size)
    return move


def _local_search(
    arrangement: Arrangement,
    analysis: EdgeCostTable,
    rng: random.Random,
    steps: int,
    candidates: int,
) -> None:
    """Apply best-improvement local search using incremental seam costs."""
    if steps <= 0 or candidates <= 0 or len(arrangement.pieces) < 2:
        return

    for _ in range(steps):
        current_score = arrangement.score(analysis)
        best_score = current_score
        best_move: Move | None = None
        for _ in range(candidates):
            move = _random_move(len(arrangement.pieces), rng)
            _apply_move(arrangement, move, analysis)
            candidate_score = arrangement.score(analysis)
            _apply_move(arrangement, _inverse_move(move), analysis)
            if candidate_score > best_score:
                best_score = candidate_score
                best_move = move

        if best_move is None:
            return
        _apply_move(arrangement, best_move, analysis)


def _mutate_child(
    arrangement: Arrangement,
    analysis: EdgeCostTable,
    rng: random.Random,
    mutation_rate: float,
    local_search_steps: int,
    local_search_candidates: int,
) -> None:
    if mutation_rate > 0.0 and rng.random() < mutation_rate:
        move = _random_move(len(arrangement.pieces), rng)
        original_score = arrangement.score(analysis)
        _apply_move(arrangement, move, analysis)
        mutated_score = arrangement.score(analysis)
        reject_worse_move = (
            mutated_score < original_score
            and rng.random() >= 0.05 + 0.15 * mutation_rate
        )
        if reject_worse_move:
            _apply_move(arrangement, _inverse_move(move), analysis)

    _local_search(
        arrangement,
        analysis,
        rng,
        local_search_steps,
        local_search_candidates,
    )


def _crossover_child(
    first_parent: Arrangement,
    second_parent: Arrangement,
    analysis: EdgeCostTable,
    rng: random.Random,
    stats: CrossoverStats | None = None,
) -> Arrangement:
    """Build a valid child, falling back to a parent if crossover cannot fill it."""
    crossover = Crossover(
        first_parent,
        second_parent,
        analysis,
        rng,
        stats=stats,
    )
    crossover.run()
    try:
        return crossover.child()
    except RuntimeError:
        return Arrangement(list(first_parent.pieces), first_parent.layout)


def build_child(task: ChildTask) -> tuple[int, ...]:
    """Build one child from compact parent permutations in a worker."""
    context = _require_worker_context()
    first_ids, second_ids, seed, mutation_rate = task
    rng = random.Random(seed)
    first_parent = _arrangement_from_ids(first_ids, context)
    second_parent = _arrangement_from_ids(second_ids, context)
    child = _crossover_child(first_parent, second_parent, context.analysis, rng)
    _mutate_child(
        child,
        context.analysis,
        rng,
        mutation_rate,
        context.local_search_steps,
        context.local_search_candidates,
    )
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
        local_search_steps: int = 1,
        local_search_candidates: int = 4,
        max_restarts: int = 2,
        restart_threshold: int | None = None,
        tournament_size: int = 3,
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
        if local_search_steps < 0:
            raise ValueError("local_search_steps must not be negative")
        if local_search_candidates < 0:
            raise ValueError("local_search_candidates must not be negative")
        if max_restarts < 0:
            raise ValueError("max_restarts must not be negative")
        if restart_threshold is not None and restart_threshold <= 0:
            raise ValueError("restart_threshold must be positive")
        if tournament_size <= 0:
            raise ValueError("tournament_size must be positive")
        if collect_stats and workers > 1:
            raise ValueError("collect_stats is only supported with one worker")

        pieces, layout = flatten_image(image, piece_size, indexed=True)
        self._analysis = EdgeCostTable()
        self._pieces = pieces
        self._pieces_by_id = {piece.identifier: piece for piece in pieces}
        self._layout = layout
        self._generations = generations
        self._population_size = population_size
        self._elite_size = elite_size
        self._rng = rng or random.Random()
        self._mutation_rate = mutation_rate
        self._workers = workers
        self._local_search_steps = local_search_steps
        self._local_search_candidates = local_search_candidates
        self._max_restarts = max_restarts
        self._restart_threshold = restart_threshold or max(
            1, self.TERMINATION_THRESHOLD // 2
        )
        self._tournament_size = tournament_size
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
        """Run evolution and return the best valid arrangement found."""
        self._analysis.analyze(self._pieces, progress=progress)
        best_arrangement = self._best_arrangement()
        best_score = self._score(best_arrangement)
        stagnant_generations = 0
        restarts = 0
        executor = self._create_executor()

        try:
            for generation in range(self._generations):
                mutation_rate = self._adaptive_mutation_rate(stagnant_generations)
                next_population = (
                    self._next_population_parallel(executor, mutation_rate)
                    if executor is not None
                    else self._next_population_serial(mutation_rate)
                )
                self._population = next_population
                current_arrangement = self._best_arrangement()
                current_score = self._score(current_arrangement)
                if current_score > best_score:
                    best_arrangement = current_arrangement
                    best_score = current_score
                    stagnant_generations = 0
                else:
                    stagnant_generations += 1

                if progress is not None:
                    progress("evolution", generation + 1, self._generations)
                if on_generation is not None:
                    on_generation(generation + 1, best_arrangement)

                if stagnant_generations >= self._restart_threshold:
                    if restarts < self._max_restarts:
                        self._restart_population(best_arrangement)
                        restarts += 1
                        stagnant_generations = 0
                    elif stagnant_generations >= self.TERMINATION_THRESHOLD:
                        return SolveResult(
                            best_arrangement,
                            generation + 1,
                            best_score,
                            True,
                        )
        finally:
            if executor is not None:
                executor.shutdown()

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
                self._local_search_steps,
                self._local_search_candidates,
            ),
        )

    def _next_population_serial(self, mutation_rate: float) -> list[Arrangement]:
        next_population = list(self._elite_individuals())
        parent_pairs = tournament_selection(
            self._population,
            score=self._score,
            rng=self._rng,
            elites=self._elite_size,
            tournament_size=self._tournament_size,
        )
        for first_parent, second_parent in parent_pairs:
            child = _crossover_child(
                first_parent,
                second_parent,
                self._analysis,
                self._rng,
                stats=self._crossover_stats,
            )
            self._mutate(child, mutation_rate)
            next_population.append(child)
        return next_population

    def _next_population_parallel(
        self,
        executor: ProcessPoolExecutor,
        mutation_rate: float,
    ) -> list[Arrangement]:
        next_population = list(self._elite_individuals())
        tasks: list[ChildTask] = []
        parent_pairs = tournament_selection(
            self._population,
            score=self._score,
            rng=self._rng,
            elites=self._elite_size,
            tournament_size=self._tournament_size,
        )
        for first_parent, second_parent in parent_pairs:
            tasks.append(
                (
                    self._identifiers(first_parent),
                    self._identifiers(second_parent),
                    self._rng.getrandbits(64),
                    mutation_rate,
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

    def _mutate(self, arrangement: Arrangement, mutation_rate: float) -> None:
        _mutate_child(
            arrangement,
            self._analysis,
            self._rng,
            mutation_rate,
            self._local_search_steps,
            self._local_search_candidates,
        )

    def _adaptive_mutation_rate(self, stagnant_generations: int) -> float:
        """Increase exploration pressure as the population stops improving."""
        if self._mutation_rate == 0.0:
            return 0.0
        stagnation_factor = min(
            2.0,
            stagnant_generations / self._restart_threshold,
        )
        scores = [self._score(individual) for individual in self._population]
        diversity_factor = 1.25 if scores and max(scores) - min(scores) < 1e-6 else 1.0
        return min(
            1.0,
            self._mutation_rate * (1.0 + stagnation_factor) * diversity_factor,
        )

    def _restart_population(self, best_arrangement: Arrangement) -> None:
        """Retain the best known individuals and refill the population randomly."""
        preserved: list[Arrangement] = [best_arrangement]
        preserved_ids = {self._identifiers(best_arrangement)}
        for elite in reversed(self._elite_individuals()):
            elite_ids = self._identifiers(elite)
            if elite_ids not in preserved_ids:
                preserved.append(elite)
                preserved_ids.add(elite_ids)
            if len(preserved) == self._elite_size:
                break

        self._population = preserved
        while len(self._population) < self._population_size:
            self._population.append(
                Arrangement.random(self._pieces, self._layout, self._rng)
            )

    def _score(self, arrangement: Arrangement) -> float:
        return arrangement.score(self._analysis)

    def _elite_individuals(self) -> Sequence[Arrangement]:
        return sorted(self._population, key=self._score)[-self._elite_size :]

    def _best_arrangement(self) -> Arrangement:
        return max(self._population, key=self._score)
