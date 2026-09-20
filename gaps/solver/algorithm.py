from __future__ import annotations

import itertools
import random
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from gaps.domain import (
    Arrangement,
    Direction,
    EdgeAxis,
    Image,
    Piece,
    PuzzleLayout,
)
from gaps.imaging.transforms import flatten_image
from gaps.solver.analysis import EdgeCostTable
from gaps.solver.crossover import Crossover, CrossoverStats
from gaps.solver.selection import tournament_selection


@dataclass(frozen=True, slots=True)
class _BeamState:
    identifiers: tuple[int, ...]
    used: frozenset[int]
    priority: float


def _seed_position_priority(
    identifiers: tuple[int, ...],
    candidate_id: int,
    layout: PuzzleLayout,
    analysis: EdgeCostTable,
    confidence_weight: float,
    border_weight: float,
) -> float:
    row, column = divmod(len(identifiers), layout.columns)
    priority = 0.0
    confidence = 0.0

    if column > 0:
        left_id = identifiers[-1]
        priority += analysis.cost((left_id, candidate_id), EdgeAxis.HORIZONTAL)
        confidence += analysis.confidence(left_id, Direction.RIGHT)
    if row > 0:
        above_id = identifiers[-layout.columns]
        priority += analysis.cost((above_id, candidate_id), EdgeAxis.VERTICAL)
        confidence += analysis.confidence(above_id, Direction.BOTTOM)

    if border_weight:
        if row == 0:
            priority -= border_weight * analysis.best_cost(candidate_id, Direction.TOP)
        if row == layout.rows - 1:
            priority -= border_weight * analysis.best_cost(
                candidate_id, Direction.BOTTOM
            )
        if column == 0:
            priority -= border_weight * analysis.best_cost(candidate_id, Direction.LEFT)
        if column == layout.columns - 1:
            priority -= border_weight * analysis.best_cost(
                candidate_id, Direction.RIGHT
            )

    return priority - confidence_weight * confidence


def build_seed_arrangements(
    pieces: Sequence[Piece],
    layout: PuzzleLayout,
    analysis: EdgeCostTable,
    rng: random.Random,
    count: int,
    beam_width: int = 4,
    candidate_width: int = 8,
    confidence_weight: float = 0.02,
    border_weight: float = 0.0,
) -> list[Arrangement]:
    """Build valid arrangements using confidence-aware row-major beam search."""
    if count <= 0 or not pieces:
        return []
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    if candidate_width <= 0:
        raise ValueError("candidate_width must be positive")
    if confidence_weight < 0.0:
        raise ValueError("confidence_weight must not be negative")
    if border_weight < 0.0:
        raise ValueError("border_weight must not be negative")

    pieces_by_id = {piece.identifier: piece for piece in pieces}
    piece_ids = tuple(pieces_by_id)
    seed_count = min(count, len(piece_ids))
    anchors = rng.sample(piece_ids, seed_count)
    states = [_BeamState((anchor,), frozenset((anchor,)), 0.0) for anchor in anchors]
    effective_beam_width = max(beam_width, seed_count)

    for _ in range(1, layout.piece_count):
        expanded: list[_BeamState] = []
        for state in states:
            candidates = [
                piece_id for piece_id in piece_ids if piece_id not in state.used
            ]
            candidates.sort(
                key=lambda candidate_id: (
                    _seed_position_priority(
                        state.identifiers,
                        candidate_id,
                        layout,
                        analysis,
                        confidence_weight,
                        border_weight,
                    ),
                    candidate_id,
                )
            )
            for candidate_id in candidates[:candidate_width]:
                priority = _seed_position_priority(
                    state.identifiers,
                    candidate_id,
                    layout,
                    analysis,
                    confidence_weight,
                    border_weight,
                )
                expanded.append(
                    _BeamState(
                        state.identifiers + (candidate_id,),
                        state.used | {candidate_id},
                        state.priority + priority,
                    )
                )

        states = sorted(
            expanded,
            key=lambda state: (state.priority, state.identifiers),
        )[:effective_beam_width]
        if not states:
            break

    seeds: list[Arrangement] = []
    seen: set[tuple[int, ...]] = set()
    for state in sorted(states, key=lambda item: (item.priority, item.identifiers)):
        if len(state.identifiers) != layout.piece_count:
            continue
        if state.identifiers in seen:
            continue
        seen.add(state.identifiers)
        seeds.append(
            Arrangement(
                [pieces_by_id[piece_id] for piece_id in state.identifiers],
                layout,
            )
        )
        if len(seeds) == count:
            break

    return seeds


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
    repair_steps: int


_WORKER_CONTEXT: _WorkerContext | None = None


def initialize_worker(
    pieces: tuple[Piece, ...],
    layout: PuzzleLayout,
    analysis: EdgeCostTable,
    mutation_rate: float,
    local_search_steps: int = 1,
    local_search_candidates: int = 4,
    repair_steps: int = 1,
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
        repair_steps=repair_steps,
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


def _repair_window_indices(
    arrangement: Arrangement,
    rng: random.Random,
) -> tuple[int, ...] | None:
    rows = arrangement.layout.rows
    columns = arrangement.layout.columns
    if rows >= 2 and columns >= 2:
        row = rng.randrange(rows - 1)
        column = rng.randrange(columns - 1)
        return (
            row * columns + column,
            row * columns + column + 1,
            (row + 1) * columns + column,
            (row + 1) * columns + column + 1,
        )
    if columns >= 3:
        start = rng.randrange(columns - 2)
        return tuple(start + offset for offset in range(3))
    if rows >= 3:
        start = rng.randrange(rows - 2)
        return tuple((start + offset) * columns for offset in range(3))
    return None


def _constraint_repair(
    arrangement: Arrangement,
    analysis: EdgeCostTable,
    rng: random.Random,
    steps: int,
) -> None:
    """Exhaustively repair small windows while preserving all piece assignments."""
    if steps <= 0:
        return

    for _ in range(steps):
        indices = _repair_window_indices(arrangement, rng)
        if indices is None:
            return
        original = tuple(arrangement.pieces[index] for index in indices)
        original_ids = tuple(piece.identifier for piece in original)
        current_score = arrangement.score(analysis)
        best_score = current_score
        best_order: tuple[Piece, ...] | None = None
        for candidate in itertools.permutations(original):
            candidate_ids = tuple(piece.identifier for piece in candidate)
            if candidate_ids == original_ids:
                continue
            arrangement.replace_positions(indices, candidate, cost_lookup=analysis)
            candidate_score = arrangement.score(analysis)
            arrangement.replace_positions(indices, original, cost_lookup=analysis)
            if candidate_score > best_score:
                best_score = candidate_score
                best_order = candidate
        if best_order is None:
            return
        arrangement.replace_positions(indices, best_order, cost_lookup=analysis)


def _mutate_child(
    arrangement: Arrangement,
    analysis: EdgeCostTable,
    rng: random.Random,
    mutation_rate: float,
    local_search_steps: int,
    local_search_candidates: int,
    repair_steps: int,
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
    _constraint_repair(arrangement, analysis, rng, repair_steps)


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
        context.repair_steps,
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
        repair_steps: int = 1,
        max_restarts: int = 2,
        restart_threshold: int | None = None,
        tournament_size: int = 3,
        seed_fraction: float = 0.25,
        seed_beam_width: int = 4,
        seed_candidate_width: int = 8,
        seed_confidence_weight: float = 0.02,
        seed_border_weight: float = 0.0,
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
        if repair_steps < 0:
            raise ValueError("repair_steps must not be negative")
        if max_restarts < 0:
            raise ValueError("max_restarts must not be negative")
        if restart_threshold is not None and restart_threshold <= 0:
            raise ValueError("restart_threshold must be positive")
        if tournament_size <= 0:
            raise ValueError("tournament_size must be positive")
        if not 0.0 <= seed_fraction <= 1.0:
            raise ValueError("seed_fraction must be between zero and one")
        if seed_beam_width <= 0:
            raise ValueError("seed_beam_width must be positive")
        if seed_candidate_width <= 0:
            raise ValueError("seed_candidate_width must be positive")
        if seed_confidence_weight < 0.0:
            raise ValueError("seed_confidence_weight must not be negative")
        if seed_border_weight < 0.0:
            raise ValueError("seed_border_weight must not be negative")
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
        self._repair_steps = repair_steps
        self._max_restarts = max_restarts
        self._restart_threshold = restart_threshold or max(
            1, self.TERMINATION_THRESHOLD // 2
        )
        self._tournament_size = tournament_size
        self._seed_fraction = seed_fraction
        self._seed_beam_width = seed_beam_width
        self._seed_candidate_width = seed_candidate_width
        self._seed_confidence_weight = seed_confidence_weight
        self._seed_border_weight = seed_border_weight
        self._population_initialized = False
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
        self._initialize_population()
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

    def _initialize_population(self) -> None:
        if self._population_initialized:
            return
        self._population_initialized = True
        if self._seed_fraction == 0.0:
            return

        try:
            seed_count = min(
                8,
                max(1, int(self._population_size * self._seed_fraction)),
                max(1, 1024 // self._layout.piece_count),
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("could not calculate seed count") from error

        try:
            seeds = build_seed_arrangements(
                self._pieces,
                self._layout,
                self._analysis,
                self._rng,
                count=seed_count,
                beam_width=self._seed_beam_width,
                candidate_width=self._seed_candidate_width,
                confidence_weight=self._seed_confidence_weight,
                border_weight=self._seed_border_weight,
            )
        except (KeyError, ValueError):
            return
        self._population = (seeds + self._population)[: self._population_size]

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
                self._repair_steps,
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
            self._repair_steps,
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
