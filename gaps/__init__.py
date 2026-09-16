"""Typed jigsaw puzzle generation and solving primitives."""

from gaps.domain import Arrangement, Direction, EdgeAxis, Piece, PuzzleLayout
from gaps.solver.algorithm import GeneticAlgorithm, SolveResult

__all__ = [
    "Arrangement",
    "Direction",
    "EdgeAxis",
    "GeneticAlgorithm",
    "Piece",
    "PuzzleLayout",
    "SolveResult",
]
