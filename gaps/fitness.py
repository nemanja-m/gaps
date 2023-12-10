from .core import Chromosome
from .image_analysis import ImageAnalysis


def rgb_dissimilarity_fitness(
    chromosome: Chromosome,
    scaling_factor: float = 1000.0,
) -> float:
    score = 1 / scaling_factor
    # For each two adjacent pieces in rows
    for row in range(chromosome.rows):
        for col in range(chromosome.columns - 1):
            ids = (chromosome[row][col].index, chromosome[row][col + 1].index)
            score += ImageAnalysis.get_dissimilarity(ids, orientation="LR")
    # For each two adjacent pieces in columns
    for row in range(chromosome.rows - 1):
        for col in range(chromosome.columns):
            ids = (chromosome[row][col].index, chromosome[row + 1][col].index)
            score += ImageAnalysis.get_dissimilarity(ids, orientation="TD")
    return scaling_factor / score
