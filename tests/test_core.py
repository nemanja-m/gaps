import numpy as np

from gaps.core import Chromosome, Gene
from gaps.utils import flatten_image


class TestGene:
    def test_size_returns_puzzle_size_in_px(self) -> None:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        gene = Gene(index=1, image=image)
        assert gene.size() == 32


class TestChromosome:
    def test_to_image_stitches_puzzle_from_pieces(self) -> None:
        image = np.random.randint(
            low=0,
            high=255,
            size=(32, 32, 3),
            dtype=np.uint8,
        )
        pieces, rows, cols = flatten_image(image, piece_size=8)
        chromosome = Chromosome(
            genes=[Gene(index, piece) for index, piece in enumerate(pieces)],
            rows=rows,
            columns=cols,
        )
        np.testing.assert_array_equal(image, chromosome.to_image())
