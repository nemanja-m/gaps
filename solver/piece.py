
class Piece(object):
    """Represents single jigsaw puzzle piece.

    Each piece has identifier so it can be
    tracked accross different individuals

    :param value: ndarray representing piece's RGB values
    :param index: Unique id withing piece's parent image

    Usage::

        >>> from models import Piece
        >>> piece = Piece(image[:28, :28, :], 42)

    """

    def __init__(self, image, index):
        self.image = image[:]
        self.id = index

    def __getitem__(self, index):
        return self.image.__getitem__(index)

    def size(self):
        """Returns piece size"""
        return self.image.shape[0]

    def shape(self):
        """Returns shape of piece's image"""
        return self.image.shape
