from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Piece:
    """A square image fragment with a stable identifier."""

    image: Any
    id: int

    def __post_init__(self) -> None:
        self.image = self.image.copy()

    def __getitem__(self, index: Any) -> Any:
        return self.image[index]

    @property
    def size(self) -> int:
        """Return the width and height of the square piece."""
        return self.image.shape[0]

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the piece image."""
        return self.image.shape
