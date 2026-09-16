from __future__ import annotations

from pathlib import Path

import cv2 as cv
import numpy as np

from gaps.imaging.transforms import assemble_image, flatten_image

IMAGE_SIZE = 256
PIECE_SIZE = 64
SHUFFLE_SEED = 23
OUTPUT_DIRECTORY = Path("images/grayscale")


def create_source_image() -> np.ndarray:
    """Create a deterministic, feature-rich grayscale image."""
    y, x = np.indices((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    image = 35.0 + 0.28 * x + 0.22 * y
    image += 28.0 * np.sin(x / 17.0) * np.cos(y / 23.0)
    image = np.clip(image, 0, 255).astype(np.uint8)

    cv.circle(image, (64, 68), 38, (235,), thickness=-1)
    cv.circle(image, (64, 68), 24, (35,), thickness=5)
    cv.rectangle(image, (134, 24), (224, 92), (28,), thickness=-1)
    cv.rectangle(image, (148, 38), (210, 78), (218,), thickness=4)
    cv.line(image, (18, 218), (230, 118), (245,), thickness=6)
    cv.putText(
        image,
        "GRAY",
        (24, 158),
        cv.FONT_HERSHEY_SIMPLEX,
        1.2,
        (20,),
        thickness=3,
        lineType=cv.LINE_AA,
    )

    noise = np.random.default_rng(7).normal(0.0, 3.0, image.shape)
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Make the intended tile seams exact while leaving the image visually
    # unchanged. This gives the solver an unambiguous manual test case.
    for boundary in range(PIECE_SIZE, IMAGE_SIZE, PIECE_SIZE):
        image[:, boundary] = image[:, boundary - 1]
        image[boundary, :] = image[boundary - 1, :]

    return image


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    source = create_source_image()
    pieces, layout = flatten_image(source, PIECE_SIZE)
    np.random.default_rng(SHUFFLE_SEED).shuffle(pieces)
    puzzle = assemble_image(pieces, layout)

    outputs = {
        "grayscale_source.png": source,
        "grayscale_puzzle.png": puzzle,
        "grayscale_expected.png": source,
    }
    for filename, image in outputs.items():
        if not cv.imwrite(str(OUTPUT_DIRECTORY / filename), image):
            raise RuntimeError(f"could not write {filename}")

    print(f"Wrote {len(outputs)} images to {OUTPUT_DIRECTORY}")
    print(f"Piece size: {PIECE_SIZE}; shuffle seed: {SHUFFLE_SEED}")


if __name__ == "__main__":
    main()
