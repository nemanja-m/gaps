from __future__ import annotations

import bisect
from collections.abc import Sequence

import cv2 as cv
import numpy as np

from gaps.domain import Image


class SizeDetector:
    """Detect the square piece size from a puzzle image."""

    RECTANGLE_TOLERANCE = 3
    EXTENT_RATIO = 0.75
    MIN_SIZE = 32
    MAX_SIZE = 128
    MIN_SIZE_COEFFICIENT = 0.9
    MAX_SIZE_COEFFICIENT = 1.3

    def __init__(self, image: Image) -> None:
        if image.ndim != 3:
            raise ValueError("image must be a color image with three dimensions")
        self._image = image.copy()
        self._possible_sizes = self._calculate_possible_sizes()

    def detect(self) -> int:
        """Return the most likely piece size in pixels."""
        if not self._possible_sizes:
            raise ValueError("image dimensions do not support a valid piece size")
        if len(self._possible_sizes) == 1:
            return self._possible_sizes[0]

        candidates = [
            candidate
            for channel_image in self._split_channel_images()
            for candidate in self._find_size_candidates(channel_image)
        ]
        if not candidates:
            raise ValueError("could not detect a piece size from the image")

        probabilities = {size: 0 for size in self._possible_sizes}
        for candidate in candidates:
            probabilities[self._find_nearest_size(candidate)] += 1
        return max(probabilities, key=lambda size: probabilities[size])

    def _split_channel_images(self) -> list[np.ndarray]:
        blue, green, red = cv.split(self._image)
        return [
            red,
            green,
            blue,
            cv.add(red, green),
            cv.add(red, blue),
            cv.add(green, blue),
        ]

    def _find_size_candidates(self, image: np.ndarray) -> list[float]:
        binary_image = self._filter_image(image)
        contours, _ = cv.findContours(
            binary_image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE
        )

        candidates = []
        for contour in contours:
            bounding_rect = cv.boundingRect(contour)
            contour_area = cv.contourArea(contour)
            if self._is_valid_contour(contour_area, bounding_rect):
                candidates.append((bounding_rect[2] + bounding_rect[3]) / 2)
        return candidates

    def _is_valid_contour(
        self, contour_area: float, bounding_rect: Sequence[int]
    ) -> bool:
        _, _, width, height = bounding_rect
        extent = contour_area / (width * height)
        lower_limit = self.MIN_SIZE_COEFFICIENT * self._possible_sizes[0]
        upper_limit = self.MAX_SIZE_COEFFICIENT * self._possible_sizes[-1]

        return (
            width > lower_limit
            and height > lower_limit
            and width < upper_limit
            and height < upper_limit
            and abs(width - height) < self.RECTANGLE_TOLERANCE
            and extent >= self.EXTENT_RATIO
        )

    def _find_nearest_size(self, candidate: float) -> int:
        index = bisect.bisect_right(self._possible_sizes, candidate)
        if index == 0:
            return self._possible_sizes[0]
        if index >= len(self._possible_sizes):
            return self._possible_sizes[-1]

        right_size = self._possible_sizes[index]
        left_size = self._possible_sizes[index - 1]
        return (
            right_size
            if abs(candidate - right_size) < abs(candidate - left_size)
            else left_size
        )

    def _calculate_possible_sizes(self) -> list[int]:
        rows, columns = self._image.shape[:2]
        return [
            size
            for size in range(self.MIN_SIZE, self.MAX_SIZE + 1)
            if rows % size == 0 and columns % size == 0
        ]

    @staticmethod
    def _filter_image(image: np.ndarray) -> np.ndarray:
        _, thresholded = cv.threshold(image, 200, 255, cv.THRESH_BINARY)
        kernel = np.ones((5, 5), dtype=np.uint8)
        opened = cv.morphologyEx(thresholded, cv.MORPH_OPEN, kernel, iterations=3)
        return cv.bitwise_not(opened)
