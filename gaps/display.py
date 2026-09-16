from __future__ import annotations

from types import TracebackType
from typing import Any, Self

import click
import cv2 as cv

from gaps.domain import Image


class PreviewError(RuntimeError):
    """Raised when an interactive debug preview cannot be displayed."""


MAX_WINDOW_TITLE_LENGTH = 80


def _fit_window_title(title: str, max_length: int = MAX_WINDOW_TITLE_LENGTH) -> str:
    if len(title) <= max_length:
        return title
    if max_length <= 1:
        return title[:max_length]

    prefix_length = (max_length - 1) // 2
    suffix_length = max_length - prefix_length - 1
    return f"{title[:prefix_length]}…{title[-suffix_length:]}"


class OpenCVPreview:
    """Display the latest arrangement in an OpenCV window."""

    def __init__(
        self,
        title: str = "gaps debug",
        total_generations: int | None = None,
    ) -> None:
        self._title = _fit_window_title(title)
        self._total_generations = total_generations
        try:
            cv.namedWindow(self._title, cv.WINDOW_NORMAL)
        except cv.error as error:
            raise PreviewError("debug preview requires a graphical display") from error

    def show(self, image: Image, generation: int | None = None) -> None:
        """Show an image and process window events once."""
        try:
            if generation is not None:
                generation_title = f"generation {generation}"
                if self._total_generations is not None:
                    generation_title += f"/{self._total_generations}"
                available_title_length = (
                    MAX_WINDOW_TITLE_LENGTH - len(generation_title) - 3
                )
                base_title = _fit_window_title(
                    self._title, max(available_title_length, 1)
                )
                window_title = _fit_window_title(f"{base_title} | {generation_title}")
                cv.setWindowTitle(self._title, window_title)
            cv.imshow(self._title, image)
            cv.waitKey(1)
        except cv.error as error:
            raise PreviewError(
                "debug preview could not update the graphical display"
            ) from error

    def close(self) -> None:
        """Close the preview window."""
        try:
            cv.destroyWindow(self._title)
        except cv.error:
            return

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()


class TerminalProgress:
    """Render solver phases with Click's terminal progress bar."""

    def __init__(self) -> None:
        self._bar: Any | None = None
        self._phase: str | None = None
        self._completed = 0

    def __call__(self, phase: str, current: int, total: int) -> None:
        if phase != self._phase:
            self._close_bar()
            self._bar = click.progressbar(
                length=total,
                label=phase,
                show_pos=True,
            )
            self._bar.__enter__()
            self._phase = phase
            self._completed = 0

        steps = current - self._completed
        if steps > 0 and self._bar is not None:
            self._bar.update(steps)
            self._completed = current

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self._close_bar()

    def _close_bar(self) -> None:
        if self._bar is not None:
            self._bar.__exit__(None, None, None)
            self._bar = None
            self._phase = None
            self._completed = 0
