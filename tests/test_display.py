from typing import cast

import click
import cv2 as cv
import numpy as np

from gaps.display import MAX_WINDOW_TITLE_LENGTH, OpenCVPreview, TerminalProgress


def test_opencv_preview_updates_current_solution(monkeypatch):
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        cv,
        "namedWindow",
        lambda title, flags: calls.append(("namedWindow", (title, flags))),
    )
    monkeypatch.setattr(
        cv,
        "setWindowTitle",
        lambda title, new_title: calls.append(("setWindowTitle", (title, new_title))),
    )
    monkeypatch.setattr(
        cv,
        "imshow",
        lambda title, image: calls.append(("imshow", (title, image))),
    )
    monkeypatch.setattr(cv, "waitKey", lambda delay: calls.append(("waitKey", delay)))
    monkeypatch.setattr(
        cv, "destroyWindow", lambda title: calls.append(("destroyWindow", title))
    )

    image = np.zeros((2, 2, 3), dtype=np.uint8)
    preview = OpenCVPreview("test", total_generations=20)
    preview.show(image, generation=3)
    preview.close()

    assert calls[0][0] == "namedWindow"
    assert calls[1] == ("setWindowTitle", ("test", "test | generation 3/20"))
    assert calls[2][0] == "imshow"
    imshow_call = cast(tuple[str, np.ndarray], calls[2][1])
    assert imshow_call[0] == "test"
    assert np.array_equal(imshow_call[1], image)
    assert calls[3] == ("waitKey", 1)
    assert calls[4] == ("destroyWindow", "test")


def test_opencv_preview_limits_window_title_length(monkeypatch):
    window_titles: list[str] = []
    monkeypatch.setattr(cv, "namedWindow", lambda _title, _flags: None)
    monkeypatch.setattr(
        cv, "setWindowTitle", lambda _title, new_title: window_titles.append(new_title)
    )
    monkeypatch.setattr(cv, "imshow", lambda _title, _image: None)
    monkeypatch.setattr(cv, "waitKey", lambda _delay: None)

    preview = OpenCVPreview("gaps | " + "input-" * 30 + ".png", 20)
    preview.show(np.zeros((2, 2, 3), dtype=np.uint8), generation=3)

    assert len(window_titles) == 1
    assert len(window_titles[0]) <= MAX_WINDOW_TITLE_LENGTH
    assert window_titles[0].endswith(" | generation 3/20")


def test_terminal_progress_reuses_bars_per_phase(monkeypatch):
    class FakeBar:
        def __init__(self, options: dict[str, object]) -> None:
            self.label = str(options["label"])
            self.updates: list[int] = []
            bars.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def update(self, steps: int) -> None:
            self.updates.append(steps)

    bars: list[FakeBar] = []
    monkeypatch.setattr(click, "progressbar", lambda **options: FakeBar(options))

    with TerminalProgress() as progress:
        progress("analysis", 1, 2)
        progress("analysis", 2, 2)
        progress("evolution", 1, 1)

    assert [bar.label for bar in bars] == ["analysis", "evolution"]
    assert bars[0].updates == [1, 1]
    assert bars[1].updates == [1]
