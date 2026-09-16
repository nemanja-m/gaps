import sys


def print_progress(
    iteration: int,
    total: int,
    prefix: str = "",
    suffix: str = "",
    decimals: int = 1,
    bar_length: int = 50,
) -> None:
    """Write a single progress-bar update to stdout."""
    progress = 1.0 if total <= 0 else min(max(iteration / total, 0.0), 1.0)
    percentage = f"{progress * 100:.{decimals}f}"
    filled_length = round(bar_length * progress)
    bar = "\033[32m█\033[0m" * filled_length
    bar += "\033[31m-\033[0m" * (bar_length - filled_length)

    sys.stdout.write(f"\r{prefix: <16} {bar} {percentage}% {suffix}")
    if progress == 1.0:
        sys.stdout.write("\n")
    sys.stdout.flush()
