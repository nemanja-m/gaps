# Grayscale validation images

These deterministic images exercise the native grayscale path (`H x W`, not
three duplicated color channels):

- `grayscale_source.png` — the original 256x256 image.
- `grayscale_puzzle.png` — the same image shuffled into sixteen 64x64 pieces.
- `grayscale_expected.png` — a copy of the source for easy comparison.

Regenerate them with:

```bash
uv run python scripts/generate_grayscale_images.py
```

Solve the puzzle explicitly specifying the piece size:

```bash
uv run gaps run \
  images/grayscale/grayscale_puzzle.png \
  /tmp/grayscale_solution.png \
  --size=64 \
  --population=100 \
  --generations=30 \
  --seed=23
```

Compare a result with the expected image:

```bash
cmp images/grayscale/grayscale_expected.png /tmp/grayscale_solution.png
```

A successful solve produces no output from `cmp`. The generated test image has
exact values at the intended tile seams, making it a deterministic functional
check while remaining visually useful for manual inspection.
