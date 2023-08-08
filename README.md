<h1 align=center>
  <img src="logo/LogoHorizontal.png" width=50%>
</h1>

Genetic Algorithm based solver for jigsaw puzzles with piece size
auto-detection.

[![gaps](https://github.com/nemanja-m/gaps/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/nemanja-m/gaps/actions/workflows/ci.yml)

<p align="center">
  <img src="images/lena.gif" alt="demo" />
</p>

# Installation

Clone repo:

```bash
git clone https://github.com/nemanja-m/gaps.git
cd gaps
```

Install requirements:

```bash
poetry install
```

Install project locally:

```bash
pip install .
```

# Creating puzzles from images

To create puzzle from image use `gaps create`

```bash
gaps create images/pillars.jpg puzzle.jpg --size=64
```

will create puzzle with 240 pieces from `images/pillars.jpg` where each piece is
64x64 pixels.

<div align="center">
  <img src="images/pillars.jpg" alt="original" width="250" height="180" />
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
  <img src="images/demo_puzzle.jpg" alt="puzzle" width="250" height="180" />
</div>

Run `gaps create --help` for detailed help.

__NOTE__: Created puzzle image dimensions may be smaller then original image
depending on the given puzzle piece size. Maximum possible rectangle is cropped
from original image.

# Solving puzzles

In order to solve puzzles, use `gaps run`:

```bash
gaps run puzzle.jpg solution.jpg --generations=20 --population=600
```

This will start genetic algorithm with initial population of 600 and 20 generations.

Following options are provided:

Option          | Description
--------------- | -----------
`--size`        | Puzzle piece size in pixels
`--generations` | Number of generations for genetic algorithm
`--population`  | Number of individuals in population
`--debug`       | Show the best solution after each generation

Run `gaps run --help` for detailed help.

## Size detection

If you don't explicitly provide `--size` argument to `gaps run`, piece size will
be detected automatically.

However, you can always provide `gaps run` with `--size` argument explicitly:

```bash
gaps run puzzle.jpg solution.jpg --generations=20 --population=600 --size=48
```

__NOTE__: Size detection feature works for the most images but there are some edge cases
where size detection fails and detects incorrect piece size. In that case you can
explicitly set piece size.

## Termination condition

The termination condition of a Genetic Algorithm is important in determining
when a GA run will end.  It has been observed that initially, the GA progresses
very fast with better solutions coming in every few iterations, but this tends
to saturate in the later stages where the improvements are very small.

`gaps` will terminate:

* when there has been no improvement in the population for `X` iterations, or
* when it reaches an absolute number of generations

# References

BibTeX entry:

```text
@article{Sholomon2016,
  doi = {10.1007/s10710-015-9258-0},
  url = {https://doi.org/10.1007/s10710-015-9258-0},
  year = {2016},
  month = feb,
  publisher = {Springer Science and Business Media {LLC}},
  volume = {17},
  number = {3},
  pages = {291--313},
  author = {Dror Sholomon and Omid E. David and Nathan S. Netanyahu},
  title = {An automatic solver for very large jigsaw puzzles using genetic algorithms},
  journal = {Genetic Programming and Evolvable Machines}
}
```

# License

This project as available as open source under the terms of the [MIT License](http://opensource.org/licenses/MIT)
