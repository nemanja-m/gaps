# About

A Genetic Algorithm-Based Solver for Jigsaw Puzzles :cyclone:

# Usage

Requirements:

- `python 2.6+`
- `numpy`
- `scipy`
- `sortedcontainers`

Install with:

``` bash
git clone https://github.com/nemanja-m/genetic-jigsaw-solver.git
chmox +x bin/create_puzzle bin/solve
```

Create puzzle from image:

``` bash
bin/create_puzzle <image> [--piece-size] [--destination]
```

Run solver:

``` bash
bin/solve [--image] [--generations] [--population] [--size] [--verbose]
```

i.e.

``` bash
bin/create_puzzle images/lena.jpg --piece-size=32 --destination=puzzle.jpg

# => Puzzle output to 'puzzle.jpg'

bin/solve --image=puzzle.jpg --size=32 --generations=30 --population=300 --verbose=True

# => Solver started
```

# Algorithm

// TODO
