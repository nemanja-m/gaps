from setuptools import setup

setup(
    name="gaps",
    author="Nemanja Milicevic",
    author_email="the.nemanja.milicevic@gmail.com",
    description="Genetic-Algorithm based jigsaw puzzle solver",
    license="MIT",
    url="https://github.com/nemanja-m/genetic-jigsaw-solver",
    packages=["solver"],
    scripts=["bin/create_puzzle", "bin/solve"]
)
