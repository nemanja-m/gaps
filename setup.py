from setuptools import setup

setup(
    name="gaps",
    version="1.0",
    author="Nemanja Milicevic",
    author_email="the.nemanja.milicevic@gmail.com",
    description="Genetic-Algorithm based jigsaw puzzle solver",
    license="MIT",
    url="https://github.com/nemanja-m/gaps",
    packages=[
        "gaps",
        "tests"
    ],
    install_requires=[
        "numpy",
        "matplotlib",
        "opencv-python"
    ],
    scripts=[
        "bin/create_puzzle",
        "bin/gaps"
    ]
)
