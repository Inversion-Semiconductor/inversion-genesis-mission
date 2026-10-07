from setuptools import find_packages, setup


setup(
    name="exploration_fbpic",
    version="0.1.0",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "pyarrow>=14",
        "PyYAML>=6.0",
        "scipy>=1.10",
        "xopt>=2.6",
    ],
    extras_require={
        "nersc": ["libensemble>=1.5", "mpi4py"],
        "dev": ["pytest>=8"],
    },
    entry_points={
        "console_scripts": [
            "explore-fbpic=exploration_fbpic.cli:main",
        ],
    },
)
