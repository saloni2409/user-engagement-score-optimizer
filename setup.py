"""Setup script for development installation."""
from setuptools import setup, find_packages

setup(
    name="user-engagement-score-optimizer",
    version="0.1.0",
    package_dir={"": "src"},  # tell setuptools packages are under src
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "pytest",
        "matplotlib",
        "jupyter",
        "scipy",
    ],
    python_requires=">=3.8",
)