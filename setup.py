#!/usr/bin/env python3
# mypy: ignore-errors
"""Plotting and analysis tools for the ARTIS 3D supernova radiative transfer code."""
import importlib.util
import sys
from pathlib import Path

from setuptools import find_namespace_packages
from setuptools import setup
from setuptools_scm import get_version

spec = importlib.util.spec_from_file_location("commands", "./artistools/commands.py")
commands = importlib.util.module_from_spec(spec)
spec.loader.exec_module(commands)

setup(
    name="artistools",
    version=get_version(),
    use_scm_version={"local_scheme": "no-local-version"},
    author="ARTIS Collaboration",
    author_email="luke.shingles@gmail.com",
    packages=find_namespace_packages(where="."),
    package_dir={"": "."},
    url="https://www.github.com/artis-mcrt/artistools/",
    long_description=(Path(__file__).absolute().parent / "README.md").open("rt").read(),
    long_description_content_type="text/markdown",
    install_requires=(Path(__file__).absolute().parent / "requirements.txt").open("rt").read().splitlines(),
    entry_points={
        "console_scripts": commands.get_console_scripts(),
    },
    scripts=["artistoolscompletions.sh"],
    setup_requires=["psutil>=5.9.0", "setuptools>=45", "setuptools_scm[toml]>=6.2", "wheel"],
    tests_require=["pytest", "pytest-runner", "pytest-cov"],
    include_package_data=True,
)
