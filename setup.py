#!/usr/bin/env python3

# coding: utf-8
"""Plotting and analysis tools for the ARTIS 3D supernova radiative transfer code."""

import datetime
import sys

from pathlib import Path
from setuptools import find_packages, setup
from setuptools.command.test import test as TestCommand

sys.path.append('artistools/')
from commands import console_scripts


class PyTest(TestCommand):
    """Setup the py.test test runner."""

    def finalize_options(self):
        """Set options for the command line."""
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        """Execute the test runner command."""
        # Import here, because outside the required eggs aren't loaded yet
        import pytest
        sys.exit(pytest.main(self.test_args))


print(datetime.datetime.now().isoformat())
setup(
    name="artistools",
    version="2021.04.24dev",
    # version=datetime.datetime.now().isoformat(),
    author="ARTIS Collaboration",
    author_email="luke.shingles@gmail.com",
    packages=find_packages(),
    url="https://www.github.com/artis-mcrt/artistools/",
    license="MIT",
    description="Plotting and analysis tools for the ARTIS 3D supernova radiative transfer code.",
    long_description=(Path(__file__).absolute().parent / "README.md").open('rt').read(),
    long_description_content_type='text/markdown',
    install_requires=(Path(__file__).absolute().parent / "requirements.txt").open('rt').read().splitlines(),
    entry_points={
        'console_scripts': console_scripts
    },
    python_requires='>==3.6',
    # test_suite='tests',
    setup_requires=['pytest', 'pytest-runner', 'pytest-cov'],
    tests_require=['pytest', 'pytest-runner', 'pytest-cov'],
    include_package_data=True)
