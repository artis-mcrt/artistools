# coding: utf-8
"""Plotting and analysis tools for the ARTIS 3D supernova radiative transfer code."""

import datetime
import os
import sys

from setuptools import find_packages, setup
from setuptools.command.test import test as TestCommand


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
    version=datetime.datetime.now().isoformat(),
    author="Luke Shingles",
    author_email="luke.shingles@gmail.com",
    packages=find_packages(),
    url="https://www.github.com/lukeshingles/artis-tools/",
    license="MIT",
    description="Plotting and analysis tools for the ARTIS 3D supernova radiative transfer code.",
    long_description=open(
        os.path.join(os.path.dirname(__file__), "README.md")).read(),
    install_requires=open(
        os.path.join(os.path.dirname(__file__), "requirements.txt")).read(),
    entry_points={
        'console_scripts': [
            'plotartislightcurve = artistools.plot.lightcurve:main',
            'plotartisnltepops = artistools.plot.nltepops:main',
            'plotartismacroatom = artistools.plot.macroatom:main',
            'plotartisnonthermal = artistools.plot.nonthermalspec:main',
            'plotartisradfield = artistools.plot.radfield:main',
            'plotartisspectrum = artistools.plot.spectrum:main',
        ]
    },
    python_requires='>==3.6',
    # test_suite='tests',
    setup_requires=['pytest-runner', 'pytest-cov'],
    tests_require=['pytest', 'pytest-runner', 'pytest-cov'],
    include_package_data=True)
