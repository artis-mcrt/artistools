#!/usr/bin/env python3

"""Plotting and analysis tools for the ARTIS 3D supernova radiative transfer code."""

import sys

from pathlib import Path
from setuptools import find_packages, setup
from setuptools_scm import get_version

sys.path.append('artistools/')
from commands import get_console_scripts, get_completioncommands


# Add the following lines to your .zshrc file to get command completion:
# autoload -U bashcompinit
# bashcompinit
# source artistoolscompletions.sh
completioncommands = get_completioncommands()
with open("artistoolscompletions.sh", "w") as f:
    f.write("\n".join(completioncommands))


setup(
    name="artistools",
    version=get_version(),
    use_scm_version=True,
    author="ARTIS Collaboration",
    author_email="luke.shingles@gmail.com",
    packages=find_packages(),
    url="https://www.github.com/artis-mcrt/artistools/",
    license="MIT",
    description="Plotting and analysis tools for the ARTIS 3D supernova radiative transfer code.",
    long_description=(Path(__file__).absolute().parent / "README.md").open("rt").read(),
    long_description_content_type="text/markdown",
    install_requires=(Path(__file__).absolute().parent / "requirements.txt").open("rt").read().splitlines(),
    entry_points={
        "console_scripts": get_console_scripts(),
    },
    scripts=["artistoolscompletions.sh"],
    setup_requires=["psutil>=5.9.0", "setuptools>=45", "setuptools_scm[toml]>=6.2", "wheel"],
    tests_require=["pytest", "pytest-runner", "pytest-cov"],
    include_package_data=True,
)
