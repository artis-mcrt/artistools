# coding: utf-8

"""
Plotting and analysis tools for the ARTIS 3D supernova radiative transfer code.
"""

import os

from setuptools import find_packages, setup

setup(name="artistools",
      version=0.1,
      author="Luke Shingles",
      author_email="luke.shingles@gmail.com",
      packages=find_packages(),
      url="https://www.github.com/lukeshingles/artis-tools/",
      license="MIT",
      description="Plotting and analysis tools for the ARTIS 3D supernova radiative transfer code.",
      long_description=open(os.path.join(os.path.dirname(__file__), "README.md")).read(),
      install_requires=open(os.path.join(os.path.dirname(__file__), "requirements.txt")).read(),
      python_requires='>==3.6',
      # test_suite='tests',
      setup_requires=['pytest-runner'],
      tests_require=['pytest', 'pytest-runner', 'pytest-cov'],)
