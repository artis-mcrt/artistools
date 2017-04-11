# coding: utf-8

"""
Plotting and analysis tools for the ARTIS 3D supernova radiative transfer code.
"""

import datetime
import os

from setuptools import find_packages, setup

print(datetime.datetime.now().isoformat())
setup(name="artistools",
      version=datetime.datetime.now().isoformat(),
      author="Luke Shingles",
      author_email="luke.shingles@gmail.com",
      packages=find_packages(),
      url="https://www.github.com/lukeshingles/artis-tools/",
      license="MIT",
      description="Plotting and analysis tools for the ARTIS 3D supernova radiative transfer code.",
      long_description=open(os.path.join(os.path.dirname(__file__), "README.md")).read(),
      install_requires=open(os.path.join(os.path.dirname(__file__), "requirements.txt")).read(),
      entry_points={
          'console_scripts': [
              'plotartislightcurve = artistools.plot.lightcurve:main',
              'plotartisnlte = artistools.plot.nltepops:main',
              'plotartisnonthermal = artistools.plot.nonthermal:main',
              'plotartisradfield = artistools.plot.radfield:main',
              'plotartisspectrum = artistools.plot.spectrum:main',
              'plotartisspectrum.py = artistools.plot.spectrum:main',
          ]},
      python_requires='>==3.6',
      # test_suite='tests',
      setup_requires=['pytest-runner', 'pytest-cov'],
      tests_require=['pytest', 'pytest-runner', 'pytest-cov'],)
