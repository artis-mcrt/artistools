# Artistools

> Artistools is collection of plotting, analysis, and file format conversion tools for the ARTIS radiative transfer code.

![GitHub Build and test status](https://github.com/artis-mcrt/artistools/workflows/Build%20and%20test/badge.svg)
[![Build Status](https://travis-ci.com/artis-mcrt/artistools.svg?branch=main)](https://travis-ci.com/lukeshingles/artistools)
[![Coverage Status](https://coveralls.io/repos/github/artis-mcrt/artistools/badge.svg?branch=main)](https://coveralls.io/github/artis-mcrt/artistools?branch=main)

ARTIS (Sim et al. 2007; Kromer & Sim 2009) is a 3D radiative transfer code for Type Ia supernovae using the Monte Carlo method with indivisible energy packets (Lucy 2002). The simulation code is not publicly available.

## Installation
First clone the repository, for example:
```sh
git clone https://github.com/artis-mcrt/artistools.git
```
Then from the repo directory run:
```sh
pip install -e .
```

## Usage
Artistools provides commands including:
  - artistools-deposition
  - artistools-spencerfano
  - artistools-make1dslicefrom3dmodel
  - artistools-estimators
  - artistools-lightcurve
  - artistools-nltepops
  - artistools-nonthermal
  - artistools-radfield
  - artistools-spectrum
  - artistools-transitions

Use the -h option to get a list of command-line arguments for each subcommand. Most of these commands would usually be run from within an ARTIS simulation folder.

## Example output

![Emission plot](images/fig-emission.png)
![NLTE plot](images/fig-nlte-Ni.png)
![Estimator plot](images/fig-estimators.png)

## Meta

Distributed under the MIT license. See ``LICENSE`` for more information.

[https://github.com/artis-mcrt/artistools](https://github.com/artis-mcrt/artistools)

-----------------------
This is also a bit of a testing ground for GitHub integrations:

[![Code Climate](https://codeclimate.com/github/artis-mcrt/artistools/badges/gpa.svg)](https://codeclimate.com/github/artis-mcrt/artistools)

[![Test Coverage](https://codeclimate.com/github/artis-mcrt/artistools/badges/coverage.svg)](https://codeclimate.com/github/artis-mcrt/artistools/coverage)

[![Issue Count](https://codeclimate.com/github/artis-mcrt/artistools/badges/issue_count.svg)](https://codeclimate.com/github/artis-mcrt/artistools)

<!---
[![Code Health](https://landscape.io/github/artis-mcrt/artistools/main/landscape.svg?style=flat)](https://landscape.io/github/artis-mcrt/artistools/main)
-->

[![CodeFactor](https://www.codefactor.io/repository/github/artis-mcrt/artistools/badge)](https://www.codefactor.io/repository/github/artis-mcrt/artistools)

[![codebeat badge](https://codebeat.co/badges/ace84544-8781-4e3f-b86b-b21fb3f9fc87)](https://codebeat.co/projects/github-com-lukeshingles-artistools-main)


