# Artistools

> Artistools is collection of plotting, analysis, and file format conversion tools for the ARTIS radiative transfer code.

[![Build Status](https://travis-ci.com/lukeshingles/artistools.svg?branch=master)](https://travis-ci.com/lukeshingles/artistools)
[![Coverage Status](https://coveralls.io/repos/github/lukeshingles/artistools/badge.svg?branch=master)](https://coveralls.io/github/lukeshingles/artistools?branch=master)

ARTIS (Sim et al. 2007; Kromer & Sim 2009) is a 3D radiative transfer code for Type Ia supernovae using the Monte Carlo method with indivisible energy packets (Lucy 2002). The simulation code is not publicly available.

## Installation
First clone the repository, for example:
```sh
git clone https://github.com/lukeshingles/artistools.git
```
Then from the repo directory run:
```sh
pip install -e .
```

## Usage
Artistools provides the following commands:
  - getartismodeldeposition
  - getartisspencerfano
  - makeartismodel1dslicefrom3d
  - makeartismodelbotyanski
  - plotartisestimators
  - plotartislightcurve
  - plotartisnltepops
  - plotartismacroatom
  - plotartisnonthermal
  - plotartisradfield
  - plotartisspectrum
  - plotartistransitions

Use the -h option to get a list of command-line arguments for each subcommand. Most of these commands would usually be run from within an ARTIS simulation folder.

## Example output

![Emission plot](images/fig-emission.png)
![NLTE plot](images/fig-nlte-Ni.png)
![Estimator plot](images/fig-estimators.png)

## Meta

Distributed under the MIT license. See ``LICENSE`` for more information.

[https://github.com/lukeshingles/artistools](https://github.com/lukeshingles/artistools)

-----------------------
This is also a bit of a testing ground for GitHub integrations:

[![Code Climate](https://codeclimate.com/github/lukeshingles/artistools/badges/gpa.svg)](https://codeclimate.com/github/lukeshingles/artistools)

[![Test Coverage](https://codeclimate.com/github/lukeshingles/artistools/badges/coverage.svg)](https://codeclimate.com/github/lukeshingles/artistools/coverage)

[![Issue Count](https://codeclimate.com/github/lukeshingles/artistools/badges/issue_count.svg)](https://codeclimate.com/github/lukeshingles/artistools)

<!---
[![Code Health](https://landscape.io/github/lukeshingles/artistools/master/landscape.svg?style=flat)](https://landscape.io/github/lukeshingles/artistools/master)
-->

[![CodeFactor](https://www.codefactor.io/repository/github/lukeshingles/artistools/badge)](https://www.codefactor.io/repository/github/lukeshingles/artistools)

[![codebeat badge](https://codebeat.co/badges/ace84544-8781-4e3f-b86b-b21fb3f9fc87)](https://codebeat.co/projects/github-com-lukeshingles-artistools-master)


