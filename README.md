# Artistools

> Artistools is collection of plotting, analysis, and file format conversion tools for the ARTIS radiative transfer code.

[![Build Status](https://travis-ci.org/lukeshingles/artistools.svg?branch=master)](https://travis-ci.org/lukeshingles/artistools)
[![Coverage Status](https://coveralls.io/repos/github/lukeshingles/artistools/badge.svg?branch=master)](https://coveralls.io/github/lukeshingles/artistools?branch=master)

ARTIS (Sim et al. 2007; Kromer & Sim 2009) is a 3D radiative transfer code for Type Ia supernovae using the Monte Carlo method with indivisible energy packets (Lucy 2002). The simulation code is not publicly available.

## Installation
Clone the respository, then run:
```sh
python setup.py develop
```

## Usage
First cd into an ARTIS simulation folder and then run artistools with one of the subcommands:

usage: artistools &lt;command&gt;, where &lt;command&gt; is one of:

  - getmodeldeposition
  - makemodel1dslicefrom3d
  - makemodelbotyanski
  - plotestimators
  - plotlightcurve
  - plotnltepops
  - plotmacroatom
  - plotnonthermal
  - plotradfield
  - plotspectrum
  - plottransitions

Using the -h option will give a list of command-line arguments.

## Meta

Distributed under the MIT license. See ``LICENSE`` for more information.

[https://github.com/lukeshingles/artistools](https://github.com/lukeshingles/artistools))

-----------------------
This is also a bit of a testing ground for GitHub integrations:

[![Code Climate](https://codeclimate.com/github/lukeshingles/artistools/badges/gpa.svg)](https://codeclimate.com/github/lukeshingles/artistools) [![Test Coverage](https://codeclimate.com/github/lukeshingles/artistools/badges/coverage.svg)](https://codeclimate.com/github/lukeshingles/artistools/coverage) [![Issue Count](https://codeclimate.com/github/lukeshingles/artistools/badges/issue_count.svg)](https://codeclimate.com/github/lukeshingles/artistools)

[![Code Issues](https://www.quantifiedcode.com/api/v1/project/be02174519b14c45bcd765b468be6ee4/badge.svg)](https://www.quantifiedcode.com/app/project/be02174519b14c45bcd765b468be6ee4)

<!---
[![Code Health](https://landscape.io/github/lukeshingles/artistools/master/landscape.svg?style=flat)](https://landscape.io/github/lukeshingles/artistools/master)
-->

[![CodeFactor](https://www.codefactor.io/repository/github/lukeshingles/artistools/badge)](https://www.codefactor.io/repository/github/lukeshingles/artistools)

[![codebeat badge](https://codebeat.co/badges/ace84544-8781-4e3f-b86b-b21fb3f9fc87)](https://codebeat.co/projects/github-com-lukeshingles-artistools-master)


