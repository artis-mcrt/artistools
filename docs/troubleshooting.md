# Caches and common problems

## The caches

A reader converts a large text file to a parquet cache at the first read, because a later read of the cache is much
faster. The cache is beside the text file, e.g. `model.txt.parquet` in the model folder and
`packetsbatch<n>_<first>_<last>.out.parquet` in the folder of the packets files. A cache of the estimators is in each
run folder.

- `artistools cache make mymodel` writes all the caches of a model at one time, e.g. in the job script after a run of
  ARTIS.
- `artistools cache list mymodel` gives the caches and their sizes.
- `artistools cache delete mymodel` deletes them.

A reader writes a new cache when the text file changes. Thus a model folder where the user cannot write a file needs
current caches. Run `artistools cache make mymodel` as a user who can write to the model folder. Other users can then
read the model.

## Common problems

### "This command needs ..., which is not installed"

The feature needs a package of `artistools[extras]`. Install artistools with the extras, e.g.
`uv tool install 'artistools[extras]'`.

### `--interactive` does not open a window

The window needs PySide6, which a free-threaded Python cannot load. Use a Python that is not free-threaded.

### The error message is not enough

Set `ARTISTOOLS_TRACEBACK=1` to get the full traceback of an error, e.g. `ARTISTOOLS_TRACEBACK=1 artistools plotspectra -t 300 mymodel`.

### The command reads the incorrect model or the incorrect time

Give `--explain`. The command then prints the model folder, the timesteps, and the output file, and it stops before
it reads the data.

### `at` runs a different command

Some systems have a command `at` that runs a job at a later time. Use `artistools`, or put the folder of artistools
first in `PATH`. For the tab completion of zsh, `artistools completions` gives the steps. These steps put the completion of artistools
before the completion of `at`.

### The command does not know the name of a variable or a column

`artistools plotestimators --listvariables mymodel` gives each estimator variable with its units.
`artistools describe mymodel` gives the output files, the ions, and the direction bins.
