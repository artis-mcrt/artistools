# Caches and common problems

## The caches

A reader converts a large text file to a parquet cache at the first read, because a later read of the cache is much
faster. The cache is beside the text file, e.g. `model.txt.parquet` in the model folder and
`packetsbatch<n>_<first>_<last>.out.parquet` in the folder of the packets files. A cache of the estimators is in each
run folder.

- `artistools cache make mymodel` writes these caches at one time, e.g. in the job script after a run of ARTIS:
  - the input model;
  - the line list;
  - the estimators;
  - the packets;
  - the virtual packets.
- `artistools cache list mymodel` gives the caches and their sizes.
- `artistools cache delete mymodel` deletes them.

`cache make` does not convert the NLTE populations. A read of all the cells converts the text of each run folder that
holds a timestep of the read to `nltepops_allranks.out.parquet` in that folder. A restarted model has more than one run
folder, thus the read takes one timestep of each run folder:

```sh
"$(artistools python)" -P - <<'EOF'
import artistools as at

modelpath = "mymodel"
timesteps = [at.misc.get_runfolder_timesteps(folder)[0] for folder in at.misc.get_runfolders(modelpath)]
at.nltepops.read_nltepops(modelpath, timestep=timesteps)
EOF
```

The conversion takes all the timesteps of the text of a run folder, thus a timestep with no NLTE populations, e.g. an
LTE timestep, also writes the cache. A read of some cells, e.g. a run of `artistools plotnltepops`, reads only the
ranks of these cells. Thus it writes no cache.

artistools makes a new NLTE cache only when the NLTE files hold 1 GiB of text or more. A reader takes a smaller text
directly. A cache that exists is different: when the text changes, a reader replaces the cache for each size of
text. Thus a model folder that holds a cache needs a current cache.

A reader writes a new cache when the text file changes. Thus a model folder where the user cannot write a file needs
current caches. Do these steps as a user who can write to the model folder:

1. Run `artistools cache make mymodel`.
2. If the model has NLTE populations, run the command above.

Other users can then read the model.

## Common problems

### "This command needs ..., which is not installed"

The feature needs a package of `artistools[extras]`. Install artistools with the extras, e.g.
`uv tool install 'artistools[extras]'`.

### `--interactive` does not open a window

The window needs PySide6, which a free-threaded Python cannot load. Use a Python that is not free-threaded.

### The error message is not enough

Set `ARTISTOOLS_TRACEBACK=1` to get the full traceback of an error, e.g. `ARTISTOOLS_TRACEBACK=1 artistools plotspectra -t 300 mymodel`.

### The command reads the incorrect model or the incorrect time

Give `--explain`. The command then prints the model folder, the timesteps, and the file of `-o`, and it stops before
it reads the data.

### `at` runs a different command

Some systems have a command `at` that runs a job at a later time. Use `artistools`, or put the folder of artistools
first in `PATH`. For the tab completion of zsh, `artistools completions` gives the steps. These steps put the
completion of artistools before the completion of `at`.

### The command does not know the name of a variable or a column

`artistools plotestimators --listvariables mymodel` gives each estimator variable with its units.
`artistools describe mymodel` gives the output files, the ions, and the direction bins.
