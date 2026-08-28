# Instructions for LLM coding agents

Artistools is a toolkit that plots data, analyses data, and converts files for the [ARTIS](https://github.com/artis-mcrt/artis) radiative transfer code. It is a Python package (`artistools/`) and a small Rust extension (`rust/`, imported as `artistools.rustext`).

The package has **no public API**. You can delete code that has no callers. You can rename or refactor a function freely. Do not add a compatibility shim or a deprecation path for a name in the code, e.g. a function, a parameter, or a module. Correct the design instead.

A command-line argument is different. A user writes such an argument in a script and in a note, thus a
new spelling that takes the old one away stops that work. Keep the old spelling of a renamed argument
as an alias, and give it `help=argparse.SUPPRESS`. The help text then gives one spelling, and a script
that holds the old one still runs.

## Writing style

Write all English in ASD-STE100 (Simplified Technical English). This applies to comments, docstrings, documentation, commit messages, pull request text, and new log, warning, and error strings.

- Use the active voice. Write "the Makefile writes `version.h`" and not "`version.h` is written by the Makefile".
- Use the simple tenses. Do not use the -ing form as a noun or as an adjective if a simple form is possible.
- Use one term for one thing. Do not change between "cell" and "grid cell", or between "time step" and "timestep".
- Write short sentences. Use a maximum of 20 words in an instruction and a maximum of 25 words in descriptive text.
- Use a maximum of three words in a noun cluster. Write "the checksums of the output files" and not "output file checksum comparison".
- Write positive statements. Do not use slang, idioms, or jokes. Do not use an abbreviation that the text does not define.
- Use a vertical list for more than three related items or conditions.

Use the British spellings that this repository uses, e.g. "normalise", "parallelise", "colour", and "centre". Use an American spelling only when an external interface makes it necessary, e.g. the matplotlib keywords `color=` and `center=`, and the named colour `"gray"`. STE controls the choice of words and the structure of the sentences. It does not control the spelling variant.

These rules do not apply to:

- Identifiers in the code, e.g. the names of variables, functions, and namespaces. Keep the conventions of the file that you change, e.g. `at.normalize_path_list`.
- The names of the columns and the keys in the ARTIS files that artistools reads and writes.
- A log string that a script reads. Do not change such a string.
- Quoted text from an external source, e.g. a compiler message or a title of a publication.

## Setup

```sh
uv sync --all-extras                       # make or update .venv (needs a rust compiler)
cd tests/data && source ./setuptestdata.sh # get the simulation data for most tests
```

Most tests read data that the repository does not contain. Run the setup script above if a test fails because a path below `tests/data/testmodel` does not exist.

## Verify before you commit

Run every check that applies to the files that you changed. Run each check from the repository root.

```sh
uv run -- ruff format
uv run -- ruff check --no-fix     # --no-fix shows the same errors as CI (the config sets fix = true)
uv run -- pyrefly check
uv run -- ty check
uv run -- refurb artistools --quiet -- --follow-imports=skip   # only artistools/, because refurb runs mypy internally, and mypy fails on pyvista
uv run -- vulture                                      # informational, see below
uv run -- python -m pytest artistools/<area> -n auto   # e.g. artistools/spectra
cargo clippy --all-features -- -D warnings -D clippy::pedantic   # in rust/, for a Rust change
```

The type checkers pyrefly and ty must both give no errors. A change that satisfies one checker must not cause an error in a different checker. The file `.github/workflows/pytest.yml` defines what CI runs. CI also runs the `prek` hooks from `.pre-commit-config.yaml`. The vulture check gives information only. It reports code that possibly has no callers, but some of these reports are incorrect, thus CI does not fail on them.

Run the tests on one interpreter, which is the `.venv` of the repository. Do not run them again on
each version that CI tests. Two versions give a different result very rarely, and CI finds such a
case. A local run of every version costs minutes and finds almost nothing.

Do not report that a check passed if you did not run it. Tell the user which checks you did not run, and give the reason for each one.

## Python and type annotations

- Write code for Python 3.13 or a later version. The syntax must also be correct on the later versions and on the free-threaded builds that CI tests. The file `.github/workflows/pytest.yml` gives the list of versions. Do not add mutable state at module level. Do not use the GIL for thread safety.
- Give a full annotation to every function. The type checkers run in strict mode, and they report an untyped def or an untyped call as an error.
- Do not prefix a function name with an underscore.
- Use the modern generics: `list[str]`, `X | None`, and PEP 695 (`def f[T](...)`, `type Alias = ...`). Do not use `typing.List` or `typing.Optional`. Use `Any` only if no more accurate type is possible.
- Pyrefly must infer the parameter types of a lambda from the call site (`implicit-any-lambda`). A lambda accepts no annotations. Thus, if pyrefly cannot infer the types of a `sorted`, `min`, or `filter` key, write an annotated `def`. Use a comprehension in place of `filter(lambda ...)`.
- Put the message of an exception in a variable. Do not use a literal:
  ```python
  msg = f"Unknown path key: {key}"
  raise KeyError(msg)
  ```
- The maximum line length is 120 characters. `ruff format` does not reflow a comment, thus you must keep a comment inside the limit.
- Prefer readable code to a comment. Give each symbol a name that says what it holds, e.g. `isfirstoccurrence`, and delete the comment that the name replaces.
- A comment must give the reason for the code. Do not repeat what the code does. Write one or two lines. A comment of three lines or more must earn each one, e.g. a measurement that justifies a number. Give the numbers alone and not the full account of the experiment.
- Write a docstring of one line for a simple function. For a more complex function, write a summary of one line, then an empty line, then a longer description. Write the summary as an instruction: "Return the sum" and not "Returns the sum". Use the `"""` quotes and not `'''`.

## Suppress lint and type errors

Correct the initial problem first. Add a suppression only if you cannot correct the problem. Each tool has a different syntax. Always give the name of the applicable rule, because the tools reject a suppression that has no rule name.

| Tool | Syntax |
| --- | --- |
| ruff | `# ruff:ignore[rule-name]`, file-level `# ruff:file-ignore[rule-name]` |
| pyrefly | `# pyrefly: ignore[rule-name]` |
| pyright | `# pyright: ignore[rule-name]` |
| ty | `# ty:ignore[rule-name]` |

The configuration gives each ruff rule by **name** and not by code (`"any-type"` and not `"ANN401"`). Use the name in a suppression. The setting `enableTypeIgnoreComments` is off for pyright, thus a `# type: ignore` comment does not apply to pyright. One line can need more than one comment. For an example, see `artistools/_polarscompat.py`.

## Imports and module layout

- Use only absolute imports. A relative import is not permitted.
- Write one import on each line (`force-single-line`). Do not write `from x import a, b`.
- Use the aliases in `[tool.ruff.lint.flake8-import-conventions.extend-aliases]`, e.g. `artistools as at`, `polars as pl`, and `typing as t`. Ruff rejects a different alias.
- In a package module, import the applicable submodule or function, e.g. `from artistools.misc import get_nu_grid`. This prevents an import cycle. Use `import artistools as at` only in a test or a top-level script.
- Import a large or optional dependency in the function that uses it. Examples are pyvista, plotly, imageio, pynonthermal, and argcomplete. The rule `import-outside-top-level` is off for this reason, because the CLI must start quickly.
- flake8-type-checking runs in `strict` mode. Put an import that only the annotations use in an `if t.TYPE_CHECKING:` block after the usual imports. This also decreases the start time. If the code then uses that name at run time, move the import out of the block.
- Pyrefly reports `missing-import` as an error. Thus CI fails if you rename a module or make an error in its name. A plain `uv sync` does not install an optional dependency. Put such a dependency in `ignore-missing-imports` in `[tool.pyrefly]`.
- The option `implicit_reexport` is off. Re-export a new public function in the parent `__init__.py`. Use the form `from module import name as name`. Keep the alphabetical order of the other lines.

## Polars

Use polars (`import polars as pl`) for all dataframe code. The package has no pandas dependency. Do not add one.

- There is no `.group_by().filter()`. Do the aggregation first, then apply the filter.
- In an expression, write a column as `pl.col("colname")` and a literal as `pl.lit(...)`.
- A `None` value and a `float("nan")` value are different. `None` shows that the value is not available.
- Use a native expression in place of `.map_elements()`, which is slow and prevents the optimisation of the query. If no expression can do the same operation, always give `return_dtype`.
- Do not read a dataframe one row at a time. Use a vectorised expression, a join, or `dict(zip(df["a"], df["b"], strict=True))`. To put a flat array of cells on a 3D grid, use `arr.reshape((nx, ny, nz), order="F")` with `np.linspace`. Do not write nested loops over `nx`, `ny`, and `nz`.
- Use lazy evaluation. Build a `pl.LazyFrame` pipeline, accept a `pl.LazyFrame` in an internal function, and call `.collect()` one time at the end. Do not call `.collect()` in a loop over cells, timesteps, or ions, because this runs the full query again at each step. Call `.collect()` before the loop, then index the `pl.DataFrame`.
- Give the `on=` argument and the `how=` argument for each join.
- A method returns a new dataframe. There is no in-place operation, thus you must use the value that the method returns.
- In a test, compare two dataframes with `pltest.assert_frame_equal`.

## Command-line entry points

A module that supplies a subcommand has this structure:

```python
def addargs(parser: argparse.ArgumentParser) -> None: ...


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Give a description of one line for the CLI help text."""
    args = at.parse_cli_args(addargs, __doc__, args, argsraw, kwargs)
```

Add a new subcommand to `subcommandtree` in `artistools/commands.py`. Do not add a new console script. The CLI cannot find a module that has the `addargs` and `main` functions but no entry in `subcommandtree`. Such a module becomes incorrect, because no user and no test calls it. Add the entry or delete the module, but do not add more code to it.

Put the code that reads the data in functions that are separate from the plot code. A test can then call `main(argsraw=[], **kwargs)` for both parts. The `main` function must parse the arguments and then make a small number of calls. No rule limits the number of lines in a function. Let the structure of the code set the length.

A `main` function can already contain the data code, the physics code, and the plot code. Move the part that you change into a new function. Do not add more code to `main`.

Use the shared functions to build a parser and to find a path. Do not write this code again:

- `at.addarg_modelpath(parser)` adds a `-modelpath` argument or a positional model path.
- `at.normalize_path_list(args.modelpath)` applies the default and returns a `list[Path]`. Do not write `if not args.modelpath: args.modelpath = Path()`.
- `at.get_timestep_times(modelpath, loc="mid")` gives the mid-point times. Do not calculate the mean of the `start` array and the `end` array.
- `at.plottools.set_axis_properties` and `at.plottools.set_axis_labels` set the usual axis and tick properties. You can use `artistools/plottools.py` only as `at.plottools.*`, because the top level re-exports only `set_mpl_style`. Read that module before you write new plot code.

## Tests

- Put a test in the same directory as the code that it tests: `artistools/<area>/test_<area>.py`. Give each test function a `test_` prefix. The `name-tests-test` hook makes this necessary.
- Find the data with `at.get_path("testdata")` or `at.get_path("testoutput")`. As an alternative, use the pytest `tmp_path` fixture. Do not write a file outside the test output directory of the repository.
- Compare a number with `np.isclose` or an `rtol` value. Do not compare two floats for exact equality. A physics result changes by a small quantity on a different platform or with a different version of a library.
- You can test a plot without an image. Patch the axes method and examine the arguments of the call:
  ```python
  @mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
  def test_something(mockplot: mock.MagicMock) -> None: ...
  ```
- Add `@pytest.mark.benchmark` to a slow test that is representative. As an alternative, give the test a `benchmark: BenchmarkFixture` argument. CodSpeed then monitors that test.
- Write a regression test for each bug that you correct. If you change a numerical result deliberately, change the expected value in the same commit. Give the reason in the commit message.

## Performance and data access

- The code puts `@lru_cache` on a pure function that reads data for a given path or a given set of arguments. Each argument must be hashable. Do not change the object that a cached function returns. Make a copy first. For an example, see `artistools/misc/dirbins.py`.
- Write the cache of a parsed text file as parquet with `at.write_parquet_atomic`. This function writes a temporary file, then changes its name. Thus a run that stops early cannot leave a corrupt cache.
- Use `at.parallel_map`. Do not make a multiprocessing pool directly. This function selects threads or processes, because the GIL can be on or off.
- Read a compressed ARTIS output file with `at.zopen` for text, or give polars a source with `at.polars_source`. `at.polars_source` takes the path of a file that exists, e.g. the path that `at.firstexisting` returns. Use `at.zopenpl` when the caller gives a name that can need a compressed sibling. Do not write your own code for the `.gz`, `.xz`, and `.zst` formats. A reader that is faster is permitted: give the measurements in the commit message.
- Do not add rows to a dataframe in a loop. A repeated `pl.concat` call has a cost of O(n²) for n steps. Collect the parts in a list, then concatenate them one time after the loop.
- Use the lazy scanner that reads a full run, e.g. `at.scan_estimators`. The eager reader `at.estimators.read_estimators` operates on one cell, and it is very slow for many cells and timesteps. Do not call an eager reader in a loop over the cells.

## Repository rules

- The file `pyproject.toml` holds all the tool configuration. Do not add a configuration file for one tool. Do not add an inline setting that repeats a value from `pyproject.toml`.
- Add a dependency only if it is necessary. Put a run-time dependency in `[project.dependencies]`. Put a large optional dependency in `[project.optional-dependencies].extras`. Put a tool in `[dependency-groups].dev`. Then run `uv lock` and commit `uv.lock`.
- Do not commit a plot file or a file larger than 1 MB. The pre-commit hook rejects a larger
  file. Simulation output below that limit is permitted, e.g. the small test models in `tests/data`.
- Before you rename or delete a name, search the full repository for it. Examine `artistools/commands.py`, the re-exports in each `__init__.py`, and the tests. Change all the occurrences in one commit.
- Delete the old code when you replace it or make it inactive. Do not keep it as a comment, because git holds the old version. The ruff rule `commented-out-code` is off and it has no fix, thus no tool finds such code for you. A comment block that is documentation is different: an example that shows the permitted values of a setting, or a published benchmark configuration that a user can run again. Keep such an example, but delete code that a new version replaced. If you are not sure which type a block is, keep it and ask the user.
