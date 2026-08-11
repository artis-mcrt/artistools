# Instructions for LLM coding agents

Artistools is a plotting, analysis, and file-conversion toolkit for the [ARTIS](https://github.com/artis-mcrt/artis) radiative transfer code. It is a Python package (`artistools/`) plus a small Rust extension (`rust/`, imported as `artistools.rustext`).

The package has **no public API**: code with no callers can be deleted, and functions can be renamed or refactored freely. Prefer fixing a design properly over adding a compatibility shim or a deprecation path.

## Setup

```sh
uv sync --all-extras                       # create/refresh .venv (needs a rust compiler)
cd tests/data && source ./setuptestdata.sh # download simulation data used by most tests
```

Most tests read data that is not in the repository. If tests fail on missing paths under `tests/data/testmodel`, run the setup script above.

## Verify before committing

Run every check that covers what you touched, from the repository root:

```sh
uv run -- ruff format
uv run -- ruff check --no-fix     # --no-fix shows what CI sees (config sets fix = true)
uv run -- mypy
uv run -- basedpyright --warnings
uv run -- pyrefly check
uv run -- ty check
uv run -- python -m pytest artistools/<area> -n auto   # e.g. artistools/spectra
cargo clippy --all-features -- -D warnings -D clippy::pedantic   # in rust/, for Rust changes
```

All four type checkers must be clean, and a change that satisfies one must not break another. `.github/workflows/pytest.yml` is the authority on what CI runs; it also runs the `prek` hooks from `.pre-commit-config.yaml`.

Never report that checks passed when you could not run them. If the environment lacks the tools, the venv, the network, or the test data, say which checks were skipped and why.

## Python and typing

- Target Python >= 3.13 and keep syntax valid on 3.14, including free-threaded builds (`3.14t`), which CI tests. Do not add module-level mutable state or rely on the GIL for thread safety.
- Annotate every function fully: mypy runs in strict mode and basedpyright/pyright in strict type-checking mode. Untyped defs and untyped calls are errors.
- Use modern generics: `list[str]`, `X | None`, PEP 695 (`def f[T](...)`, `type Alias = ...`). No `typing.List`, `typing.Optional`, or bare `Any` where a real type fits.
- A lambda's parameter types have to be inferable from the call site (pyrefly's `implicit-any-lambda`). A `sorted`/`min`/`filter` key that pyrefly cannot infer becomes an annotated `def` instead — lambdas take no annotations. Prefer a comprehension over `filter(lambda ...)` either way.
- Raise exceptions with a named message variable, not a literal:
  ```python
  msg = f"Unknown path key: {key}"
  raise KeyError(msg)
  ```
- Line length is 120. Comments should explain *why*, not restate the code.
- Docstrings should be one line for simple functions, or a one-line summary followed by a blank line and a longer description. Use imperative mood for summaries: "Return the sum" not "Returns the sum". Use `"""` triple quotes, not `'''`.

## Suppressing lint and type errors

Fix the underlying issue first; a suppression is a last resort and each tool has its own syntax. Always include the specific rule name — blanket suppressions are rejected.

| Tool | Syntax |
| --- | --- |
| ruff | `# ruff:ignore[rule-name]`, file-level `# ruff:file-ignore[rule-name]` |
| mypy | `# type: ignore[error-code]` (a bare `# type: ignore` fails `ignore-without-code`) |
| basedpyright | `# pyright: ignore[reportRuleName]` |
| pyrefly | `# pyrefly: ignore[rule-name]` |
| ty | `# ty:ignore[rule-name]` |

Ruff rules are configured by **name**, not by code (`"any-type"`, not `"ANN401"`) — match that in suppressions. `enableTypeIgnoreComments` is off for pyright, so `# type: ignore` silences only mypy. One line may need several comments (see `artistools/lightcurve/plotlightcurve.py`).

`warn_unused_ignores` is on: when you fix the underlying error, delete the now-stale suppression or mypy will fail.

## Imports and module layout

- Absolute imports only; relative imports are banned.
- One import per line (`force-single-line`); no `from x import a, b`.
- Use the configured aliases: `artistools as at`, `polars as pl`, `polars.selectors as cs`, `polars.testing as pltest`, `numpy as np`, `numpy.typing as npt`, `matplotlib.pyplot as plt`, `matplotlib.axes as mplax`, `matplotlib.figure as mplfig`, `typing as t`.
- Inside package modules, import the specific submodule or function (`from artistools.misc import get_nu_grid`) to avoid import cycles. `import artistools as at` is for tests and top-level scripts.
- Import heavy or optional dependencies (astropy, pyvista, plotly, imageio, pynonthermal, argcomplete) inside the function that needs them. `import-outside-top-level` is disabled deliberately to keep CLI startup fast.
- flake8-type-checking runs in `strict` mode: an import used *only* in annotations belongs in an `if t.TYPE_CHECKING:` block after the normal imports, for the same startup-cost reason. Adding a runtime use of that name means moving the import back out.
- Pyrefly's `missing-import` is an error, so a renamed or mistyped module fails CI. A genuinely optional dependency that a plain `uv sync` would not install belongs in `ignore-missing-imports` in `[tool.pyrefly]`.
- `implicit_reexport` is off: a new public function must be re-exported in the parent `__init__.py` using the `from module import name as name` form, matching the surrounding alphabetical order.

## Polars

Use polars (`import polars as pl`) for all new dataframe code. pandas remains only where an external interface requires it — do not introduce new pandas usage. Never round-trip polars → pandas → polars to run a Python loop; if a helper needs pandas input, rewrite the helper as a polars expression or join rather than converting around it.

- `.group_by()`, not pandas' `.groupby()`.
- There is no `.group_by().filter()`: aggregate first, then filter.
- Wrap column references in expressions with `pl.col("colname")`, and literals with `pl.lit(...)`.
- `None` is a missing value and `float("nan")` is NaN; they are distinct.
- No multi-indexing — use columns for grouping and sorting.
- Prefer native expressions over `.map_elements()`, which is slow and blocks query optimisation. When there is genuinely no expression equivalent, always pass `return_dtype`.
- Never walk a dataframe row by row. `.iterrows()`, `.itertuples()`, and `.apply(..., axis=1)` are the pandas equivalents of `.map_elements()` and are slower still — replace them with a vectorised expression, a join, or `dict(zip(df["a"], df["b"], strict=True))`. Likewise, reshape a flat per-cell array onto a 3D grid with `arr.reshape((nx, ny, nz), order="F")` and `np.linspace`, never nested `nx`/`ny`/`nz` loops.
- Prefer lazy evaluation: build `pl.LazyFrame` pipelines, take `pl.LazyFrame` in internal function signatures, and `.collect()` once at the end. Never call `.collect()` inside a loop over cells, timesteps, or ions — that re-runs the entire query every iteration. Collect once before the loop and index the resulting `pl.DataFrame`.
- Join with explicit `on=` (and `how=`) arguments; convert types with `.cast()`.
- Methods return new frames — there are no in-place operations, so use the return value.
- Compare frames in tests with `pltest.assert_frame_equal`.

## Command-line entry points

Modules that expose a subcommand follow one pattern:

```python
def addargs(parser: argparse.ArgumentParser) -> None: ...


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """One-line description used as the CLI help text."""
    args = at.parse_cli_args(addargs, __doc__, args, argsraw, kwargs)
```

Register new subcommands in `subcommandtree` in `artistools/commands.py` rather than adding a new console script. A module with the `addargs`/`main` shape that is not registered there is unreachable from the CLI and tends to rot unnoticed — register it or delete it, but do not extend it.

Keep data loading in importable functions separate from plotting, so both can be tested by calling `main(argsraw=[], **kwargs)`. `main` should be argument parsing plus a handful of calls, roughly under 50 lines. When editing a `main` that already inlines loading, physics, and plotting, extract the part you are touching rather than adding to it.

Build parsers and resolve paths with the shared helpers rather than hand-rolling them:

- `at.add_modelpath_arg(parser)` for a `-modelpath` or positional model-path argument.
- `at.normalize_path_list(args.modelpath)` to apply the default and coerce to `list[Path]`, instead of `if not args.modelpath: args.modelpath = Path()`.
- `at.get_timestep_times(modelpath, loc="mid")` rather than averaging the `start` and `end` arrays.
- `at.plottools.set_axis_properties` / `set_axis_labels` for standard axis and tick setup. `artistools/plottools.py` is reachable only as `at.plottools.*` — apart from `set_mpl_style` it is not re-exported at top level, so read it before writing new plot boilerplate.

## Tests

- Tests live beside the code they cover: `artistools/<area>/test_<area>.py`, with `test_*` functions. The `name-tests-test` hook enforces the `test_` prefix.
- Locate data with `at.get_path("testdata")` / `at.get_path("testoutput")`, or use pytest's `tmp_path`. Never write outside the repository's test output directory.
- Assert on numbers with `np.isclose`/`rtol`, not exact float equality; physics results shift slightly across platforms and library versions.
- To test plots without inspecting images, patch the axes method and check the call args:
  ```python
  @mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
  def test_something(mockplot: t.Any) -> None: ...
  ```
- Mark representative slow tests `@pytest.mark.benchmark`, or take a `benchmark: BenchmarkFixture` argument, so CodSpeed tracks them.
- Every bug fix gets a regression test. When you change a numerical result on purpose, update the expected value in the same commit and say why in the commit message.

## Performance and data handling

- `@lru_cache` is used on pure loaders keyed by path/arguments. Arguments must be hashable, and callers must not mutate what a cached function returned — copy first (see `artistools/misc/dirbins.py`).
- Cache parsed text files as parquet with `at.write_parquet_atomic`, which writes to a temporary file and renames, so an interrupted run cannot leave a corrupt cache.
- Use `at.parallel_map` instead of building multiprocessing pools directly; it picks threads or processes depending on whether the GIL is enabled.
- Read compressed ARTIS output through `at.zopen`/`at.zopenpl` rather than handling `.gz`/`.xz`/`.zst` yourself.
- Never grow a dataframe inside a loop: repeated `pd.concat([df, new])` or `.loc[key] = ...` row-appends are O(n²) in the iteration count. Accumulate into a list and concatenate once after the loop.
- Prefer the lazy whole-run scanners (`at.scan_estimators`) to the eager per-cell readers (`at.estimators.read_estimators`), which are very slow when many cells and timesteps are collected. Never call an eager loader inside a per-cell loop.

## Repository hygiene

- `pyproject.toml` is the single source of tool configuration; do not add per-tool config files or inline overrides that duplicate it.
- Add dependencies sparingly: runtime deps in `[project.dependencies]`, heavy optional ones in `[project.optional-dependencies].extras` with a lazy in-function import, tooling in `[dependency-groups].dev`. Run `uv lock` afterwards and commit `uv.lock`.
- Do not commit simulation output, generated plots, or files over 800 kB (blocked by pre-commit).
- When renaming or deleting, search the whole repository for the old name — including `artistools/commands.py`, the `__init__.py` re-exports, and tests — and update it all in one commit.
- Do not leave commented-out code behind when you replace or disable something — delete it, since git has the old version. Ruff's `commented-out-code` rule is deliberately both ignored and unfixable, so nothing catches this for you. The exception is a commented block that serves as documentation rather than as leftovers: a menu of alternative configuration entries showing what a setting accepts, or a published benchmark setup kept so it can be reproduced. Delete superseded code, keep worked examples — and if you are unsure which a block is, leave it and ask.
