"""Run the commands of examples.md on the test model and write the outputs to docs/examples.

Read the Docs runs this script before each build. Run it before a local build, because the page needs these files.
It needs the folder tests/data/testmodel, which tests/data/setuptestdata.sh makes.
"""

import subprocess  # ruff:ignore[suspicious-subprocess-import]
import sys
import tempfile
from pathlib import Path

DOCS_FOLDER = Path(__file__).resolve().parent
OUTPUT_FOLDER = DOCS_FOLDER / "examples"
MODEL_FOLDER = (DOCS_FOLDER.parent / "tests" / "data" / "testmodel").resolve()

PLOT_COMMANDS = {
    "plotspectra.png": ["plotspectra", ".", "-t", "300"],
    "plotlightcurves.png": ["plotlightcurves", "."],
    "plotestimators.png": ["plotestimators", "Te", "TR", ".", "-t", "300"],
    "plotnltepops.png": ["plotnltepops", "--modelpath", ".", "-t", "300", "--modelgridindex", "0"],
}

TEXT_COMMANDS = {
    "describe.txt": ["describe", "."],
    "timesteps.txt": ["timesteps", "."],
    "plotestimators_print_data.txt": ["plotestimators", "Te", ".", "-t", "300", "--print-data"],
}


def run_artistools(arguments: list[str]) -> str:
    """Run one command in the model folder of the test model and return its standard output."""
    return subprocess.run(  # ruff:ignore[subprocess-without-shell-equals-true]
        [sys.executable, "-m", "artistools", *arguments],
        cwd=MODEL_FOLDER,
        stdout=subprocess.PIPE,
        encoding="utf-8",
        check=True,
    ).stdout


def main() -> None:
    """Write each output file."""
    OUTPUT_FOLDER.mkdir(exist_ok=True)
    for filename, arguments in PLOT_COMMANDS.items():
        run_artistools([*arguments, "-o", str(OUTPUT_FOLDER / filename)])

    # a plot command with --print-data also makes a figure, and the model folder must stay as it was
    with tempfile.TemporaryDirectory() as scratchfolder:
        for filename, arguments in TEXT_COMMANDS.items():
            isplot = arguments[0].startswith("plot")
            text = run_artistools([*arguments, "-o", f"{scratchfolder}/figure.png"] if isplot else arguments)
            # the page must show no path of the computer that ran the script
            (OUTPUT_FOLDER / filename).write_text(
                text.replace(str(MODEL_FOLDER), "/path/to/testmodel"), encoding="utf-8"
            )


if __name__ == "__main__":
    main()
