"""Split a combined metadata.yml into one .meta.yml file beside each observation file it describes."""

import argparse
import typing as t
from collections.abc import Sequence
from pathlib import Path

import yaml

import artistools as at


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    parser.add_argument(
        "-metadatafile", type=Path, default=Path("metadata.yml"), help="Path to the combined metadata.yml file"
    )
    at.add_outputpath_arg(parser, default=None, astype=Path, helptext="Folder for the .meta.yml files")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Write a separate .meta.yml file for every entry in a combined metadata.yml."""
    args = at.parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    with Path(args.metadatafile).open(encoding="utf-8") as yamlfile:
        metadata = yaml.safe_load(yamlfile)

    for obsfile in metadata:
        metafilepath = Path(obsfile).with_suffix(f"{Path(obsfile).suffix}.meta.yml")
        if args.outputpath is not None:
            metafilepath = Path(args.outputpath) / metafilepath.name
        with metafilepath.open("w", encoding="utf-8") as metafile:
            yaml.dump(metadata[obsfile], metafile)
        at.print_saved(metafilepath)


if __name__ == "__main__":
    main()
