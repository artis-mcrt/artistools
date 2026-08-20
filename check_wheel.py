#!/usr/bin/env python3
"""Check that an installed artistools wheel contains a rustext extension that loads.

cibuildwheel runs this as its test command. A full run of the command-line interface would prove
more, but it needs every run-time dependency, and the installation of those on Linux arm64 takes
about 20 minutes. This check loads the extension module directly from its file, thus it imports
neither the artistools package nor polars, and it needs nothing outside the wheel. It still fails
if the wheel omits the extension or if the extension has an incompatible ABI.
"""

import importlib.machinery
import importlib.util
import sys
from pathlib import Path


def main() -> None:
    """Load artistools.rustext from the installed wheel and raise an error if it fails."""
    # Python puts the directory of this script on sys.path, and cibuildwheel runs the script from
    # the project root. The source tree there shadows the installed wheel, and the source tree has
    # no compiled extension. Remove that entry, so that find_spec resolves to site-packages.
    scriptdir = Path(__file__).resolve().parent
    sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != scriptdir]

    pkgspec = importlib.util.find_spec("artistools")
    if pkgspec is None or pkgspec.submodule_search_locations is None:
        msg = "Found no installed artistools package"
        raise RuntimeError(msg)

    pkgpath = Path(next(iter(pkgspec.submodule_search_locations)))
    # find_spec("artistools.rustext") would import the parent package first, and that needs polars
    extpaths = sorted(
        path for suffix in importlib.machinery.EXTENSION_SUFFIXES for path in pkgpath.glob(f"rustext{suffix}")
    )
    if not extpaths:
        contents = sorted(path.name for path in pkgpath.iterdir())
        msg = f"Found no rustext extension in {pkgpath}, which contains: {contents}"
        raise RuntimeError(msg)

    extspec = importlib.util.spec_from_file_location("rustext", extpaths[0])
    if extspec is None or extspec.loader is None:
        msg = f"Could not make a module spec for {extpaths[0]}"
        raise RuntimeError(msg)

    extspec.loader.exec_module(importlib.util.module_from_spec(extspec))

    print(f"Loaded {extpaths[0]} on Python {sys.version}")


if __name__ == "__main__":
    main()
