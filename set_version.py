#!/usr/bin/env python3
"""Set the package version and release date to today, across pyproject.toml and the other files that carry them."""

import argparse
import datetime as dt
import re
import subprocess
from pathlib import Path

repo_path = Path(__file__).parent


def replace_line(path: Path, pattern: str, replacement: str) -> None:
    """Replace the single line of a text file matching pattern, raising unless it matched exactly once."""
    text_in = path.read_text(encoding="utf-8")
    text_out, subcount = re.subn(pattern, replacement, text_in, flags=re.MULTILINE)

    if subcount != 1:
        msg = f"Expected one line matching {pattern!r} in {path}, but found {subcount}"
        raise RuntimeError(msg)

    path.write_text(text_out, encoding="utf-8")


def main() -> None:
    """Write today's date as the version and release date wherever they appear in the repository."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-revision",
        type=int,
        default=0,
        help="Number of the release on this date. Use 1 or more for a second release on the same date.",
    )
    args = parser.parse_args()

    if args.revision < 0:
        msg = f"revision must be zero or more, but it is {args.revision}"
        raise ValueError(msg)

    today = dt.datetime.now(dt.UTC).date()
    date_released = today.isoformat()
    # version has no zero padding and '.' for separator, e.g. 2026.4.20
    version = f"{today.year}.{today.month}.{today.day}"
    # PEP 440 permits a fourth release segment, thus a second release on one date is possible.
    # Cargo accepts only three segments, thus rust/Cargo.toml keeps the date on its own. Nothing
    # reads the version of the crate: maturin takes the version of the wheel from pyproject.toml,
    # and the crate is a cdylib that does not go to crates.io
    version_py = f"{version}.{args.revision}" if args.revision else version
    print(f"Setting version to: {version_py}")
    print(f"Date released: {date_released}")

    # a stale lockfile would otherwise join the version bump in one commit, and the workflows that
    # set UV_FROZEN would then use a lockfile that no test run has seen
    subprocess.check_call(["uv", "lock", "--check"], cwd=repo_path)

    replace_line(repo_path / "pyproject.toml", r"^version = .*$", f'version = "{version_py}"')

    # dependency versions sit inside inline tables, so only the [package] version starts a line
    rust_path = repo_path / "rust"
    replace_line(rust_path / "Cargo.toml", r"^version = .*$", f'version = "{version}"')

    # --workspace syncs Cargo.lock with the new package version, leaving the dependencies alone
    subprocess.check_call(["cargo", "update", "--workspace"], cwd=rust_path)

    replace_line(repo_path / "CITATION.cff", r"^version: .*$", f"version: {version_py}")
    replace_line(repo_path / "CITATION.cff", r"^date-released: .*$", f"date-released: {date_released}")

    subprocess.check_call(["uv", "lock"], cwd=repo_path)


if __name__ == "__main__":
    main()
