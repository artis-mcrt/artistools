#!/usr/bin/env python3
"""Set the package version and release date to today, across pyproject.toml and the other files that carry them."""

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
    today = dt.datetime.now(dt.UTC).date()
    date_released = today.isoformat()
    # version has no zero padding and '.' for separator, e.g. 2026.4.20
    version = f"{today.year}.{today.month}.{today.day}"
    print(f"Setting version to: {version}")
    print(f"Date released: {date_released}")

    replace_line(repo_path / "pyproject.toml", r"^version = .*$", f'version = "{version}"')

    # dependency versions sit inside inline tables, so only the [package] version starts a line
    rust_path = repo_path / "rust"
    replace_line(rust_path / "Cargo.toml", r"^version = .*$", f'version = "{version}"')

    # --workspace syncs Cargo.lock with the new package version, leaving the dependencies alone
    subprocess.check_call(["cargo", "update", "--workspace"], cwd=rust_path)

    replace_line(repo_path / "CITATION.cff", r"^version: .*$", f"version: {version}")
    replace_line(repo_path / "CITATION.cff", r"^date-released: .*$", f"date-released: {date_released}")

    # version.py is regenerated in full, so it must carry its own docstring or ruff's D100 fails after a release
    (repo_path / "artistools" / "version.py").write_text(
        f'"""Package version, updated by set_version.py at release time."""\n\nversion = "{version}"\n',
        encoding="utf-8",
    )

    subprocess.check_call(["uv", "lock"], cwd=repo_path)


if __name__ == "__main__":
    main()
