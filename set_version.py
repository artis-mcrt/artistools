#!/usr/bin/env python3
import datetime as dt
import re
import subprocess
from pathlib import Path

repo_path = Path(__file__).parent


def update_pyproject_version(version: str) -> None:
    pyproject_path = repo_path / "pyproject.toml"
    pyproject_text_in = pyproject_path.read_text(encoding="utf-8")

    pyproject_text_out = re.sub(r"^version = .*$", f'version = "{version}"', pyproject_text_in, flags=re.MULTILINE)

    pyproject_path.write_text(pyproject_text_out, encoding="utf-8")


def update_cff_citation(version: str, date_released: str) -> None:
    cff_path = repo_path / "CITATION.cff"
    cff_in = cff_path.read_text(encoding="utf-8")

    cff_out = re.sub(r"^version: .*$", f"version: {version}", cff_in, flags=re.MULTILINE)
    cff_out = re.sub(r"^date-released: .*$", f"date-released: {date_released}", cff_out, flags=re.MULTILINE)

    cff_path.write_text(cff_out, encoding="utf-8")


def main() -> None:
    # get YYYY-MM-DD (no zero padding) from current date in UTC
    date_released = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d")
    # get version with no zero padding and '.' for separator, e.g. 2026.4.20
    version = ".".join(f"{int(part):d}" for part in date_released.split("-"))
    print(f"Setting version to: {version}")
    print(f"Date released: {date_released}")

    update_pyproject_version(version)

    # Update Cargo.toml
    cargo_toml_path = repo_path / "rust" / "Cargo.toml"
    subprocess.check_call(["cargo", "install", "-q", "cargo-edit"])
    update_command = ["cargo", "set-version", f"--manifest-path={cargo_toml_path}", version]
    try:
        subprocess.check_call(update_command)  # noqa: S603
    except subprocess.CalledProcessError as e:
        msg = f"Failed:\n{' '.join(update_command)}"
        raise RuntimeError(msg) from e

    update_cff_citation(version=version, date_released=date_released)

    (repo_path / "artistools" / "version.py").write_text(f'version = "{version}"\n', encoding="utf-8")


if __name__ == "__main__":
    main()
