#!/usr/bin/env python3
import re
import subprocess
from pathlib import Path

repo_path = Path(__file__).parent


def update_pyproject_version(version: str) -> None:
    pyproject_path = repo_path / "pyproject.toml"
    pyproject_text_in = pyproject_path.read_text(encoding="utf-8")

    pyproject_text_out = re.sub(r"^version = .*$", f'version = "{version}"', pyproject_text_in, flags=re.MULTILINE)

    pyproject_path.write_text(pyproject_text_out, encoding="utf-8")


def update_cff_citation(version: str) -> None:

    # Convert e.g. "v2026.1.2" to "2026-01-02"
    date_released = "-".join(part.zfill(2) for part in version.lstrip("v").split("."))
    cff_in = (repo_path / "CITATION.cff").read_text(encoding="utf-8")

    cff_out = re.sub(r"^version: .*$", f"version: {version}", cff_in, flags=re.MULTILINE)
    cff_out = re.sub(r"^date-released: .*$", f"date-released: {date_released}", cff_out, flags=re.MULTILINE)

    Path("CITATION.cff").write_text(cff_out, encoding="utf-8")


def main() -> None:
    version = subprocess.check_output(["git", "describe", "--tags", "--abbrev=0"], text=True).strip().removeprefix("v")
    assert "." in version, "Version must contain a dot"
    print(version)

    update_pyproject_version(version)

    # Update Cargo.toml
    cargo_toml_path = repo_path / "rust" / "Cargo.toml"
    # this can fail because the larger cargo-edit crate is already installed to supply the same command, which is fine
    subprocess.run(["cargo", "install", "-q", "cargo-set-version"], shell=False, capture_output=True, check=False)
    update_command = ["cargo", "set-version", f"--manifest-path={cargo_toml_path}", version]
    try:
        subprocess.check_call(update_command)  # noqa: S603
    except subprocess.CalledProcessError as e:
        msg = f"Failed:\n{' '.join(update_command)}"
        raise RuntimeError(msg) from e

    update_cff_citation(version.removesuffix("+dev"))

    (repo_path / "artistools" / "version.py").write_text(f'version = "{version}"\n', encoding="utf-8")


if __name__ == "__main__":
    main()
