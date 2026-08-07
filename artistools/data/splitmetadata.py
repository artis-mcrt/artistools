"""Split a combined metadata.yml into one .meta.yml file beside each observation file it describes."""

from pathlib import Path

import yaml


def main() -> None:
    """Write a separate .meta.yml file for every entry in metadata.yml."""
    with Path("metadata.yml").open("r", encoding="utf-8") as yamlfile:
        metadata = yaml.safe_load(yamlfile)

    for obsfile in metadata:
        metafilepath = Path(obsfile).with_suffix(f"{Path(obsfile).suffix}.meta.yml")
        with metafilepath.open("w", encoding="utf-8") as metafile:
            yaml.dump(metadata[obsfile], metafile)


if __name__ == "__main__":
    main()
