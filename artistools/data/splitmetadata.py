#!/usr/bin/env python3
from pathlib import Path

import yaml


def main() -> None:
    with Path("metadata.yml").open("r", encoding="utf-8") as yamlfile:
        metadata = yaml.safe_load(yamlfile)

    for obsfile in metadata:
        metafilepath = Path(obsfile).with_suffix(f"{Path(obsfile).suffix}.meta.yml")
        with metafilepath.open("w", encoding="utf-8") as metafile:
            yaml.dump(metadata[obsfile], metafile)


if __name__ == "__main__":
    main()
