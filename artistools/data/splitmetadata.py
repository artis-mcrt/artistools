#!/usr/bin/env python3
from pathlib import Path

import yaml


def main() -> None:
    with Path("metadata.yml").open("r") as yamlfile:
        metadata = yaml.load(yamlfile, Loader=yaml.FullLoader)

    for obsfile in metadata:
        metafilepath = Path(obsfile).with_suffix(f"{Path(obsfile).suffix}.meta.yml")
        with metafilepath.open("w") as metafile:
            yaml.dump(metadata[obsfile], metafile)


if __name__ == "__main__":
    main()
