"""Run the original Robot-ICL baseline from the source-only package."""

from pathlib import Path

import train
from prepare import read_json


class PackagedRoboICLConfig(train.RoboICLConfig):
    def code_resources(self):
        # The RoboICL plan supplies training data. These hashes preserve the
        # original checkpoint identity without copying task descriptions.
        return read_json(Path(__file__).with_name("backend_source_resources.json"))


if __name__ == "__main__":
    train.RoboICLConfig = PackagedRoboICLConfig
    train.main()
