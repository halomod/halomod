#!/usr/bin/python
"""Auto-update the CHANGELOG for a new version."""

import sys
from datetime import datetime
from pathlib import Path

if __name__ == "__main__":
    newversion = sys.argv[1]

    with Path("CHANGELOG.rst").open() as fl:
        lines = fl.readlines()

    for _i, line in enumerate(lines):
        if line == "dev-version\n":
            break
    else:
        raise OSError("Couldn't Find 'dev-version' tag")

    lines.insert(_i + 2, "----------------------\n")
    now = datetime.now(tz=datetime.tzinfo).strftime("%d %b %Y")
    lines.insert(_i + 2, f"v{newversion} [{now}]\n")
    lines.insert(_i + 2, "\n")

    with Path("CHANGELOG.rst").open("w") as fl:
        fl.writelines(lines)
