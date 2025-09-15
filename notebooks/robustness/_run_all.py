#!/usr/bin/env python
"""Execute every script and notebook in this directory.

Python scripts are run with ``python`` and Jupyter notebooks are executed
in-place using ``jupyter nbconvert``.
"""

import subprocess
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
THIS_FILE = Path(__file__).name

for path in sorted(DIRECTORY.iterdir()):
    if path.name == THIS_FILE or not path.is_file():
        continue
    if path.suffix == ".py":
        subprocess.run(["python", str(path)], check=True)
    elif path.suffix == ".ipynb":
        subprocess.run(
            [
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--inplace",
                str(path),
            ],
            check=True,
        )