#!/usr/bin/env python
"""Execute every script and notebook in this directory quietly.

Print which file is being run. Suppress tqdm bars and warnings.
"""

import os
import subprocess
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
THIS_FILE = Path(__file__).name

ENV = os.environ.copy()
ENV.update({
    "PYTHONWARNINGS": "ignore",  # suppress DeprecationWarning etc.
    "TQDM_DISABLE": "1",         # disable tqdm progress bars
})

QUIET = {
    "stdout": subprocess.DEVNULL,
    "stderr": subprocess.DEVNULL,
}

for path in sorted(DIRECTORY.iterdir()):
    if path.name == THIS_FILE or not path.is_file():
        continue

    if path.suffix == ".py":
        print(f"Running script: {path.name}", flush=True)
        subprocess.run(["python", str(path)], check=True, env=ENV, **QUIET)

    elif path.suffix == ".ipynb":
        print(f"Executing notebook: {path.name}", flush=True)
        subprocess.run(
            [
                "jupyter",
                "nbconvert",
                "--to", "notebook",
                "--execute",
                "--inplace",
                "--log-level=ERROR",
                str(path),
            ],
            check=True,
            env=ENV,
            **QUIET,
        )
