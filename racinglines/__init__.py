"""racinglines: fair prices and market making for race sports. See `racinglines -h`."""

import sys

if sys.version_info < (3, 11):
    raise SystemExit(
        f"racinglines needs Python 3.11+ (this is {sys.version.split()[0]} at {sys.executable}).\n"
        "Set up the project environment once:\n"
        "    python3.14 -m venv .venv && source .venv/bin/activate\n"
        "    pip install -r requirements.txt && pip install -e .")
