"""Entry point: launch the Streamlit rPPG dashboard.

Usage:
    python run.py

This is a thin wrapper around ``streamlit run src/dashboard.py`` using the current
Python interpreter, so it works inside the project's virtual environment without
Streamlit needing to be on PATH.
"""

import subprocess
import sys
from pathlib import Path


def main() -> int:
    app = Path(__file__).resolve().parent / "src" / "dashboard.py"
    return subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(app), *sys.argv[1:]]
    )


if __name__ == "__main__":
    raise SystemExit(main())
