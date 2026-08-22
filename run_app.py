"""Run the public AML portfolio application."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    app = ROOT / "src" / "ui" / "streamlit_app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app)])


if __name__ == "__main__":
    main()
