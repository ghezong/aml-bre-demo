"""Launch the AML Streamlit web UI."""

import os
import subprocess
import sys


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(current_dir, "ui", "streamlit_app.py")

    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("Streamlit is not installed. Install with: pip install streamlit>=1.25.0")
        return

    print("Starting AML web UI...")
    subprocess.run([
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app_path,
        "--server.headless",
        "false",
        "--server.enableCORS",
        "false",
        "--server.enableXsrfProtection",
        "false",
    ])


if __name__ == "__main__":
    main()
