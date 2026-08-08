"""Console entry point for launching the packaged Streamlit application."""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path


def _app_path() -> Path:
    """Find ``streamlit_app.py`` in a wheel install or a source checkout."""

    try:
        installed = distribution("sol-fable")
    except PackageNotFoundError:
        installed = None
    if installed is not None:
        for item in installed.files or ():
            if item.name == "streamlit_app.py":
                candidate = Path(installed.locate_file(item)).resolve()
                if candidate.is_file():
                    return candidate

    source_app = Path(__file__).resolve().parents[2] / "streamlit_app.py"
    if source_app.is_file():
        return source_app
    raise RuntimeError("The Sol-Fable Streamlit application is missing from this installation.")


def main() -> int:
    """Launch Streamlit with remaining command-line options passed through."""

    try:
        from streamlit.web import cli as streamlit_cli
    except ImportError as exc:  # pragma: no cover - depends on optional installation
        raise SystemExit("Install the UI extra first: pip install 'sol-fable[ui]'") from exc

    sys.argv = ["streamlit", "run", str(_app_path()), *sys.argv[1:]]
    result = streamlit_cli.main()
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())
