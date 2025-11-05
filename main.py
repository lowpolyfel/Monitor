"""Convenient launcher for the command-line monitor interface."""

from __future__ import annotations

import sys
from pathlib import Path


def _resolve_cli():
    try:
        from src.cli import main as cli_main  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parent
        src_dir = repo_root / "src"
        if src_dir.is_dir():
            sys.path.insert(0, str(src_dir))
        from src.cli import main as cli_main  # type: ignore[import-not-found]
    return cli_main


def main() -> None:
    """Entrypoint invoked by ``python main.py``."""

    cli_main = _resolve_cli()
    cli_main()


if __name__ == "__main__":
    main()
