"""Entry point for `python -m stackward`, and the target PyInstaller freezes."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
