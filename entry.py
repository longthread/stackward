"""Freeze target for PyInstaller.

Deliberately separate from `stackward/__main__.py`, which exists for
`python -m stackward` and uses a relative import. PyInstaller runs its entry
script as a top-level module with no package context, so a relative import
there fails at runtime with "attempted relative import with no known parent
package" — a bundle that builds perfectly and then cannot start.

Absolute import, one line, no logic: the entry script is not the place to
discover packaging problems.
"""

from stackward.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
