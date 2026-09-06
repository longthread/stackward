"""Source that is shipped to be *run by another interpreter*, not imported.

`stackward` may run as a frozen binary carrying its own interpreter, which
cannot import the consuming repository's classes — so the one step that must
import them, generating the declared-secrets artifact, runs under the
repository's own interpreter instead (`[python]` in `.stackward.toml`). That
step needs pydantic; `stackward` itself must never import or bundle it.

The two requirements together are what this package exists for. `regen.py` is
a real, standalone module — readable, lintable, and directly runnable — but
`stackward` only ever reads it as **text** and pipes it to the repository's
interpreter. Nothing in `stackward` imports it, so PyInstaller's analysis
never follows it, and pydantic never enters the bundle.

Because it is data rather than an imported module, the frozen build has to
carry it explicitly: `.github/workflows/release.yml` passes
`--add-data src/stackward/bootstrap/regen.py:stackward/bootstrap`, and
`generator_source` reads it back through `importlib.resources`, which
resolves both an ordinary installation and a PyInstaller bundle. That
round trip was verified against a real frozen bundle rather than assumed.
"""

from __future__ import annotations

import importlib.resources

GENERATOR_NAME = "regen.py"


def generator_source() -> str:
    """The generator program's text, for the repository's interpreter.

    The single accessor for it, used by `commands.sync_declared` and by the
    tests alike — a test that read the file by path instead would assert
    nothing about the frozen build, where that path does not exist.
    """
    resource = importlib.resources.files(__package__).joinpath(GENERATOR_NAME)
    return resource.read_text(encoding="utf-8")
