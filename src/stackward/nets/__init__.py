"""The credential-detection nets: independent ways to answer "is this a
plaintext credential?" over a parsed stack config document.

`heuristic` is the key-name/parent-name net and ships unconditionally. A
model net (walking a repository's own pydantic models, via `[check]`'s
`model_net`/`stack_models`) is a separate, later addition — this package is
named for the plural on purpose.
"""

from __future__ import annotations
