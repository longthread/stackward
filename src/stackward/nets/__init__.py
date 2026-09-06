"""The credential-detection nets: independent ways to answer "is this a
plaintext credential?" over a parsed stack config document.

`heuristic` is the key-name/parent-name net and ships unconditionally: it
needs nothing from the repository beyond `[check]` policy. `model` is the
declaration-driven net, built from a repository's own pydantic marks and
carried across the frozen-binary boundary as a committed graph (`[check]`'s
`model_net`/`stack_models`); it runs only when that repository declares it.

The two are **unioned**, never ranked — a leaf either can name is a finding —
and they share one leaf decision (`heuristic.is_encrypted`,
`heuristic.is_empty`) so they cannot drift into disagreeing about whether the
same value is a credential. Both take an already-parsed document and do no
I/O, which is what lets one caller scan a file on disk and another scan a
staged git blob without either round-tripping content through a temp file.
"""

from __future__ import annotations
