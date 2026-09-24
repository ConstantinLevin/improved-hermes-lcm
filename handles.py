"""Handles for the plugin's record (#29, W5).

A handle is a kind letter and eight characters of base32 from 40 random bits. It
is never derived from content and never reissued by construction, so a handle from
a replaced or restored store resolves to nothing instead of to other content. The
caller draws again when the handle is already taken in its table.
"""

from __future__ import annotations

import base64
import secrets

# The kind letters in use. ``n`` names a plugin session.
SESSION = "n"
_KINDS = frozenset({SESSION})


def new_handle(kind: str) -> str:
    if kind not in _KINDS:
        raise ValueError(f"unknown handle kind {kind!r}")
    return kind + base64.b32encode(secrets.token_bytes(5)).decode("ascii").lower()
