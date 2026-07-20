"""Centralized ``.env`` loading for local runs.

Keys accumulate (LangSmith, Anthropic, OpenAI, Tavily, pgvector, ...), so they
live in one gitignored ``.env`` at the project root. ``load_env()`` reads it
into ``os.environ`` so ``config.get_config()`` and the LLM/LangSmith SDKs pick
everything up automatically.

Design choice: **not** auto-loaded on package import. Loading is an entry-point
concern — the CLI scripts (``scripts/*.py``) call ``load_env()`` at startup;
``langgraph dev`` loads ``.env`` itself via ``langgraph.json``. Keeping it out
of import means the test suite stays hermetic/offline (it never loads real
keys) and importing the library has no hidden side effects.

``override=False`` by default, so an already-exported shell variable wins over
the file — handy for one-off overrides and CI.
"""

from __future__ import annotations

import os
from typing import Optional

_LOADED = False


def load_env(path: Optional[str] = None, *, override: bool = False) -> bool:
    """Load the project ``.env`` into ``os.environ`` (idempotent).

    Returns True if a ``.env`` file was found and loaded, else False. No-ops
    (returns False) if python-dotenv is not installed, so core stays importable.
    """
    global _LOADED
    if _LOADED and path is None:
        return True
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        return False

    dotenv_path = path or find_dotenv(usecwd=True)
    if not dotenv_path or not os.path.exists(dotenv_path):
        return False

    load_dotenv(dotenv_path, override=override)
    if path is None:
        _LOADED = True
    return True
