"""Test session setup — keep the suite hermetic and offline.

The suite must never emit LangSmith traces or hit a real LLM / web API, even
when the shell that launched pytest has keys exported (e.g. after ``source
.env``). Force offline providers and drop keys for the whole session;
individual tests that exercise provider-selection logic use ``monkeypatch`` and
restore afterwards.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def _hermetic_env():
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LLM_PROVIDER"] = "mock"
    os.environ["WEB_PROVIDER"] = "stub"
    for key in (
        "LANGSMITH_API_KEY",
        "TAVILY_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
    ):
        os.environ.pop(key, None)
    yield
