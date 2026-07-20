"""`.env` loader behavior."""

from __future__ import annotations

import os

from langconnect_agent.env import load_env


def test_load_env_reads_file(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text("WAYFINDER_TEST_VAR=hello\n")
    monkeypatch.delenv("WAYFINDER_TEST_VAR", raising=False)

    assert load_env(str(p)) is True
    assert os.environ["WAYFINDER_TEST_VAR"] == "hello"


def test_load_env_does_not_override_shell_by_default(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text("WAYFINDER_TEST_VAR=fromfile\n")
    monkeypatch.setenv("WAYFINDER_TEST_VAR", "fromshell")

    load_env(str(p))
    assert os.environ["WAYFINDER_TEST_VAR"] == "fromshell"


def test_load_env_override_true(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text("WAYFINDER_TEST_VAR=fromfile\n")
    monkeypatch.setenv("WAYFINDER_TEST_VAR", "fromshell")

    load_env(str(p), override=True)
    assert os.environ["WAYFINDER_TEST_VAR"] == "fromfile"


def test_load_env_missing_file_returns_false(tmp_path):
    assert load_env(str(tmp_path / "does-not-exist.env")) is False
