"""Tests for the drop / auto-task / auto-detect-type behavior added 2026-04-26."""
import os
import tempfile
import time
from datetime import date
from pathlib import Path

import pytest

from context_orchestrator import server


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch, tmp_path):
    """Point the server at fresh DB / Chroma / transcripts dir per test."""
    from context_orchestrator.db import Database
    from context_orchestrator.search import VectorSearch

    monkeypatch.setattr(server, "db", Database(db_path=tmp_path / "ctx.db"))
    monkeypatch.setattr(server, "vs", VectorSearch(chroma_path=tmp_path / "chroma"))


def test_detect_source_type_url():
    assert server._detect_source_type("https://example.com/foo") == "url"
    assert server._detect_source_type("http://example.com") == "url"


def test_detect_source_type_file(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("hi")
    assert server._detect_source_type(str(f)) == "file"


def test_detect_source_type_text(tmp_path):
    assert server._detect_source_type("not a file or url, just words") == "text"
    assert server._detect_source_type(str(tmp_path / "nonexistent.txt")) == "text"


def test_resolve_default_task_falls_back_to_inbox(tmp_path):
    # No transcripts dir → inbox-{today}
    assert server._resolve_default_task_name() == f"inbox-{date.today().isoformat()}"


def test_resolve_default_task_uses_recent_meeting():
    # Appended just now (meeting-capture writes straight to the table).
    server.db.put_transcript("meeting-2026-04-26T18-43-05", "# Meeting transcript\n[18:43:05] hello\n")
    assert server._resolve_default_task_name() == "meeting-2026-04-26T18-43-05"


def test_resolve_default_task_ignores_old_meeting():
    server.db.put_transcript("meeting-2025-01-01T00-00-00", "ancient", now=time.time() - 7200)
    assert server._resolve_default_task_name() == f"inbox-{date.today().isoformat()}"


def test_drop_text_no_task():
    result = server.drop("just some text I want saved")
    assert "auto-task" in result or "created task" in result
    assert f"inbox-{date.today().isoformat()}" in result


def test_drop_url_auto_detects_type():
    result = server.drop("https://example.com/article")
    assert "url source" in result
    assert "https://example.com/article" in result


def test_drop_file_auto_detects_type(tmp_path):
    f = tmp_path / "design.md"
    f.write_text("# Design doc\nThe widget should foo when bar.")
    result = server.drop(str(f))
    assert "file source" in result
    assert str(f) in result


def test_drop_attaches_to_recent_meeting():
    server.db.put_transcript("meeting-2026-04-26T18-43-05", "# Meeting transcript\n[18:43:05] hello\n")

    result = server.drop("https://example.com/q3-design", notes="presenter Sarah")
    assert "meeting-2026-04-26T18-43-05" in result
    assert "presenter Sarah" in result


def test_add_source_still_works_with_explicit_task():
    server.create_task("explicit-task", description="test")
    result = server.add_source(
        task_name="explicit-task",
        source_type="text",
        reference="some content",
        notes="manual",
    )
    assert "Added text source to task 'explicit-task'" in result
    # No auto-task note when task was explicit
    assert "auto-task" not in result and "created task" not in result
