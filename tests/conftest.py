"""Pytest config — point Chroma at a per-session temp dir so importing the
server module doesn't try to dial the production HTTP daemon."""
import os
import tempfile
from pathlib import Path

# Set before any context_orchestrator import. Each test that needs isolation
# overrides server.vs / server.db with its own tmp_path-scoped instance, so
# this only matters for module-level instantiation in server.py.
os.environ.setdefault(
    "CO_CHROMA_PATH",
    str(Path(tempfile.mkdtemp(prefix="co-tests-chroma-")))
)


import pytest


@pytest.fixture(autouse=True)
def _no_real_gemini_key(monkeypatch, tmp_path):
    """Keep the suite hermetic: never auto-detect the developer's real Gemini
    key (~/.config/google/key or env) and call the live API. Without this,
    tests passed on CI and failed on any machine with a key — whenever the
    API errored or hit quota. Tests that want Gemini patch it back in."""
    from context_orchestrator import search
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(search, "GEMINI_KEY_FILE", tmp_path / "no-gemini-key")
