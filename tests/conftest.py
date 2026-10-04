"""Pytest config — point Chroma at a per-session temp dir so importing the
server module doesn't try to dial the production HTTP daemon."""
import os
import tempfile
from pathlib import Path

# A scratch HOME for the whole run, set before anything computes
# Path.home(): no test may read or write the developer's real
# ~/.context-orchestrator, ~/.claude or ~/.config. Only chromadb's downloaded
# ONNX model (~80 MB, read-only use) is shared, so it isn't fetched every run.
_REAL_HOME = Path.home()
_SCRATCH_HOME = Path(tempfile.mkdtemp(prefix="co-tests-home-"))
_model_cache = _REAL_HOME / ".cache" / "chroma"
if _model_cache.is_dir():
    (_SCRATCH_HOME / ".cache").mkdir()
    (_SCRATCH_HOME / ".cache" / "chroma").symlink_to(_model_cache)
os.environ["HOME"] = str(_SCRATCH_HOME)
os.environ.pop("CLAUDE_CONFIG_DIR", None)

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
    # Same for the embedding choice: code under test may set it (e.g.
    # `contorch-memory embeddings none`); setenv-then-delenv makes monkeypatch
    # restore the original state afterwards, so it can't leak between tests.
    monkeypatch.setenv("CO_EMBEDDING_MODEL", "")
    monkeypatch.delenv("CO_EMBEDDING_MODEL")
    # Backups (written before any delete) must never land in the real
    # ~/.context-orchestrator/backups from a test.
    from context_orchestrator import transcripts
    monkeypatch.setattr(transcripts, "BACKUP_DIR", tmp_path / "backups")
    # Nothing may resolve to the real ~/.context-orchestrator/chroma (HTTP mode
    # keeps its collection-name map there).
    monkeypatch.setattr(search, "DEFAULT_CHROMA_PATH", tmp_path / "default-chroma")
