"""M1.8 `contorch-transcripts import --json` and M1.9 the env-file writer."""
import io
import json
import os
import subprocess
import sys
from pathlib import Path

from context_orchestrator import search
from context_orchestrator import transcripts as T
from context_orchestrator.db import Database
from context_orchestrator.settings import set_env_value

SRC = Path(__file__).resolve().parent.parent / "src"

VTT = """WEBVTT

00:00:05.000 --> 00:00:09.000
<v Jane Doe>We doubled revenue after moving to annual pricing.
"""


class _GeminiLikeEF:
    """3072-d vectors under Gemini's identity, without any API."""
    def name(self):
        return "gemini-gemini-embedding-001"

    def embed_documents(self, input):
        return [[0.001 * ((i + len(t)) % 7) for i in range(3072)] for t in input]

    __call__ = embed_documents


def _lines(out):
    return [json.loads(l) for l in out.splitlines() if l.strip()]


def test_3072d_bundle_into_a_minilm_index_is_keyword_only_no_crash(tmp_path, monkeypatch, capfd):
    src = Database(db_path=tmp_path / "other.db")
    mid, _ = T.add_text(src, VTT, title="Jane", started_at="2026-09-28T14:00:00")
    bundle = tmp_path / "bundle.jsonl"
    with bundle.open("w") as f:
        T.write_bundle([src.get_transcript(mid)], f, _GeminiLikeEF())
    monkeypatch.setenv("CO_DB_PATH", str(tmp_path / "here.db"))
    monkeypatch.setenv("CO_CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("CO_EMBEDDING_MODEL", "local")        # MiniLM, 384-d
    here = search.VectorSearch(chroma_path=tmp_path / "chroma")
    here.upsert(["x"], ["an existing 384-d document"], [{"type": "t"}])   # the index already has 384-d vectors
    assert T.main(["import", str(bundle), "--json"]) == 0
    events = _lines(capfd.readouterr().out)
    assert [e["event"] for e in events] == ["start", "stored", "result"]
    r = events[-1]
    assert r["schema"] == "contorch-transcripts.import/1" and r["ok"]
    assert r["embeddings"] == "keyword_only" and r["imported"] == 1 and r["vectors_loaded"] == 0
    assert not r["compatible"]
    db = Database(db_path=tmp_path / "here.db")
    assert db.search_text("annual pricing")[0]["meeting_id"] == mid, "full-text finds it at once"


def test_compatible_bundle_is_imported(tmp_path, monkeypatch, capfd):
    class EF:
        def name(self):
            return "fake-ef"

        def embed_documents(self, input):
            return [[float((hash(t) >> i) & 1) for i in range(8)] for t in input]
        __call__ = embed_documents
    src = Database(db_path=tmp_path / "other.db")
    mid, _ = T.add_text(src, VTT, title="Jane", started_at="2026-09-28T14:00:00")
    bundle = tmp_path / "b.jsonl"
    with bundle.open("w") as f:
        T.write_bundle([src.get_transcript(mid)], f, EF())
    monkeypatch.setenv("CO_DB_PATH", str(tmp_path / "here.db"))
    monkeypatch.setenv("CO_CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr(search, "_build_embedding_function", lambda: EF())
    assert T.main(["import", str(bundle), "--json"]) == 0
    r = _lines(capfd.readouterr().out)[-1]
    assert r["ok"] and r["embeddings"] == "imported" and r["vectors_loaded"] == 1


def test_files_import_json_and_text_mode_unchanged(tmp_path, monkeypatch, capfd):
    d = tmp_path / "old"
    d.mkdir()
    (d / "meeting-2026-05-01T10-00-00.md").write_text("[10:00:01] **Them:** the old meeting text")
    os.utime(d / "meeting-2026-05-01T10-00-00.md", (1, 1))
    monkeypatch.setenv("CO_DB_PATH", str(tmp_path / "c.db"))
    monkeypatch.setenv("CO_CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("CO_EMBEDDING_MODEL", "none")
    assert T.main(["import", str(d), "--json"]) == 0
    events = _lines(capfd.readouterr().out)
    assert [e["event"] for e in events] == ["start", "stored", "indexing", "result"]
    assert events[-1]["imported"] == 1 and events[-1]["embeddings"] == "keyword_only"
    assert T.main(["import", str(d)]) == 0
    assert capfd.readouterr().out.startswith("0 stored, 1 already there")


def test_import_json_missing_path(tmp_path, monkeypatch, capfd):
    monkeypatch.setenv("CO_DB_PATH", str(tmp_path / "c.db"))
    assert T.main(["import", str(tmp_path / "nope"), "--json"]) == 1
    r = _lines(capfd.readouterr().out)[-1]
    assert r["event"] == "result" and not r["ok"] and r["error"]["code"] == "not_found"


def test_env_file_survives_20_concurrent_writers(tmp_path):
    env = tmp_path / "env"
    env.write_text("# settings\nCO_EMBEDDING_MODEL=local\n")
    code = ("import sys; sys.path.insert(0, %r); from pathlib import Path; "
            "from context_orchestrator.settings import set_env_value; "
            "set_env_value(sys.argv[1], sys.argv[2], Path(sys.argv[3]))") % str(SRC)
    procs = [subprocess.Popen([sys.executable, "-c", code, f"KEY_{i}", str(i), str(env)],
                              env=dict(os.environ, HOME=str(tmp_path))) for i in range(20)]
    assert all(p.wait(60) == 0 for p in procs)
    lines = env.read_text().splitlines()
    assert lines[:2] == ["# settings", "CO_EMBEDDING_MODEL=local"]
    assert sorted(l for l in lines if l.startswith("KEY_")) == sorted(f"KEY_{i}={i}" for i in range(20))
    set_env_value("CO_EMBEDDING_MODEL", "none", env)
    assert "CO_EMBEDDING_MODEL=none" in env.read_text() and env.read_text().count("CO_EMBEDDING_MODEL") == 1
