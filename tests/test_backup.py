"""`contorch-memory backup|restore|index migrate` (backup.py)."""
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from context_orchestrator import backup as B
from context_orchestrator import chroma_daemon, launchd, search, settings
from context_orchestrator.db import Database

SRC = Path(__file__).resolve().parent.parent / "src"


class _FakeEF:
    def name(self):
        return "fake-ef"

    def __call__(self, input):
        return [[float((sum(map(ord, t)) >> i) & 1) + 0.01 for i in range(8)] for t in input]

    def embed_query(self, input):
        return self(input if isinstance(input, list) else [input])


@pytest.fixture
def mem(tmp_path, monkeypatch):
    monkeypatch.setenv("CO_DB_PATH", str(tmp_path / "home" / "context.db"))
    monkeypatch.setenv("CO_CHROMA_PATH", str(tmp_path / "home" / "chroma"))
    monkeypatch.setenv("CO_EMBEDDING_MODEL", "local")
    monkeypatch.setattr(search, "_build_embedding_function", lambda: _FakeEF())
    monkeypatch.setattr(chroma_daemon, "LAUNCHD_PLIST", tmp_path / "no-agent.plist")
    db = Database(db_path=tmp_path / "home" / "context.db")
    t = db.create_task("launch", project="p")
    db.add_source(t["id"], "text", "zebra migration ships Thursday", notes="")
    db.update_repo_knowledge("r", "run make test before pushing")
    vs = search.VectorSearch(chroma_path=tmp_path / "home" / "chroma")
    vs.upsert([f"d{i}" for i in range(30)], [f"doc number {i} about budgets" for i in range(30)],
              [{"type": "t"}] * 30)
    return tmp_path


def test_backup_verified_and_restore_round_trip(mem, capfd):
    out = mem / "bk"
    assert settings.main(["backup", "--to", str(out), "--json"]) == 0
    doc = json.loads(capfd.readouterr().out)
    assert doc["schema"] == "contorch-memory.backup/1" and doc["ok"]
    assert doc["context_db"]["integrity"] == "ok"
    assert doc["context_db"]["fts_counts"] == {"transcripts": 0, "repo_knowledge": 1, "sources": 1}
    assert doc["chroma"]["collections"] == {"context": 30}
    assert sorted(p.name for p in out.iterdir()) == ["chroma", "context.db", "manifest.json"]
    assert oct(out.stat().st_mode & 0o777) == "0o700"

    # the live index changes; restore brings the backup's back, current one kept aside
    vs = search.VectorSearch(chroma_path=mem / "home" / "chroma")
    vs.remove_many([f"d{i}" for i in range(10)])
    assert vs.count() == 20
    assert settings.main(["restore", "--from", str(out), "--json"]) == 0
    r = json.loads(capfd.readouterr().out)
    assert r["ok"] and r["chroma"]["collections"] == {"context": 30}
    assert Path(r["previous_index"]).is_dir()
    assert search.VectorSearch(chroma_path=mem / "home" / "chroma").count() == 30
    assert "never rolled back" in r["context_db"]


def test_backup_refuses_a_non_empty_dir(mem, capfd):
    d = mem / "full"
    d.mkdir()
    (d / "x").write_text("x")
    assert settings.main(["backup", "--to", str(d), "--json"]) == 1
    assert json.loads(capfd.readouterr().out)["error"]["code"] == "dir_not_empty"


@pytest.mark.parametrize("damage", ["truncate", "drop_row"])
def test_corrupted_copy_is_backup_unverified(mem, monkeypatch, damage):
    real = B._copy_chroma

    def lossy(src, dst):
        real(src, dst)
        db = dst / "chroma.sqlite3"
        if damage == "truncate":
            data = db.read_bytes()
            db.write_bytes(data[: len(data) // 3])
        else:
            con = sqlite3.connect(db)
            con.execute("DELETE FROM embeddings WHERE id = (SELECT MIN(id) FROM embeddings)")
            con.commit()
            con.close()
    monkeypatch.setattr(B, "_copy_chroma", lossy)
    with pytest.raises(B.DataOpError) as e:
        B.backup(mem / "bk")
    assert e.value.code == "backup_unverified"


def test_backup_while_a_writer_runs(mem):
    """meeting-capture-style SQLite appends + in-process index writes from
    another process, during the backup: the copy is consistent and verifies."""
    code = f"""
import sys, time, sqlite3
sys.path.insert(0, {str(SRC)!r}); sys.path.insert(0, {str(Path(__file__).parent)!r})
from context_orchestrator import search
import test_backup as t
search._build_embedding_function = lambda: t._FakeEF()
vs = search.VectorSearch(chroma_path=__import__('pathlib').Path({str(mem / 'home' / 'chroma')!r}))
con = sqlite3.connect({str(mem / 'home' / 'context.db')!r}, timeout=30)
end = time.time() + 6
i = 0
while time.time() < end:
    con.execute("INSERT INTO transcripts (meeting_id, title, source, started_at, body, created_at, updated_at) "
                "VALUES (?, '', 'meeting-capture', '', ?, 1, 1)", (f"m{{i}}", f"[10:00] line {{i}} about zebras"))
    con.commit()
    vs.add(f"w{{i}}", f"writer doc {{i}} about zebras", {{"type": "t"}})
    i += 1
print(i)
"""
    w = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(1.5)
    doc = B.backup(mem / "bk")
    out, err = w.communicate(timeout=60)
    assert w.returncode == 0, err[-2000:]
    assert doc["ok"] and doc["context_db"]["integrity"] == "ok"
    c = doc["context_db"]
    assert c["counts"]["transcripts"] == c["fts_counts"]["transcripts"] > 0, "FTS consistent with its table"
    assert doc["chroma"]["collections"]["context"] > 30
    assert int(out.strip()) > 0


def test_server_running_is_refused_without_stop_server(mem, monkeypatch, capfd):
    monkeypatch.setattr(B, "server_running", lambda: True)
    assert settings.main(["backup", "--to", str(mem / "bk"), "--json"]) == 1
    assert json.loads(capfd.readouterr().out)["error"]["code"] == "server_running"


def test_restore_refuses_a_backup_from_a_newer_chromadb(mem):
    B.backup(mem / "bk")
    m = json.loads((mem / "bk" / "manifest.json").read_text())
    m["index_written_by"] = "9.9.9"
    (mem / "bk" / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(B.DataOpError) as e:
        B.restore(mem / "bk")
    assert e.value.code == "chroma_downgrade"


def test_launchctl_is_refused_in_tests():
    with pytest.raises(launchd.LaunchctlDisabled):
        launchd.launchctl("list")


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_migrate_from_a_scratch_server(tmp_path, monkeypatch, capfd):
    """A real `chroma run` (not launchd) on a free port stands in for the
    agent; stop/unload are stubbed at the launchd boundary."""
    folder = tmp_path / "home" / "chroma"
    folder.mkdir(parents=True)
    port = _free_port()
    chroma_cli = Path(sys.executable).parent / "chroma"
    srv = subprocess.Popen([str(chroma_cli), "run", "--path", str(folder), "--host", "127.0.0.1",
                            "--port", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           env=dict(os.environ, ANONYMIZED_TELEMETRY="False"))
    try:
        assert B._wait(lambda: chroma_daemon.is_listening("127.0.0.1", port), 60), "scratch server did not start"
        plist = tmp_path / "agent.plist"
        plist.write_text("<plist/>")
        monkeypatch.setattr(chroma_daemon, "LAUNCHD_PLIST", plist)
        monkeypatch.setattr(chroma_daemon, "CHROMA_PATH", folder)
        monkeypatch.delenv("CO_CHROMA_PATH", raising=False)
        monkeypatch.setenv("CO_CHROMA_PORT", str(port))
        monkeypatch.setenv("CO_DB_PATH", str(tmp_path / "home" / "context.db"))
        monkeypatch.setenv("CO_EMBEDDING_MODEL", "local")
        unloaded = []

        def fake_stop():
            srv.terminate()
            srv.wait(30)
        monkeypatch.setattr(B, "stop_server", fake_stop)
        monkeypatch.setattr(launchd, "launchctl", lambda *a, **k: unloaded.append(a))
        import chromadb
        from chromadb.api.shared_system_client import SharedSystemClient
        client = chromadb.HttpClient(host="127.0.0.1", port=port)
        col = client.get_or_create_collection("context", metadata={"hnsw:space": "cosine"})
        col.upsert(ids=[f"s{i}" for i in range(40)], embeddings=[[float(i), 1.0, 0.5] for i in range(40)],
                   documents=[f"server doc {i}" for i in range(40)])
        del col, client
        SharedSystemClient.clear_system_cache()

        assert settings.main(["index", "migrate", "--in-process", "--backup-dir",
                              str(tmp_path / "bk"), "--json"]) == 0
        doc = json.loads(capfd.readouterr().out)
        assert doc["ok"] and doc["performed"] and doc["vector_index"] == "in_process"
        assert doc["backup"]["chroma"]["collections"] == {"context": 40}
        assert doc["chroma"]["collections"] == {"context": 40}
        assert not plist.exists() and unloaded and unloaded[0][:2] == ("unload", "-w")
        assert srv.poll() is not None, "server stopped and not restarted"
        assert any("claude install" in t for t in doc["todo"])
        # in-process now: the same folder, through the session lock
        monkeypatch.delenv("CO_CHROMA_PORT")
        monkeypatch.setattr(search, "_build_embedding_function", lambda: None)
        monkeypatch.setattr(search, "DEFAULT_CHROMA_PATH", folder)
        vs = search.VectorSearch(verify=False)
        assert vs.in_process and vs.chroma_path == folder and vs.count() == 40
        # nothing left to migrate
        assert B.migrate_in_process()["performed"] is False
    finally:
        if srv.poll() is None:
            srv.kill()
            srv.wait()
