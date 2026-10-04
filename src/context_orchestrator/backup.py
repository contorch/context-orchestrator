"""context-orchestrator's own data operations (run by its own interpreter,
which has chromadb in every channel):

    contorch-memory backup --to DIR [--stop-server] [--json]
    contorch-memory restore --from DIR [--stop-server] [--json]
    contorch-memory index migrate --in-process [--backup-dir DIR] [--json]

Backup = context.db through SQLite's backup API (a consistent snapshot even
with meeting-capture writing), reopened: integrity_check + FTS counts; the
Chroma folder copied under the in-process session lock (chroma.sqlite3
through the backup API, the HNSW files copied, WAL/SHM left out), then the
copy opened with chromadb and every collection counted against the source.
A copy that doesn't verify is `backup_unverified`. With a chroma server
running the folder is live: refused (`server_running`) unless --stop-server,
which stops it for the copy and starts it again.

Restore puts a backup's index back (the current one is moved aside, never
deleted) and verifies the counts; context.db is never rolled back — its copy
stays in the backup for a person to use.

`index migrate --in-process` retires the chroma server: backup with the
server stopped, remove its launchd agent; the same folder is then opened
in-process (under the session lock) and counted.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Optional

SCHEMA = "contorch-memory.backup/1"
FTS_TABLES = {"transcripts": ("transcripts", "transcripts_fts"),
              "repo_knowledge": ("repo_knowledge", "knowledge_fts"),
              "sources": ("sources", "sources_fts")}
SKIP_SUFFIXES = ("-wal", "-shm", "-journal")
SAMPLE_VECTORS = 50


class DataOpError(RuntimeError):
    def __init__(self, code: str, message: str, **detail):
        super().__init__(message)
        self.code = code
        self.detail = detail

    def as_json(self) -> dict:
        return {"code": self.code, "message": str(self), **self.detail}


# ---------------------------------------------------------------- server

def _server_hostport() -> tuple[str, int]:
    from .chroma_daemon import DEFAULT_HOST, DEFAULT_PORT
    return (os.environ.get("CO_CHROMA_HOST", DEFAULT_HOST),
            int(os.environ.get("CO_CHROMA_PORT", DEFAULT_PORT)))


def server_running() -> bool:
    from .chroma_daemon import LAUNCHD_PLIST, is_listening
    if not LAUNCHD_PLIST.exists() and not (os.environ.get("CO_CHROMA_HOST") or os.environ.get("CO_CHROMA_PORT")):
        return False
    return is_listening(*_server_hostport())


def _wait(pred, seconds: float) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.25)
    return pred()


def stop_server() -> None:
    """Stop the chroma server's launchd job (it stays installed)."""
    from .chroma_daemon import LAUNCHD_LABEL, is_listening
    from .launchd import domain, launchctl
    launchctl("bootout", f"{domain()}/{LAUNCHD_LABEL}", quiet=True)
    if not _wait(lambda: not is_listening(*_server_hostport()), 20):
        raise DataOpError("server_stop_failed", "the chroma server kept running after bootout")


def start_server() -> bool:
    from .chroma_daemon import LAUNCHD_PLIST, is_listening
    from .launchd import domain, launchctl
    launchctl("bootstrap", domain(), str(LAUNCHD_PLIST), quiet=True)
    return _wait(lambda: is_listening(*_server_hostport()), 60)


# ---------------------------------------------------------------- sqlite

def _sqlite_copy(src: Path, dst: Path) -> None:
    """A consistent copy through SQLite's online backup API."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    s = sqlite3.connect(str(src), timeout=30)
    d = sqlite3.connect(str(dst))
    try:
        s.backup(d)
        d.execute("PRAGMA journal_mode=DELETE")   # one self-contained file, no -wal/-shm
    finally:
        d.close()
        s.close()


def verify_context_db(path: Path) -> dict:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        counts, fts = {}, {}
        for key, (base, ftst) in FTS_TABLES.items():
            if base in tables:
                counts[key] = con.execute(f"SELECT COUNT(*) FROM {base}").fetchone()[0]
            if ftst in tables:
                fts[key] = con.execute(f"SELECT COUNT(*) FROM {ftst}").fetchone()[0]
        if "tasks" in tables:
            counts["tasks"] = con.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    finally:
        con.close()
    return {"integrity": integrity, "counts": counts, "fts_counts": fts}


# ---------------------------------------------------------------- chroma

def chroma_counts_raw(chroma_dir: Path) -> dict:
    """{collection: documents} from chroma.sqlite3, without chromadb."""
    db = chroma_dir / "chroma.sqlite3"
    if not db.exists():
        return {}
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=10)
    try:
        rows = con.execute(
            "SELECT c.name, COUNT(e.id) FROM collections c JOIN segments s ON s.collection = c.id "
            "AND s.scope = 'METADATA' LEFT JOIN embeddings e ON e.segment_id = s.id GROUP BY c.name")
        return {name: n for name, n in rows}
    finally:
        con.close()


def _copy_chroma(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for root, dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for f in files:
            if f.endswith(SKIP_SUFFIXES) or f.startswith("."):
                continue
            s, d = Path(root) / f, dst / rel / f
            if f == "chroma.sqlite3":
                _sqlite_copy(s, d)
            else:
                shutil.copy2(s, d)


def verify_chroma_copy(path: Path, private_copy: bool = False) -> dict:
    """Open a folder with chromadb and count; fetch the vectors of a sample of
    ids (a vector-less id raises "Error finding id"). `private_copy`: the
    folder is a backup nobody else opens, so its lock file is removed after
    (never for a live index — other processes may be waiting on that file)."""
    from .chroma_lock import lock_path_for, session_lock
    from .search import _clear_chroma_system_cache
    import chromadb
    out: dict = {}
    with session_lock(lock_path_for(path), timeout=60, on_first_acquire=_clear_chroma_system_cache,
                      on_last_release=_clear_chroma_system_cache):
        client = chromadb.PersistentClient(path=str(path))
        for col in client.list_collections():
            c = client.get_collection(col.name)
            n = c.count()
            ids = c.get(include=[], limit=SAMPLE_VECTORS)["ids"]
            for i in ids:
                try:
                    got = c.get(ids=[i], include=["embeddings"])["embeddings"]
                except Exception as exc:   # "Error finding id": metadata without a vector
                    raise DataOpError("backup_unverified", f"{col.name}: {i} has no vector "
                                      f"in the copy ({str(exc)[:120]})")
                if got is None or len(got) == 0:
                    raise DataOpError("backup_unverified", f"{col.name}: {i} has no vector in the copy")
            out[col.name] = n
        del client
    if private_copy:
        lock_path_for(path).unlink(missing_ok=True)
    return out


def _chroma_dir() -> Path:
    from . import search
    from .chroma_daemon import CHROMA_PATH
    return search._embedded_path_if_no_server() or CHROMA_PATH


def _db_path() -> Path:
    from .db import DEFAULT_DB_PATH
    p = os.environ.get("CO_DB_PATH")
    return Path(p) if p else DEFAULT_DB_PATH


# ---------------------------------------------------------------- verbs

def backup(to: Path, stop: bool = False, restart: bool = True) -> dict:
    from .chroma_lock import lock_path_for, session_lock
    from .search import _clear_chroma_system_cache, chromadb_version, read_index_stamp
    to = Path(to).expanduser()
    doc: dict = {"schema": SCHEMA, "ok": False, "action": "backup", "dir": str(to)}
    if to.exists() and any(to.iterdir()):
        raise DataOpError("dir_not_empty", f"{to} is not empty")
    chroma_dir, db_path = _chroma_dir(), _db_path()
    was_running = server_running()
    if was_running and not stop:
        raise DataOpError("server_running", "the chroma server is running; pass --stop-server "
                          "to stop it for the copy (it is started again afterwards)")
    to.mkdir(parents=True, exist_ok=True)
    stopped = False
    try:
        if was_running:
            stop_server()
            stopped = True
        # context.db: consistent snapshot, verified from the copy itself.
        if db_path.exists():
            _sqlite_copy(db_path, to / "context.db")
            cdb = verify_context_db(to / "context.db")
            doc["context_db"] = cdb
            if cdb["integrity"] != "ok":
                raise DataOpError("backup_unverified", f"context.db copy: integrity_check = {cdb['integrity']}")
        else:
            doc["context_db"] = None
        # Chroma: copied with no other process inside a session.
        source_counts: dict = {}
        if chroma_dir.exists():
            with session_lock(lock_path_for(chroma_dir), timeout=120,
                              on_first_acquire=_clear_chroma_system_cache,
                              on_last_release=_clear_chroma_system_cache):
                source_counts = chroma_counts_raw(chroma_dir)
                _copy_chroma(chroma_dir, to / "chroma")
            try:
                copy_counts = verify_chroma_copy(to / "chroma", private_copy=True)
            except DataOpError:
                raise
            except Exception as exc:     # the copy doesn't even open
                raise DataOpError("backup_unverified", f"the index copy does not open: "
                                  f"{type(exc).__name__}: {str(exc)[:200]}")
            if copy_counts != source_counts:
                raise DataOpError("backup_unverified", "chroma copy counts differ from the source",
                                  source=source_counts, copy=copy_counts)
        doc["chroma"] = {"collections": source_counts, "path": str(chroma_dir)}
    finally:
        if stopped:
            doc["server_restarted"] = start_server() if restart else False
    manifest = {"schema": SCHEMA, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "chromadb_version": chromadb_version(),
                "index_written_by": read_index_stamp(chroma_dir).get("chromadb_version"),
                "context_db": doc["context_db"], "chroma": doc["chroma"],
                "source": {"db": str(db_path), "chroma": str(chroma_dir)}}
    (to / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    os.chmod(to, 0o700)
    doc.update(ok=True, chromadb_version=manifest["chromadb_version"],
               index_written_by=manifest["index_written_by"])
    return doc


def restore(frm: Path, stop: bool = False) -> dict:
    from .chroma_lock import lock_path_for, session_lock
    from .memstatus import index_compatible
    from .search import _clear_chroma_system_cache, chromadb_version
    frm = Path(frm).expanduser()
    try:
        manifest = json.loads((frm / "manifest.json").read_text())
    except (OSError, ValueError):
        raise DataOpError("not_a_backup", f"{frm} has no readable manifest.json")
    if manifest.get("schema") != SCHEMA or not (frm / "chroma").is_dir():
        raise DataOpError("not_a_backup", f"{frm} is not a contorch-memory backup with an index")
    installed = chromadb_version()
    made_by = manifest.get("index_written_by") or manifest.get("chromadb_version")
    if not index_compatible(made_by, installed):
        raise DataOpError("chroma_downgrade", f"the backup was written by chromadb {made_by}; "
                          f"this is {installed}")
    chroma_dir = _chroma_dir()
    was_running = server_running()
    if was_running and not stop:
        raise DataOpError("server_running", "the chroma server is running; pass --stop-server")
    doc: dict = {"schema": SCHEMA, "ok": False, "action": "restore", "dir": str(frm)}
    stopped = False
    aside = chroma_dir.with_name(f"{chroma_dir.name}.before-restore-{time.strftime('%Y%m%d-%H%M%S')}")
    try:
        if was_running:
            stop_server()
            stopped = True
        with session_lock(lock_path_for(chroma_dir), timeout=120,
                          on_first_acquire=_clear_chroma_system_cache,
                          on_last_release=_clear_chroma_system_cache):
            if chroma_dir.exists():
                os.replace(chroma_dir, aside)
                doc["previous_index"] = str(aside)
            _copy_chroma(frm / "chroma", chroma_dir)
            counts = chroma_counts_raw(chroma_dir)
            want = (manifest.get("chroma") or {}).get("collections") or {}
            if counts != want:
                shutil.rmtree(chroma_dir)
                if aside.exists():
                    os.replace(aside, chroma_dir)
                    doc.pop("previous_index", None)
                raise DataOpError("restore_unverified", "restored counts differ from the backup",
                                  backup=want, restored=counts)
        doc["chroma"] = {"collections": counts, "path": str(chroma_dir)}
        doc["context_db"] = "kept (never rolled back; the backup's copy is " + str(frm / "context.db") + ")"
        doc["ok"] = True
    finally:
        if stopped:
            doc["server_restarted"] = start_server()
    return doc


def migrate_in_process(backup_dir: Optional[Path] = None) -> dict:
    """Retire the chroma server: back up (server stopped), remove its agent,
    then open the same folder in-process and count."""
    from . import chroma_daemon
    from .launchd import launchctl
    doc: dict = {"schema": SCHEMA, "ok": False, "action": "index_migrate", "todo": []}
    if not chroma_daemon.LAUNCHD_PLIST.exists():
        doc.update(ok=True, performed=False, vector_index="in_process",
                   note="no chroma server installed; the index is already in-process")
        return doc
    backup_dir = Path(backup_dir).expanduser() if backup_dir else (
        Path.home() / ".context-orchestrator" / "backups" / f"migrate-{time.strftime('%Y%m%d-%H%M%S')}")
    b = backup(backup_dir, stop=True, restart=False)
    doc["backup"] = b
    # The server is stopped; remove its agent for good.
    try:
        launchctl("unload", "-w", str(chroma_daemon.LAUNCHD_PLIST), quiet=True)
    except Exception:
        pass
    chroma_daemon.LAUNCHD_PLIST.unlink(missing_ok=True)
    # The same folder, in-process now (the agent is gone, so search picks it).
    folder = Path(b["chroma"]["path"])
    counts = verify_chroma_copy(folder) if folder.exists() else {}
    if counts != b["chroma"]["collections"]:
        raise DataOpError("migrate_unverified", "in-process counts differ from the server's",
                          server=b["chroma"]["collections"], in_process=counts)
    doc.update(ok=True, performed=True, vector_index="in_process",
               chroma={"collections": counts, "path": str(folder)})
    doc["todo"].append("Run `contorch-memory claude install` (drops CO_CHROMA_HOST/PORT from the MCP "
                       "entry) and restart Claude Code")
    return doc


def describe(doc: dict) -> str:
    if not doc.get("ok"):
        e = doc.get("error") or {}
        return f"{doc.get('action')}: failed — {e.get('code')}: {e.get('message')}"
    lines = [f"{doc['action']}: ok" + (f" → {doc['dir']}" if doc.get("dir") else "")]
    if doc.get("context_db") and isinstance(doc["context_db"], dict):
        c = doc["context_db"]
        lines.append(f"  context.db: integrity {c['integrity']}, {c['counts']}")
    ch = doc.get("chroma") or (doc.get("backup") or {}).get("chroma")
    if ch:
        lines.append(f"  index: {ch['collections']}")
    for t in doc.get("todo") or []:
        lines.append(f"  todo: {t}")
    return "\n".join(lines)
