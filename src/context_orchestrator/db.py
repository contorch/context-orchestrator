import re
import sqlite3
import sys
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("context-orchestrator")

DEFAULT_DB_PATH = Path.home() / ".context-orchestrator" / "context.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    project TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(name, project)
);

CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    source_type TEXT NOT NULL,
    reference TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT '',
    added_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(task_id, source_type, reference)
);

CREATE TABLE IF NOT EXISTS repo_knowledge (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_url TEXT NOT NULL,
    insight TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_repo_knowledge_url ON repo_knowledge(repo_url);
CREATE INDEX IF NOT EXISTS idx_sources_task_id ON sources(task_id);
CREATE INDEX IF NOT EXISTS idx_tasks_project ON tasks(project);
""" + """
-- Meeting / podcast transcripts. The full text lives here (there are no
-- transcript files); Chroma holds only the search index built from `body`.
-- meeting-capture appends to this table directly while a meeting runs, so
-- this DDL is a shared contract: keep it identical in
-- meeting-capture/src/meeting_capture/store.py.
CREATE TABLE IF NOT EXISTS transcripts (
    meeting_id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT '',
    started_at TEXT NOT NULL DEFAULT '',
    body TEXT NOT NULL DEFAULT '',
    content_sha TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    indexed_at REAL,
    indexed_with TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_transcripts_updated ON transcripts(updated_at);
"""

# Full-text search (SQLite FTS5) over everything the vector index covers, so
# search works with no embedding model at all and finds a transcript the
# moment it is stored. Kept in sync by triggers — including meeting-capture's
# line-by-line appends, which never need to know these tables exist. The FTS
# rowid is the source row's rowid, so every sync is an indexed lookup.
FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS transcripts_fts USING fts5(
    title, body, tokenize='porter unicode61');
CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
    body, tokenize='porter unicode61');
CREATE VIRTUAL TABLE IF NOT EXISTS sources_fts USING fts5(
    body, notes, tokenize='porter unicode61');

CREATE TRIGGER IF NOT EXISTS transcripts_fts_ai AFTER INSERT ON transcripts BEGIN
    INSERT INTO transcripts_fts(rowid, title, body) VALUES (new.rowid, new.title, new.body);
END;
CREATE TRIGGER IF NOT EXISTS transcripts_fts_au AFTER UPDATE OF title, body ON transcripts BEGIN
    DELETE FROM transcripts_fts WHERE rowid = old.rowid;
    INSERT INTO transcripts_fts(rowid, title, body) VALUES (new.rowid, new.title, new.body);
END;
CREATE TRIGGER IF NOT EXISTS transcripts_fts_ad AFTER DELETE ON transcripts BEGIN
    DELETE FROM transcripts_fts WHERE rowid = old.rowid;
END;

CREATE TRIGGER IF NOT EXISTS knowledge_fts_ai AFTER INSERT ON repo_knowledge BEGIN
    INSERT INTO knowledge_fts(rowid, body) VALUES (new.id, new.insight);
END;
CREATE TRIGGER IF NOT EXISTS knowledge_fts_ad AFTER DELETE ON repo_knowledge BEGIN
    DELETE FROM knowledge_fts WHERE rowid = old.id;
END;

CREATE TRIGGER IF NOT EXISTS sources_fts_ai AFTER INSERT ON sources BEGIN
    INSERT INTO sources_fts(rowid, body, notes) VALUES (
        new.id, CASE WHEN new.source_type = 'text' THEN new.reference ELSE '' END, new.notes);
END;
CREATE TRIGGER IF NOT EXISTS sources_fts_ad AFTER DELETE ON sources BEGIN
    DELETE FROM sources_fts WHERE rowid = old.id;
END;
"""

_FTS_BACKFILL = (
    ("transcripts_fts", "transcripts",
     "INSERT INTO transcripts_fts(rowid, title, body) SELECT rowid, title, body FROM transcripts"),
    ("knowledge_fts", "repo_knowledge",
     "INSERT INTO knowledge_fts(rowid, body) SELECT id, insight FROM repo_knowledge"),
    ("sources_fts", "sources",
     "INSERT INTO sources_fts(rowid, body, notes) SELECT id, "
     "CASE WHEN source_type = 'text' THEN reference ELSE '' END, notes FROM sources"),
)


def fts_query(text: str) -> str:
    """Free text → a safe FTS5 query: every word, OR-ed, each quoted (so
    punctuation and FTS operators in the user's text can't break the query).
    bm25 ranks documents matching more of the words higher."""
    words = [w for w in re.findall(r"\w+", text.lower()) if len(w) > 1 or w.isdigit()]
    seen, out = set(), []
    for w in words:
        if w not in seen:
            seen.add(w)
            out.append('"' + w + '"')
    return " OR ".join(out)


class Database:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # timeout: meeting-capture writes transcript lines from another process.
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=10)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()
        logger.info(f"Database initialized at {self.db_path}")

    def _create_tables(self):
        self.conn.executescript(SCHEMA)
        # A transcripts table created by an older meeting-capture lacks the
        # newest columns; add them in place.
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(transcripts)")}
        if "indexed_with" not in cols:
            self.conn.execute("ALTER TABLE transcripts ADD COLUMN indexed_with TEXT NOT NULL DEFAULT ''")
        self.conn.executescript(FTS_SCHEMA)
        # First run with FTS (or rows written while it didn't exist): fill it.
        for fts, table, fill in _FTS_BACKFILL:
            n_fts = self.conn.execute(f"SELECT COUNT(*) FROM {fts}").fetchone()[0]
            n_src = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            if n_fts != n_src:
                self.conn.execute(f"DELETE FROM {fts}")
                self.conn.execute(fill)
        self.conn.commit()

    # --- Tasks ---

    def create_task(self, name: str, description: str = "", project: str = "") -> dict:
        try:
            cur = self.conn.execute(
                "INSERT INTO tasks (name, description, project) VALUES (?, ?, ?)",
                (name, description, project),
            )
            self.conn.commit()
            return self._get_task_by_id(cur.lastrowid)
        except sqlite3.IntegrityError:
            raise ValueError(f"Task '{name}' already exists in project '{project}'")

    def list_tasks(self, project: Optional[str] = None) -> list[dict]:
        if project is not None:
            rows = self.conn.execute(
                """SELECT t.*, COUNT(s.id) as source_count
                   FROM tasks t LEFT JOIN sources s ON t.id = s.task_id
                   WHERE t.project = ?
                   GROUP BY t.id ORDER BY t.created_at DESC""",
                (project,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """SELECT t.*, COUNT(s.id) as source_count
                   FROM tasks t LEFT JOIN sources s ON t.id = s.task_id
                   GROUP BY t.id ORDER BY t.created_at DESC""",
            ).fetchall()
        return [dict(r) for r in rows]

    def get_task_by_name(self, name: str, project: Optional[str] = None) -> Optional[dict]:
        if project is not None:
            row = self.conn.execute(
                "SELECT * FROM tasks WHERE name = ? AND project = ?",
                (name, project),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT * FROM tasks WHERE name = ?", (name,)
            ).fetchone()
        return dict(row) if row else None

    def _get_task_by_id(self, task_id: int) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        return dict(row) if row else None

    # --- Sources ---

    def add_source(
        self, task_id: int, source_type: str, reference: str, notes: str = ""
    ) -> dict:
        try:
            cur = self.conn.execute(
                "INSERT INTO sources (task_id, source_type, reference, notes) VALUES (?, ?, ?, ?)",
                (task_id, source_type, reference, notes),
            )
            self.conn.commit()
            row = self.conn.execute(
                "SELECT * FROM sources WHERE id = ?", (cur.lastrowid,)
            ).fetchone()
            return dict(row)
        except sqlite3.IntegrityError:
            raise ValueError(
                f"Source ({source_type}: {reference}) already exists in this task"
            )

    def get_sources_for_task(self, task_id: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM sources WHERE task_id = ? ORDER BY added_at",
            (task_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def remove_source(self, source_id: int) -> bool:
        cur = self.conn.execute("DELETE FROM sources WHERE id = ?", (source_id,))
        self.conn.commit()
        return cur.rowcount > 0

    # --- Repo Knowledge ---

    def update_repo_knowledge(self, repo_url: str, insight: str) -> dict:
        cur = self.conn.execute(
            "INSERT INTO repo_knowledge (repo_url, insight) VALUES (?, ?)",
            (repo_url, insight),
        )
        self.conn.commit()
        row = self.conn.execute(
            "SELECT * FROM repo_knowledge WHERE id = ?", (cur.lastrowid,)
        ).fetchone()
        return dict(row)

    def get_repo_knowledge(self, repo_url: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM repo_knowledge WHERE repo_url = ? ORDER BY created_at",
            (repo_url,),
        ).fetchall()
        return [dict(r) for r in rows]

    # --- Transcripts ---

    def put_transcript(self, meeting_id: str, body: str, title: str = "", source: str = "",
                       started_at: str = "", content_sha: str = "",
                       now: Optional[float] = None) -> bool:
        """Insert or replace a transcript's full text. Returns True if anything
        changed. An identical body is a no-op, so re-importing the same file
        doesn't re-embed it."""
        now = time.time() if now is None else now
        row = self.get_transcript(meeting_id)
        if row and row["body"] == body:
            return False
        if row:
            self.conn.execute(
                "UPDATE transcripts SET body = ?, title = COALESCE(NULLIF(?, ''), title), "
                "source = COALESCE(NULLIF(?, ''), source), "
                "started_at = COALESCE(NULLIF(?, ''), started_at), content_sha = ?, "
                "updated_at = ? WHERE meeting_id = ?",
                (body, title, source, started_at, content_sha, now, meeting_id),
            )
        else:
            self.conn.execute(
                "INSERT INTO transcripts (meeting_id, title, source, started_at, body, "
                "content_sha, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (meeting_id, title, source, started_at, body, content_sha, now, now),
            )
        self.conn.commit()
        return True

    def get_transcript(self, meeting_id: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM transcripts WHERE meeting_id = ?", (meeting_id,)
        ).fetchone()
        return dict(row) if row else None

    def find_transcript_by_sha(self, content_sha: str) -> Optional[dict]:
        if not content_sha:
            return None
        row = self.conn.execute(
            "SELECT * FROM transcripts WHERE content_sha = ? LIMIT 1", (content_sha,)
        ).fetchone()
        return dict(row) if row else None

    def list_transcripts(self, limit: int = 50) -> list[dict]:
        rows = self.conn.execute(
            "SELECT meeting_id, title, source, started_at, created_at, updated_at, indexed_at, "
            "length(body) AS chars FROM transcripts ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def latest_transcript(self) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT meeting_id, updated_at FROM transcripts ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def transcripts_to_index(self, settled_before: float, embedding: str = "") -> list[dict]:
        """Transcripts changed since they were last indexed — or indexed with
        a different embedding model than `embedding` — and quiet since
        `settled_before` (a live meeting is still being appended to)."""
        if embedding:
            # Rows indexed before models were tracked were indexed with
            # whatever was configured then; adopt the current one once.
            self.conn.execute(
                "UPDATE transcripts SET indexed_with = ? WHERE indexed_with = '' "
                "AND indexed_at IS NOT NULL", (embedding,))
            self.conn.commit()
        rows = self.conn.execute(
            "SELECT * FROM transcripts WHERE (indexed_at IS NULL OR indexed_at < updated_at "
            "OR (? != '' AND indexed_with != ?)) AND updated_at <= ? ORDER BY updated_at",
            (embedding, embedding, settled_before),
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_transcript_indexed(self, meeting_id: str, as_of: float, embedding: str = "") -> None:
        """`as_of` is the updated_at the index was built from, not the clock:
        a line appended mid-index leaves the row pending for the next pass."""
        self.conn.execute(
            "UPDATE transcripts SET indexed_at = ?, indexed_with = ? WHERE meeting_id = ?",
            (as_of, embedding, meeting_id)
        )
        self.conn.commit()

    # --- Full-text search ---

    def search_text(self, query: str, limit: int = 10, meeting_id: str = "",
                    after: str = "", before: str = "", project: str = "") -> list[dict]:
        """Keyword search (FTS5, bm25-ranked, stemmed) over transcripts, repo
        knowledge and text sources/notes. Works with no embedding model.
        Date filters (ISO strings) apply to transcripts' start time and
        restrict results to transcripts; `project` restricts to that
        project's task sources."""
        q = fts_query(query)
        if not q:
            return []
        hits: list[dict] = []
        if not project:
            where, args = ["transcripts_fts MATCH ?"], [q]
            if meeting_id:
                where.append("t.meeting_id = ?"); args.append(meeting_id)
            if after:
                where.append("t.started_at >= ?"); args.append(after)
            if before:
                where.append("t.started_at <= ?"); args.append(before)
            for r in self.conn.execute(
                "SELECT t.meeting_id, t.title, t.started_at, bm25(transcripts_fts, 4.0, 1.0) AS rank, "
                "snippet(transcripts_fts, 1, '', '', ' … ', 48) AS snip "
                f"FROM transcripts_fts JOIN transcripts t ON t.rowid = transcripts_fts.rowid "
                f"WHERE {' AND '.join(where)} ORDER BY rank LIMIT ?", (*args, limit)):
                hits.append({"kind": "transcript", "meeting_id": r["meeting_id"], "title": r["title"],
                             "started_at": r["started_at"], "text": r["snip"], "rank": r["rank"]})
        if meeting_id or after or before:
            return hits[:limit]
        if not project:
            for r in self.conn.execute(
                "SELECT k.id, k.repo_url, k.insight, bm25(knowledge_fts) AS rank FROM knowledge_fts "
                "JOIN repo_knowledge k ON k.id = knowledge_fts.rowid WHERE knowledge_fts MATCH ? "
                "ORDER BY rank LIMIT ?", (q, limit)):
                hits.append({"kind": "repo_knowledge", "repo_url": r["repo_url"], "text": r["insight"],
                             "rank": r["rank"]})
        where, args = ["sources_fts MATCH ?"], [q]
        if project:
            where.append("t.project = ?"); args.append(project)
        for r in self.conn.execute(
            "SELECT s.id, s.source_type, s.reference, s.notes, t.name AS task_name, "
            "bm25(sources_fts) AS rank FROM sources_fts JOIN sources s ON s.id = sources_fts.rowid "
            f"JOIN tasks t ON t.id = s.task_id WHERE {' AND '.join(where)} ORDER BY rank LIMIT ?",
            (*args, limit)):
            text = r["reference"] if r["source_type"] == "text" else f"{r['reference']} — {r['notes']}"
            hits.append({"kind": "source", "task_name": r["task_name"], "source_type": r["source_type"],
                         "text": text, "rank": r["rank"]})
        hits.sort(key=lambda h: h["rank"])  # bm25: lower is better
        return hits[:limit]

    def delete_transcript(self, meeting_id: str) -> bool:
        cur = self.conn.execute("DELETE FROM transcripts WHERE meeting_id = ?", (meeting_id,))
        self.conn.commit()
        return cur.rowcount > 0

    def count_transcripts(self) -> tuple[int, int]:
        """(total, not yet indexed)."""
        total, pending = self.conn.execute(
            "SELECT COUNT(*), SUM(CASE WHEN indexed_at IS NULL OR indexed_at < updated_at "
            "THEN 1 ELSE 0 END) FROM transcripts"
        ).fetchone()
        return total, pending or 0
