import sys
import os
import logging
import time
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

import anyio
from mcp.server.fastmcp import FastMCP
from context_orchestrator.db import Database
from context_orchestrator.search import VectorSearch
from context_orchestrator.ingest import index_file_content
from context_orchestrator.watcher import catch_up as _catch_up_transcripts

# How long after a meeting transcript was last touched do we still auto-link new sources to it?
RECENT_MEETING_WINDOW_SECONDS = 10 * 60
TRANSCRIPTS_DIR = Path.home() / "transcripts"


def _resolve_default_task_name(project: str = "") -> str:
    """If a meeting transcript was modified in the last RECENT_MEETING_WINDOW_SECONDS,
    use its name (sans .md). Otherwise fall back to inbox-YYYY-MM-DD."""
    if TRANSCRIPTS_DIR.exists():
        candidates = sorted(
            TRANSCRIPTS_DIR.glob("meeting-*.md"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            most_recent = candidates[0]
            age = time.time() - most_recent.stat().st_mtime
            if age <= RECENT_MEETING_WINDOW_SECONDS:
                return most_recent.stem  # e.g. "meeting-2026-04-26T18-43-05"
    return f"inbox-{date.today().isoformat()}"


def _detect_source_type(reference: str) -> str:
    """Heuristic: url / file / text from the reference string alone."""
    ref = reference.strip()
    if ref.startswith(("http://", "https://")):
        return "url"
    expanded = Path(ref).expanduser()
    if expanded.exists() and expanded.is_file():
        return "file"
    return "text"

# CRITICAL: Never print to stdout — it corrupts the JSON-RPC protocol.
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger("context-orchestrator")

db_path = os.environ.get("CO_DB_PATH")
db = Database(db_path=Path(db_path) if db_path else None)

chroma_path = os.environ.get("CO_CHROMA_PATH")
vs = VectorSearch(chroma_path=Path(chroma_path) if chroma_path else None)


def _sync_index():
    """Ensure all existing data is indexed. Runs in background on startup.

    Fast-path: if chroma already has at least one entry for every
    repo_knowledge id and every source id, skip the whole sync. This
    avoids re-embedding on every startup and — more importantly —
    eliminates the timing race where the background sync thread is mid-
    Gemini-call when an MCP tool request lands and the stdio pipe drops.
    """
    existing_ids = set(vs.collection.get(include=[])["ids"])
    rk_ids = {f"repo_knowledge:{r[0]}" for r in db.conn.execute(
        "SELECT id FROM repo_knowledge").fetchall()}
    # Only text sources get indexed as `source:<id>`; repo/url types are
    # never embedded, so we don't expect them in chroma.
    src_ids = {f"source:{r[0]}" for r in db.conn.execute(
        "SELECT id FROM sources WHERE source_type='text'").fetchall()}
    notes_ids = {f"source_notes:{r[0]}" for r in db.conn.execute(
        "SELECT id FROM sources WHERE notes != ''").fetchall()}
    expected = rk_ids | src_ids | notes_ids
    if expected.issubset(existing_ids):
        logger.info("Vector index already in sync — skipping startup reindex")
        return

    indexed = 0

    # Index repo knowledge
    for row in db.conn.execute("SELECT * FROM repo_knowledge").fetchall():
        row = dict(row)
        doc_id = f"repo_knowledge:{row['id']}"
        if doc_id in existing_ids:
            continue
        vs.add(doc_id, row["insight"], {
            "type": "repo_knowledge",
            "repo_url": row["repo_url"],
        })
        indexed += 1

    # Index sources: text content, file content (chunked), and notes
    for row in db.conn.execute(
        "SELECT s.*, t.name as task_name, t.project FROM sources s JOIN tasks t ON s.task_id = t.id"
    ).fetchall():
        row = dict(row)
        if row["source_type"] == "text":
            doc_id = f"source:{row['id']}"
            if doc_id not in existing_ids:
                vs.add(doc_id, row["reference"], {
                    "type": "source",
                    "source_type": row["source_type"],
                    "task_name": row["task_name"],
                    "project": row["project"],
                })
                indexed += 1
        elif row["source_type"] == "file":
            num_chunks = index_file_content(
                vs, row["id"], row["reference"],
                task_name=row["task_name"],
                project=row["project"],
            )
            indexed += num_chunks
        if row["notes"]:
            doc_id = f"source_notes:{row['id']}"
            if doc_id in existing_ids:
                continue
            vs.add(doc_id, row["notes"], {
                "type": "source_notes",
                "source_type": row["source_type"],
                "reference": row["reference"],
                "task_name": row["task_name"],
                "project": row["project"],
            })
            indexed += 1

    logger.info(f"Synced {indexed} documents to vector index")


# Transcripts are indexed on demand — there is no watcher daemon by default.
# At most one catch-up per CATCH_UP_EVERY_S from search(), plus one at startup.
CATCH_UP_EVERY_S = 30.0
_last_catch_up = 0.0


def _index_new_transcripts(force: bool = False) -> None:
    global _last_catch_up
    now = time.monotonic()
    if not force and now - _last_catch_up < CATCH_UP_EVERY_S:
        return
    _last_catch_up = now
    try:
        indexed = _catch_up_transcripts(vs)
        if indexed:
            logger.info("indexed %d new/changed transcript(s): %s",
                        len(indexed), ", ".join(p.name for p in indexed))
    except Exception:
        logger.exception("transcript catch-up failed")


def _startup_sync() -> None:
    _sync_index()
    _index_new_transcripts(force=True)


@asynccontextmanager
async def _lifespan(app: FastMCP):
    """Run the index sync in a background thread so the server starts accepting connections immediately."""
    async with anyio.create_task_group() as tg:
        tg.start_soon(anyio.to_thread.run_sync, _startup_sync)
        yield
        tg.cancel_scope.cancel()


mcp = FastMCP(
    "context-orchestrator",
    instructions="""You have access to the Context Orchestrator — a persistent task and context management system.

IMPORTANT BEHAVIORS:
1. AUTO-DETECT PROJECT: Before creating or listing tasks, detect the current project by running
   `git remote get-url origin` in the working directory. Pass the result as the `project` parameter.
   This scopes tasks to the correct project.

2. AUTO-LOAD CONTEXT: When the user mentions working on a task (e.g., "I'm working on auth-refactor"),
   call get_task() to load the full context manifest. Then read any file sources and clone any repo
   sources as needed.

3. AUTO-LEARN: When you discover something about a repository — setup steps, test commands, build
   requirements, conventions, gotchas, PR structure — proactively call update_repo_knowledge() to
   save it. Do NOT ask the user. Just save it. This knowledge will automatically appear in future
   sessions for any task that references this repo.

4. SOURCE CAPTURE: When the user pastes links, file paths, or text and associates them with a task,
   call add_source() to persist them. Use the appropriate source_type:
   - "file" for local file paths
   - "repo" for GitHub/GitLab repository URLs
   - "url" for other URLs (Slack, Confluence, etc.)
   - "text" for inline text pasted in conversation

5. AUTO-SEARCH: When you're unsure about something or need context you don't have, call search()
   BEFORE saying you don't know. search() finds relevant information across ALL tasks and repos
   using semantic search. You don't need to know which task or repo — just describe what you need.
""",
    lifespan=_lifespan,
)


@mcp.tool()
def create_task(name: str, description: str = "", project: str = "") -> str:
    """Create a new task to group related context sources.

    Args:
        name: Short, unique name for the task (e.g., "auth-refactor")
        description: What this task is about
        project: Git remote URL to scope this task to a project. Claude should auto-detect this.
    """
    try:
        task = db.create_task(name, description, project)
        proj = f" (project: {project})" if project else ""
        return f"Created task '{task['name']}'{proj}"
    except ValueError as e:
        return f"Error: {e}"


@mcp.tool()
def list_tasks(project: str = "") -> str:
    """List all tasks, optionally filtered by project.

    Args:
        project: Git remote URL to filter tasks by project. If empty, lists all tasks.
    """
    tasks = db.list_tasks(project=project if project else None)
    if not tasks:
        scope = f" for project {project}" if project else ""
        return f"No tasks found{scope}. Create one with create_task()."

    lines = []
    for t in tasks:
        proj = f" [{t['project']}]" if t["project"] else ""
        lines.append(
            f"- {t['name']} ({t['source_count']} sources){proj}: {t['description']}"
        )
    return "\n".join(lines)


@mcp.tool()
def add_source(
    task_name: str = "", source_type: str = "", reference: str = "", notes: str = "", project: str = ""
) -> str:
    """Add a source to a task. Sources can be file paths, URLs, repo links, or inline text.

    Args:
        task_name: Name of the task to add the source to. If empty, auto-picks: the most
                   recent meeting-capture transcript (if modified within the last 10 minutes)
                   or `inbox-YYYY-MM-DD`. Auto-creates the task if it doesn't exist.
        source_type: One of "file", "repo", "url", "text". If empty, auto-detected from
                     reference (url/file/text).
        reference: The file path, URL, or inline text content.
        notes: Optional description of this source.
        project: Git remote URL to scope the task. Auto-detected by Claude per CLAUDE.md.
    """
    if not source_type:
        source_type = _detect_source_type(reference)

    valid_types = ("file", "repo", "url", "text")
    if source_type not in valid_types:
        return f"Error: source_type must be one of {valid_types}, got '{source_type}'"

    auto_task_note = ""
    if not task_name:
        task_name = _resolve_default_task_name(project)
        auto_task_note = f" (auto-task: {task_name})"

    task = db.get_task_by_name(task_name, project=project if project else None)
    if not task:
        # Auto-create the task. Use a description that signals auto-creation.
        try:
            task = db.create_task(
                task_name,
                description=f"Auto-created on {date.today().isoformat()} by add_source / drop",
                project=project,
            )
            auto_task_note = f" (created task: {task_name})"
        except ValueError as e:
            return f"Error auto-creating task '{task_name}': {e}"

    if source_type == "file":
        path = Path(reference).expanduser().resolve()
        if not path.exists():
            return f"Error: File not found: {path}"
        reference = str(path)

    try:
        source = db.add_source(task["id"], source_type, reference, notes)

        # Index in vector search
        if source_type == "text":
            vs.add(f"source:{source['id']}", reference, {
                "type": "source",
                "source_type": source_type,
                "task_name": task_name,
                "project": task.get("project", ""),
            })
        elif source_type == "file":
            index_file_content(
                vs, source["id"], reference,
                task_name=task_name,
                project=task.get("project", ""),
            )
        if notes:
            vs.add(f"source_notes:{source['id']}", notes, {
                "type": "source_notes",
                "source_type": source_type,
                "reference": reference,
                "task_name": task_name,
                "project": task.get("project", ""),
            })

        return (
            f"Added {source_type} source to task '{task_name}'{auto_task_note}: {reference}"
            + (f" ({notes})" if notes else "")
        )
    except ValueError as e:
        return f"Error: {e}"


@mcp.tool()
def drop(reference: str, notes: str = "", project: str = "") -> str:
    """Quick-save a source without thinking about task names or types.

    Use this when the user pastes a file path, URL, or chunk of text and wants it
    captured for later recall — typically during a meeting where they don't want to
    pause and create a task. The orchestrator picks the right task automatically:
    if a meeting transcript is currently being written (modified within the last 10
    minutes), the source attaches to that meeting; otherwise it goes into
    `inbox-YYYY-MM-DD` for triage later.

    Args:
        reference: A file path, URL, or inline text. Type is auto-detected.
        notes: Optional human description ("design doc Sarah presented", "Q3 plan").
        project: Optional git remote URL to scope to a project. Auto-detected per CLAUDE.md.
    """
    return add_source(
        task_name="",
        source_type="",
        reference=reference,
        notes=notes,
        project=project,
    )


@mcp.tool()
def get_task(task_name: str, project: str = "") -> str:
    """Get the full context manifest for a task — all sources plus repo knowledge for linked repos.

    Args:
        task_name: Name of the task
        project: Git remote URL to find the task in the right project
    """
    task = db.get_task_by_name(task_name, project=project if project else None)
    if not task:
        return f"Error: No task named '{task_name}'."

    sources = db.get_sources_for_task(task["id"])

    lines = [
        f"Task: {task['name']}",
        f"Description: {task['description']}",
    ]
    if task["project"]:
        lines.append(f"Project: {task['project']}")
    lines.append(f"Created: {task['created_at']}")
    lines.append("")

    if not sources:
        lines.append("No sources yet. Use add_source() to add files, repos, URLs, or text.")
    else:
        lines.append(f"Sources ({len(sources)}):")
        for s in sources:
            note = f" — {s['notes']}" if s["notes"] else ""
            lines.append(f"  [{s['id']}] ({s['source_type']}) {s['reference']}{note}")

    # Auto-include repo knowledge for linked repos
    repo_sources = [s for s in sources if s["source_type"] == "repo"]
    for repo_source in repo_sources:
        knowledge = db.get_repo_knowledge(repo_source["reference"])
        if knowledge:
            lines.append("")
            lines.append(f"Repo knowledge ({repo_source['reference']}):")
            for k in knowledge:
                lines.append(f"  - {k['insight']}")

    return "\n".join(lines)


@mcp.tool()
def remove_source(task_name: str, source_id: int, project: str = "") -> str:
    """Remove a source from a task by its ID.

    Args:
        task_name: Name of the task (for verification)
        source_id: The numeric ID of the source to remove (shown in get_task output)
        project: Git remote URL to find the task in the right project
    """
    task = db.get_task_by_name(task_name, project=project if project else None)
    if not task:
        return f"Error: No task named '{task_name}'."

    sources = db.get_sources_for_task(task["id"])
    source_ids = {s["id"] for s in sources}
    if source_id not in source_ids:
        return f"Error: Source {source_id} does not belong to task '{task_name}'."

    if db.remove_source(source_id):
        vs.remove(f"source:{source_id}")
        vs.remove(f"source_notes:{source_id}")
        # Clean up file chunks (try up to 100 chunks)
        for i in range(100):
            vs.remove(f"file_chunk:{source_id}:{i}")
        return f"Removed source {source_id} from task '{task_name}'."
    return f"Error: Source {source_id} not found."


@mcp.tool()
def update_repo_knowledge(repo_url: str, insight: str) -> str:
    """Record a learning about a repository — setup steps, test commands, conventions, gotchas.

    Call this proactively whenever you discover something useful about a repo.
    Knowledge accumulates over time and is automatically shown when any task references this repo.

    Args:
        repo_url: The repository URL (e.g., "https://github.com/org/repo")
        insight: A single piece of knowledge (e.g., "Needs Node 18", "Run tests with pytest -x")
    """
    entry = db.update_repo_knowledge(repo_url, insight)
    vs.add(f"repo_knowledge:{entry['id']}", insight, {
        "type": "repo_knowledge",
        "repo_url": repo_url,
    })
    return f"Saved repo knowledge for {repo_url}: {insight}"


@mcp.tool()
def get_repo_knowledge(repo_url: str) -> str:
    """Get all stored knowledge for a repository.

    Args:
        repo_url: The repository URL
    """
    knowledge = db.get_repo_knowledge(repo_url)
    if not knowledge:
        return f"No knowledge stored for {repo_url}."

    lines = [f"Repo knowledge ({repo_url}):"]
    for k in knowledge:
        lines.append(f"  - {k['insight']} (added {k['created_at']})")
    return "\n".join(lines)


def _parse_iso_date(s: str):
    """Parse a date or datetime string to a unix timestamp. Accepts:
        - "2026-04-30"           (date only — interpreted as 00:00 UTC)
        - "2026-04-30T14:25:00"  (no tz — interpreted as UTC)
        - "2026-04-30T14:25:00+00:00"  (full ISO 8601)
    Returns None if unparseable.
    """
    from datetime import datetime, timezone
    try:
        if "T" not in s:
            s = s + "T00:00:00"
        if not s.endswith("Z") and "+" not in s and "-" not in s.split("T", 1)[1]:
            s = s + "+00:00"
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except (ValueError, IndexError):
        return None


@mcp.tool()
def search(
    query: str,
    project: str = "",
    after_date: str = "",
    before_date: str = "",
    meeting_id: str = "",
    rerank: bool = False,
) -> str:
    """Semantic search across ALL stored knowledge — task sources, repo knowledge,
    transcript chunks, and notes.

    The retrieval pipeline is BM25 + dense vector hybrid (Reciprocal Rank Fusion)
    re-ranked with MMR for diversity, so proper-noun queries (engineer names,
    product codenames) work as well as paraphrased questions.

    Args:
        query: Natural language query (e.g., "what did we decide about Megatune",
            "Gang Chu shadow queue context", "Lyft Teen brand awareness").
            Proper nouns are matched lexically AND semantically.
        project: Optional git remote URL to limit search to a project's tasks.
            Falls back to global search if no results.
        after_date: Optional ISO date/datetime — only return chunks captured at
            or after this point. Examples: "2026-04-29", "2026-04-30T14:00:00".
            Useful for "what happened in the last meeting" / "this week" queries.
        before_date: Optional ISO date/datetime — only return chunks captured at
            or before this point. Pair with `after_date` for time-window queries.
        meeting_id: Optional exact match on a transcript filename stem (without
            .md). Use when you know which meeting holds the answer and want
            chunks scoped to it. Example: "meeting-2026-04-30T13-40-35".
        rerank: When True, send the top candidates through an LLM (configured
            via the CO_RERANK_MODEL env var, e.g. "gemini-flash-latest") for
            relevance scoring and re-rank by score. Adds 1-3s latency and a
            small per-query cost, but corrects cases where vector search
            surfaces tangentially-related chunks instead of true matches —
            and crucially can recognise when no chunk in the corpus actually
            answers the query. Use for important queries; skip for
            speculative or exploratory ones.

    Project filter and time-window filter are applied via Chroma's native
    metadata `where` clause — they constrain the dense retriever. When any
    filter is specified, hybrid mode is disabled (BM25 is unfiltered).
    """
    # Pick up meetings that ended since the last look. Never blocks behind
    # another process's indexing run (catch_up returns immediately then).
    _index_new_transcripts()
    vs.reload()  # pick up writes from the watcher / other processes

    # Build metadata filter
    filters = []
    if project:
        filters.append({"project": project})
    if after_date:
        ts = _parse_iso_date(after_date)
        if ts is not None:
            filters.append({"start_ts_unix": {"$gte": ts}})
    if before_date:
        ts = _parse_iso_date(before_date)
        if ts is not None:
            filters.append({"start_ts_unix": {"$lte": ts}})
    if meeting_id:
        filters.append({"meeting_id": meeting_id})

    where = None
    if len(filters) == 1:
        where = filters[0]
    elif len(filters) > 1:
        where = {"$and": filters}

    # When filters are in play, hybrid is incompatible (BM25 is unfiltered).
    # Otherwise default to hybrid + MMR for best general-purpose ranking.
    use_hybrid = where is None
    hits = vs.search(
        query,
        n_results=10,
        where=where,
        hybrid=use_hybrid,
        mmr=True,
        mmr_lambda=0.7,
        rerank=rerank,
    )
    if not hits:
        # Fallback: drop the project filter (repo knowledge isn't project-scoped)
        if project and where:
            other_filters = [f for f in filters if "project" not in f]
            fallback_where = (
                None if not other_filters
                else other_filters[0] if len(other_filters) == 1
                else {"$and": other_filters}
            )
            hits = vs.search(
                query,
                n_results=10,
                where=fallback_where,
                hybrid=fallback_where is None,
                mmr=True,
                mmr_lambda=0.7,
                rerank=rerank,
            )
        if not hits:
            return f"No results for '{query}'."

    lines = [f"Search results for '{query}':"]
    for h in hits:
        meta = h["metadata"]
        doc_type = meta.get("type", "unknown")

        if doc_type == "repo_knowledge":
            lines.append(f"  [repo: {meta.get('repo_url', '?')}] {h['text']}")
        elif doc_type == "file_chunk":
            chunk_info = f"chunk {meta.get('chunk_index', '?')}/{meta.get('total_chunks', '?')}"
            lines.append(
                f"  [task: {meta.get('task_name', '?')}] {meta.get('filename', '?')} ({chunk_info}):\n"
                f"    {h['text'][:200]}..."
                f"\n    → Full file: {meta.get('file_path', '?')}"
            )
        elif doc_type == "source":
            lines.append(f"  [task: {meta.get('task_name', '?')}] ({meta.get('source_type', '?')}) {h['text'][:200]}")
        elif doc_type == "source_notes":
            lines.append(f"  [task: {meta.get('task_name', '?')}] note on {meta.get('reference', '?')}: {h['text']}")
        elif doc_type == "transcript":
            ts = meta.get("start_ts_iso", "")
            ts_prefix = f"{ts} " if ts else ""
            lines.append(
                f"  [transcript: {meta.get('meeting_id', meta.get('filename', '?'))}] "
                f"{ts_prefix}{h['text'][:200]}"
            )
        else:
            lines.append(f"  [{doc_type}] {h['text'][:200]}")

    return "\n".join(lines)


def main():
    logger.info(f"Starting context-orchestrator, db={db.db_path}")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
