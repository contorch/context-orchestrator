<p align="left">
  <img src="assets/contorch-orange.png" width="96" alt="Contorch logo">
</p>

# context-orchestrator

> Part of **Contorch** — the persistent context layer for Claude Code. Captures meetings, indexes everything, carries the torch from one Claude session to the next.

MCP server that gives Claude persistent memory of your tasks, sources, and repo knowledge across sessions, with a vector index over everything you've captured.

Pairs with [meeting-capture](https://github.com/contorch/meeting-capture) — that daemon writes meeting transcripts straight into this project's database (`~/.context-orchestrator/context.db`, table `transcripts`), and the MCP server indexes each meeting once it ends. There are no transcript files. Either project runs without the other.

## Requirements

- macOS or Linux
- Python 3.10+
- Claude Code (connected by `contorch-memory claude install`, see below)

## Install

### Recommended — full stack, one command

Installs context-orchestrator + [meeting-capture](https://github.com/contorch/meeting-capture) + [pipeline-monitor](https://github.com/contorch/pipeline-monitor) + the auto-context Claude Code hook + (optional) Gemini activation:

```bash
curl -fsSL https://raw.githubusercontent.com/contorch/context-orchestrator/main/bootstrap.sh | bash
```

The script is idempotent (safe to re-run), installs missing prereqs via Homebrew, prompts for a Gemini API key when it needs one (skip with empty input to stay all-local), and finishes with a clear punch-list of the few things macOS requires you to click yourself: restarting Claude Code so it re-reads `~/.claude/settings.json`, and granting Microphone + Screen Recording permissions in System Settings.

### Just this repo

```bash
git clone https://github.com/contorch/context-orchestrator.git
cd context-orchestrator
./setup.sh
```

`setup.sh` checks prerequisites, creates a Python venv, installs the chroma HTTP server launchd agent (with a one-time backup of any existing chroma data), and connects Claude Code with `contorch-memory claude install --no-hook`; `install-claude-context.sh` runs the same command with the auto-context hook.

### Lightweight — database + index only, no daemons

For a machine that only needs the memory (e.g. a work laptop): SQLite for the transcripts and tasks, ChromaDB opened in-process from a folder, the MCP server Claude Code starts per session, and nothing running in the background — no chroma server, no watcher, no meeting-capture.

```bash
python3 -m venv ~/.contorch-lite                     # Python 3.10+
~/.contorch-lite/bin/pip install "context-orchestrator[embeddings-gemini] @ git+https://github.com/contorch/context-orchestrator"
mkdir -p ~/.context-orchestrator
echo "CO_EMBEDDING_MODEL=gemini-embedding-001" >> ~/.context-orchestrator/env   # must match the machine that embeds
~/.contorch-lite/bin/contorch-memory claude install   # MCP server, hook, CLAUDE.md block, transcripts skill
```

With no chroma launchd agent installed, the index is opened from `~/.context-orchestrator/chroma/` directly. Several processes share it (one MCP server per Claude Code session, the CLIs, the auto-context hook), and chromadb is not process-safe on a shared folder, so every operation runs inside one cross-process lock (`~/.context-orchestrator/chroma.lock`): open fresh, operate, close. Embeddings are computed before the lock is taken; the hook waits at most 2 s for it and otherwise answers from full-text search. chromadb is pinned (1.5.9) because the lock was proven on that version — see `tests/test_chroma_concurrency.py` (`pytest -m slow`). A Gemini key is optional here: without one, vectors come from embedding bundles made on another machine (`contorch-transcripts embed … -o bundle.jsonl` there, `contorch-transcripts import bundle.jsonl` here) and `search` falls back to keyword matching for the query. With a key (`~/.config/google/key`), transcripts are embedded locally and search is semantic.

After any install path, restart Claude Code so the new MCP server is loaded.

To verify:

```bash
.venv/bin/transcript-watcher doctor
```

## MCP tools

| Tool | Purpose |
|---|---|
| `create_task` | Create a task scoped to a project (git remote URL) |
| `list_tasks` | List tasks, optionally filtered by project |
| `get_task` | Get all sources and repo knowledge attached to a task |
| `add_source` | Add a file path, repo URL, link, or inline text to a task |
| `drop` | Quick-save a source without naming a task — auto-attaches to the active meeting or daily inbox |
| `remove_source` | Remove a source from a task |
| `update_repo_knowledge` | Save a learning about a repo (setup, testing, conventions, gotchas) |
| `get_repo_knowledge` | Get all stored knowledge for a repo |
| `search` | Semantic search across all tasks, sources, transcripts, and repo knowledge (keyword fallback when the query can't be embedded) |
| `add_transcript` | Store a transcript (text, WebVTT or SRT) in the database and index it |
| `get_transcript` | Full text of a stored transcript |
| `list_transcripts` | Most recent transcripts |

## Components

### MCP server

Long-lived process spawned by Claude Code on session start. Exposes the tools above. Uses a local SQLite database at `~/.context-orchestrator/context.db` for tasks, sources and transcripts, and ChromaDB for the vector index — via the chroma HTTP server when its launchd agent is installed, otherwise in-process from `~/.context-orchestrator/chroma/` (`CO_CHROMA_PATH` overrides).

### Claude Code integration

`contorch-memory claude install|uninstall|status [--channel app|brew|dev] [--no-hook] [--backup-dir D] [--json]` is the only thing that touches Claude Code:

- the MCP server `context-orchestrator` (user scope, `claude mcp add`), pointing at this install's own `contorch-mcp`; only known `CO_*` env keys are kept, and `CO_CHROMA_HOST/PORT` are dropped when the index is in-process;
- one `UserPromptSubmit` hook entry, guarded and tagged — `[ -x '<path>/contorch-hook' ] && '<path>/contorch-hook'; exit 0 # contorch-hook:<channel>` with a 10 s timeout — so a removed install is silent instead of erroring on every prompt;
- one marked block in `CLAUDE.md` (`<!-- contorch -->` … `<!-- /contorch -->`);
- the `transcripts` skill, written with this install's absolute `contorch-transcripts` path and a `.contorch-skill.json` stamp.

Older installs are migrated: a copied `~/.claude/hooks/auto-context.py`, the old `CONTEXT-ORCHESTRATOR` / `auto-context-section` CLAUDE.md blocks and a curl'ed skill are replaced only when they are byte-identical to a version we shipped; edited copies are kept and reported as a to-do. `uninstall` removes only what carries Contorch's tag, marker or stamp. Claude Code's files follow `CLAUDE_CONFIG_DIR`; managed settings that block hooks or MCP servers are detected and reported (`blocked_by_managed_settings`). Running the same install twice changes nothing.

### Memory status, self-test and data operations (`contorch-memory`)

Every `--json` form prints one JSON document on stdout (human text goes to stderr):

| Command | What it does |
|---|---|
| `status --json [--deep]` | `contorch-memory.status/1`: embeddings, `vector_index` (`server` / `in_process` / `none`), docs, transcripts, pending, `chromadb_version`, `index_written_by`, `index_compatible`. Doesn't import chromadb unless `--deep`. |
| `selftest --json` | Writes, searches and deletes a marker document. Reports the stage reached and an error code (`offline`, `proxy`, `tls`, `key`, `quota`, `busy`, `db`). |
| `where --json` | Absolute paths of this install's `contorch-mcp`, `contorch-hook`, `contorch-transcripts`, the skill, the DB and the index. |
| `backup --to DIR [--stop-server] --json` | `context.db` through SQLite's backup API (integrity check + FTS counts), and the index copied under the session lock. The copy is reopened and every collection is counted; a copy that doesn't verify is `backup_unverified`. Refuses `server_running` unless `--stop-server`. |
| `restore --from DIR [--stop-server] --json` | Puts a backup's index back (the current one is moved aside) and verifies the counts. `context.db` is never rolled back. |
| `index migrate --in-process [--backup-dir DIR] --json` | Retires the chroma server: backs up with the server stopped, removes its agent, and counts the same folder in-process. |

### chroma server

The vector index is served by a `chroma run` HTTP daemon (launchd-managed) listening on `127.0.0.1:8765`. The MCP server, the watcher, and one-shot CLIs all connect as HTTP clients, which avoids the SQLite-lock contention that `PersistentClient`-per-process configurations are prone to under concurrent access. Override the host or port via the `CO_CHROMA_HOST` / `CO_CHROMA_PORT` environment variables. CLI:

| Command | Purpose |
|---|---|
| `context-orchestrator-chroma status` | Plist state, port listening check, HTTP heartbeat |
| `context-orchestrator-chroma install` | Install and start the launchd agent |
| `context-orchestrator-chroma uninstall` | Stop and remove the launchd agent |

### Search: full-text always, embeddings by choice

`search` combines two retrievers. **Full-text search** (SQLite FTS5, English stemming) is always on: every transcript, repo insight and text source is keyword-searchable the moment it is stored, with no model, key or network — kept current by triggers, including meeting-capture's live appends. **Embeddings** add meaning-based matching, and are a setup choice:

| `contorch-memory embeddings …` | Understands | Needs |
|---|---|---|
| `gemini` | paraphrases best | a Gemini key; text is sent to Google to embed |
| `local` | paraphrases, less well | nothing — all-MiniLM-L6-v2 runs on this Mac (~80 MB, downloaded once) |
| `none` | exact words (stemmed) | nothing — nothing leaves this Mac |

The choice is stored in `~/.context-orchestrator/env`; `contorch-memory status` shows it with counts. Each model gets its own Chroma collection and each transcript records which model embedded it, so switching re-embeds from the stored text (next search, or `contorch-transcripts reindex`) and switching back reuses the old vectors. Machines exchanging embedding bundles must use the same choice. The Contorch menu bar shows the mode on its Index line.

### Transcripts

Every transcript is a row in the `transcripts` table (full text in `body`); Chroma holds only the index built from it. The text is stored before anything is embedded, so a failing embedding call (bad key, quota, offline) loses nothing — the row stays pending and is indexed on a later `search`. A meeting still being appended to by meeting-capture is indexed once it has been quiet for 60 s.

| Command | Purpose |
|---|---|
| `contorch-transcripts add FILE\|- --title T --started-at ISO --source URL` | Add one transcript (.txt/.md/.vtt/.srt; captions are put on the wall clock) |
| `contorch-transcripts import PATH [--delete]` | Import a file, folder or zip (read in memory), or an embedding bundle; `--delete` first zips the files into `~/.context-orchestrator/backups/transcripts-<time>.zip`, verifies the zip byte-for-byte, then removes them (nothing is deleted if the backup fails) |
| `contorch-transcripts embed PATH -o bundle.jsonl` | On a machine with a Gemini key: chunk + embed into a bundle to import elsewhere without API calls |
| `contorch-transcripts export OUT.jsonl --pending` | Text of not-yet-embedded transcripts, to `embed` on another machine |
| `contorch-transcripts list \| show ID \| rm ID \| reindex [ID]` | Inspect, delete (the text is kept in `backups/deleted-transcripts.zip` first), re-embed |

A bundle records the embedding function and vector size; `import` loads the vectors only when they match this index, otherwise it stores the text and embeds it locally.

The `transcripts` Claude Code skill (installed by `contorch-memory claude install`) wraps all of this: give Claude a URL, a pasted transcript, a zip or a bundle and it stores it verbatim.

Moving from the old `~/transcripts/*.md` files: `contorch-transcripts import ~/transcripts --delete`.

### transcript-watcher

Optional. Transcripts are indexed on demand by the MCP server; the watcher (and the MCP server's catch-up) also imports any `.md` still dropped into `~/transcripts/` by an older meeting-capture into the database. CLI:

| Command | Purpose |
|---|---|
| `transcript-watcher status` | Daemon state, watch dir, indexed file count |
| `transcript-watcher doctor` | Full health check |
| `transcript-watcher run` | Run the watch loop in the foreground |
| `transcript-watcher once` | Single-pass scan and exit (cron-friendly) |
| `transcript-watcher install` | Install the launchd auto-start agent |
| `transcript-watcher uninstall` | Remove the launchd auto-start agent |

### save-transcript

Manual transcript capture utility. Stores the current clipboard as a transcript in the database and indexes it (`--file PATH` stores a file instead).

## Storage

- `~/.context-orchestrator/context.db` — SQLite (tasks, sources, repo knowledge, transcripts)
- `~/.context-orchestrator/env` — optional `KEY=VALUE` settings read by every entry point (environment wins)
- `~/.context-orchestrator/chroma/` — ChromaDB vector index, served by the chroma daemon (uses the default `all-MiniLM-L6-v2` embeddings; no API key required)
- `~/.context-orchestrator/chroma-daemon.log` — chroma server stdout / stderr
- `~/.context-orchestrator/watcher_state.json` — transcript-watcher mtime cache
- `~/.context-orchestrator/chroma.backup-<timestamp>/` — one-time backup created by `setup.sh` before first switching an existing install to the HTTP server

All local. No cloud sync. Each machine has its own independent state.

## Typical workflows

### Loading task context

```
You: "I'm picking up the auth-refactor work"

Claude → get_task("auth-refactor") →

  Task: auth-refactor
  Sources:
    [1] (file)  ~/docs/auth-spec.md      Auth spec v2
    [2] (repo)  github.com/org/backend   Main backend
    [3] (text)  Meeting summary 2026-04-22

  Repo knowledge for github.com/org/backend:
    - Setup: docker-compose up
    - Tests: pytest -x
    - Uses FastAPI + SQLAlchemy
```

Claude then reads the files and clones the repo using its built-in tools.

### Quick-saving during a meeting

```
You: "drop ~/Downloads/q3-design.pdf, presenter is Sarah"

Claude → drop("~/Downloads/q3-design.pdf", notes="presenter is Sarah") →

  Added file source to task 'meeting-2026-04-26T18-43-05'
  (auto-task: meeting-2026-04-26T18-43-05): /Users/.../q3-design.pdf
```

The PDF is auto-linked to the in-progress meeting transcript. If no meeting is active, it goes to `inbox-YYYY-MM-DD`.

### Recall

```
You: "what was that thing about Q3 architecture Sarah was talking about"

Claude → search("Q3 architecture Sarah") →

  [transcript chunk] meeting-2026-04-26T...md  "...Sarah explained the Q3..."
  [file chunk]       q3-design.pdf             "...service architecture for Q3..."
  [source_notes]                               "presenter is Sarah"
```

Claude reads the relevant files via the returned references and answers.

## Tests

```bash
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
```

## Architecture notes

- **Project scoping** uses the git remote URL as a stable identifier. Tasks are unique within `(name, project)` so the same task name can exist in different repos without collision.
- **Repo knowledge is append-only.** Calls to `update_repo_knowledge` accumulate insights rather than overwrite, so context grows organically as Claude discovers things.
- **The vector index runs as a separate HTTP service** so the watcher, the MCP server, and one-shot CLIs share a single writer. The on-disk format is identical to `chromadb.PersistentClient`, so existing data carries over without migration; `setup.sh` makes a one-time backup before switching an existing install to the server.
- **The transcript-watcher uses a 2-second settle window** to avoid reading meeting-capture's `.md` files mid-write.
