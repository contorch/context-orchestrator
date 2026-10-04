<!-- contorch -->
## Contorch memory (context-orchestrator)

The `context-orchestrator` MCP server is this machine's persistent memory: tasks, sources, repo knowledge and meeting transcripts, kept across sessions.

- When I mention working on a task, call `get_task()` to load its context before doing anything else.
- When creating or listing tasks, detect the project with `git remote get-url origin` and pass it as `project`.
- When I paste links, file paths or text for a task, call `add_source()`; without a task (e.g. during a meeting) call `drop()`.
- When you learn something about a repo (setup, test commands, conventions, gotchas), call `update_repo_knowledge()` without asking.
- Before saying you don't know something, call `search()`. Meeting transcripts: `get_transcript(meeting_id)` reads a whole meeting; the `transcripts` skill stores new ones.
- An `[auto-context]` block at the top of a prompt was pre-loaded by Contorch's hook (top search hits and git state): use what is relevant and don't repeat those lookups.
<!-- /contorch -->
