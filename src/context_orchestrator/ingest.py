"""Index file content into ChromaDB for search."""

import logging
from pathlib import Path

from context_orchestrator.chunking import chunk_text
from context_orchestrator.search import VectorSearch

logger = logging.getLogger("context-orchestrator")


def index_file_content(vs: VectorSearch, source_id: int, file_path: str,
                       task_name: str = "", project: str = "") -> int:
    """Read a file, chunk it, and index all chunks in ChromaDB.

    Used by the MCP server when a file source is added to a task.
    Returns the number of chunks indexed.
    """
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        return 0

    try:
        content = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, PermissionError):
        return 0

    chunks = chunk_text(content)

    # One upsert (one Chroma session) for the whole file.
    vs.upsert(
        [f"file_chunk:{source_id}:{i}" for i in range(len(chunks))],
        list(chunks),
        [{
            "type": "file_chunk",
            "source_id": str(source_id),
            "chunk_index": i,
            "total_chunks": len(chunks),
            "file_path": file_path,
            "filename": path.name,
            "task_name": task_name,
            "project": project,
        } for i in range(len(chunks))],
    )

    return len(chunks)
