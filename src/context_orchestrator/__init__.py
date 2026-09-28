"""context-orchestrator.

Settings: environment variables win; any not set are read from
~/.context-orchestrator/env (KEY=VALUE lines, '#' comments), so the MCP
server, the CLIs and hooks all agree without per-process configuration —
e.g. CO_EMBEDDING_MODEL=gemini-embedding-001 on a machine without a key
that only imports embedding bundles.
"""
import os
from pathlib import Path

ENV_FILE = Path.home() / ".context-orchestrator" / "env"


def load_env_file(path: Path = ENV_FILE) -> dict:
    loaded = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return loaded
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip().removeprefix("export ").strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


load_env_file()
