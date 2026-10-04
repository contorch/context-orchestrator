"""One cross-process lock around every in-process Chroma session.

chromadb's PersistentClient is not process-safe (Chroma's own docs: "not
process-safe for concurrent writers sharing the same local persistence
path"). On a lightweight install several processes open the same folder at
once — one MCP server per Claude Code session, the CLIs, the per-prompt hook —
and the lab (contorch-macos design, phase1-modules-both-channels §2) measured
what happens on chromadb 1.5.9 without coordination:

  * up to 147 of 2662 documents silently lost their vector (metadata kept,
    "Error finding id" on read), deletes came back, and long-lived readers
    never saw other processes' writes, even after reload();
  * an exclusive lock around writes only was NOT enough (11 lost, one reader
    died with SIGBUS): opening and reading also map/persist the HNSW files;
  * one exclusive lock around every session — clear chromadb's per-path
    System cache, open fresh, operate, drop, clear, unlock — gave 0 lost,
    0 resurrected, 0 crashes, at ~130 ms per uncontended operation.

This module is that lock. It has no chromadb import, so the hook and
`contorch-memory status` can use it without loading Chroma.

The lock is an fcntl.flock on `<chroma folder>.lock` — for the default folder
that is ~/.context-orchestrator/chroma.lock. It is re-entrant inside one
process (a nested session reuses the open one), serialised between threads
of one process, and released by the kernel if the process dies.
"""
from __future__ import annotations

import contextlib
import fcntl
import threading
import time
from pathlib import Path
from typing import Iterator, Optional

# How long an ordinary caller (MCP server tool, CLI) waits for the lock before
# giving up with LockTimeout. Sessions are short (one operation each), so a
# long wait means something is wrong; the caller then degrades (keyword
# search) instead of hanging Claude Code.
DEFAULT_TIMEOUT_S = 60.0
# The per-prompt hook must never hold up a prompt: it waits this long, then
# answers from SQLite full-text search alone.
HOOK_TIMEOUT_S = 2.0

_POLL_S = 0.02


class LockTimeout(TimeoutError):
    """The Chroma session lock could not be taken in time."""


def lock_path_for(chroma_path: Path) -> Path:
    """`<folder>.lock` next to the Chroma folder (never inside it, so backups
    and wipes of the folder don't touch the lock)."""
    p = Path(chroma_path)
    return p.parent / (p.name + ".lock")


class _Held:
    __slots__ = ("fh", "depth", "owner")

    def __init__(self, fh, owner: int):
        self.fh = fh
        self.depth = 1
        self.owner = owner


# One RLock for the whole process: chromadb's System cache is process-global
# (SharedSystemClient.clear_system_cache drops every path's System), so two
# threads must never be inside sessions at once, whatever the path.
_THREAD_LOCK = threading.RLock()
_held: dict[str, _Held] = {}


def is_held(lock_file: Path) -> bool:
    """True when this thread holds the lock (used by tests and assertions)."""
    h = _held.get(str(lock_file))
    return h is not None and h.owner == threading.get_ident()


@contextlib.contextmanager
def session_lock(lock_file: Path, timeout: Optional[float] = DEFAULT_TIMEOUT_S,
                 on_first_acquire=None, on_last_release=None) -> Iterator[bool]:
    """Hold the exclusive cross-process lock for the duration of the block.

    Yields True for the outermost holder, False for a nested (re-entrant)
    one. `on_first_acquire` / `on_last_release` run only for the outermost
    holder, while the lock is held — that is where the chromadb System cache
    is cleared. Raises LockTimeout when `timeout` seconds pass first
    (None = wait forever)."""
    key = str(lock_file)
    deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
    if not _THREAD_LOCK.acquire(timeout=-1 if timeout is None else max(0.0, timeout)):
        raise LockTimeout(f"another thread holds the Chroma lock ({key})")
    try:
        held = _held.get(key)
        if held is not None:
            held.depth += 1
            try:
                yield False
            finally:
                held.depth -= 1
            return
        Path(lock_file).parent.mkdir(parents=True, exist_ok=True)
        fh = open(lock_file, "a")
        try:
            while True:
                try:
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if deadline is not None and time.monotonic() >= deadline:
                        raise LockTimeout(
                            f"Chroma is busy in another process (waited {timeout:.1f}s for {key})")
                    time.sleep(_POLL_S)
            _held[key] = _Held(fh, threading.get_ident())
            try:
                if on_first_acquire is not None:
                    on_first_acquire()
                yield True
            finally:
                try:
                    if on_last_release is not None:
                        on_last_release()
                finally:
                    del _held[key]
                    fcntl.flock(fh, fcntl.LOCK_UN)
        finally:
            fh.close()
    finally:
        _THREAD_LOCK.release()


def lock_info(lock_file: Path) -> dict:
    """Non-blocking probe for status output: is anyone holding it right now?"""
    if not Path(lock_file).exists():
        return {"path": str(lock_file), "busy": False}
    if is_held(lock_file):
        return {"path": str(lock_file), "busy": True}
    try:
        with open(lock_file, "a") as fh:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return {"path": str(lock_file), "busy": True}
            fcntl.flock(fh, fcntl.LOCK_UN)
    except OSError:
        pass
    return {"path": str(lock_file), "busy": False}
