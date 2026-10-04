"""Keep an MCP stdio session alive across a code swap underneath it.

Claude Code starts one `contorch-mcp` per session and keeps it for hours.
Meanwhile the code it was started from can be replaced: Homebrew rebuilds the
venv on upgrade, Sparkle swaps the whole app bundle. The old process then
runs half-old, half-missing code (lazy imports fail) until the user restarts
Claude Code.

The guard (proven in the contorch-macos lab, rumps-in-bundle-quit-updates
mods/lab_mcp_server.py):

1. reads fd 0 itself, into its own buffer, so it always knows exactly which
   bytes have been received but not yet dispatched;
2. before dispatching each message, compares the identity (st_dev, st_ino)
   of `lstat(sys.executable)` with the one seen at start;
3. on a change, waits until no request is in flight, then `execve`s the same
   command. execve keeps the PID and the stdio file descriptors, so Claude
   Code's pipe stays connected. The new image replays the undispatched bytes
   first and runs the session `stateless=True` (already initialized — the MCP
   SDK's own flag), so the client never notices.

A missing executable (mid-swap) is "not changed yet"; the next message checks
again. CONTORCH_MCP_GUARD=0 turns the guard off (plain stdio transport).
"""
from __future__ import annotations

import base64
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

log = logging.getLogger("context-orchestrator")

ENV_RESUMED = "CONTORCH_MCP_RESUMED"
ENV_PENDING = "CONTORCH_MCP_PENDING"            # base64 of the undispatched bytes
ENV_PENDING_FILE = "CONTORCH_MCP_PENDING_FILE"  # …or a 0600 file when they are large
ENV_WATCH = "CONTORCH_MCP_GUARD_WATCH"          # tests: watch this path instead of sys.executable
ENV_DISABLE = "CONTORCH_MCP_GUARD"
_INLINE_LIMIT = 32 * 1024                       # keep the environment small


def _identity(path: str) -> Optional[tuple[int, int]]:
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return None
    return (st.st_dev, st.st_ino)


def _take_pending() -> bytes:
    data = base64.b64decode(os.environ.pop(ENV_PENDING, "") or b"")
    f = os.environ.pop(ENV_PENDING_FILE, "")
    if f:
        try:
            data += Path(f).read_bytes()
        finally:
            Path(f).unlink(missing_ok=True)
    return data


def _pending_env(buf: bytes) -> dict:
    if len(buf) <= _INLINE_LIMIT:
        return {ENV_PENDING: base64.b64encode(buf).decode()}
    fd, path = tempfile.mkstemp(prefix="contorch-mcp-pending-")
    with os.fdopen(fd, "wb") as fh:
        fh.write(buf)
    return {ENV_PENDING_FILE: path}


def default_argv() -> list[str]:
    """The command the new image runs: same interpreter, same module, same
    arguments (works whether we were started as `contorch-mcp` or `-m`)."""
    return [sys.executable, "-m", "context_orchestrator.server", *sys.argv[1:]]


def run(mcp, argv: Optional[list[str]] = None) -> None:
    """Serve `mcp` (a FastMCP) over stdio with the re-exec guard."""
    if os.environ.get(ENV_DISABLE, "1") == "0":
        mcp.run(transport="stdio")
        return
    import anyio
    anyio.run(_serve, mcp, argv or default_argv())


async def _serve(mcp, argv: list[str]) -> None:
    import anyio
    import mcp.types as types
    from mcp.shared.message import SessionMessage

    resumed = os.environ.pop(ENV_RESUMED, None) == "1"
    buf = bytearray(_take_pending())
    watch = os.environ.get(ENV_WATCH) or sys.executable
    start = _identity(watch)
    inflight: set = set()
    out = sys.stdout.buffer
    read_w, read_r = anyio.create_memory_object_stream(0)
    write_w, write_r = anyio.create_memory_object_stream(0)
    if resumed:
        log.info("stdio guard: resumed in a new image (pid %d), replaying %d byte(s)",
                 os.getpid(), len(buf))

    def changed() -> bool:
        now = _identity(watch)
        return start is not None and now is not None and now != start

    async def reader():
        async with read_w:
            while True:
                while b"\n" not in buf:
                    chunk = await anyio.to_thread.run_sync(os.read, 0, 65536)
                    if not chunk:
                        return
                    buf.extend(chunk)
                if changed():
                    while inflight:
                        await anyio.sleep(0.01)
                    log.info("stdio guard: %s was replaced — re-exec (pid %d, %d undispatched byte(s))",
                             watch, os.getpid(), len(buf))
                    out.flush()
                    sys.stderr.flush()
                    env = dict(os.environ, **{ENV_RESUMED: "1"}, **_pending_env(bytes(buf)))
                    os.execve(argv[0], argv, env)
                line, _, _ = bytes(buf).partition(b"\n")
                del buf[:len(line) + 1]
                if not line.strip():
                    continue
                try:
                    msg = types.JSONRPCMessage.model_validate_json(line)
                except Exception as exc:  # the SDK's stdio transport forwards these too
                    await read_w.send(exc)
                    continue
                if isinstance(msg.root, types.JSONRPCRequest):
                    inflight.add(msg.root.id)
                await read_w.send(SessionMessage(msg))

    async def writer():
        async with write_r:
            async for sm in write_r:
                data = sm.message.model_dump_json(by_alias=True, exclude_none=True).encode() + b"\n"

                def _write(d=data):
                    out.write(d)
                    out.flush()
                await anyio.to_thread.run_sync(_write)
                if isinstance(sm.message.root, (types.JSONRPCResponse, types.JSONRPCError)):
                    inflight.discard(sm.message.root.id)

    server = mcp._mcp_server
    async with anyio.create_task_group() as tg:
        tg.start_soon(reader)
        tg.start_soon(writer)
        await server.run(read_r, write_w, server.create_initialization_options(), stateless=resumed)
        tg.cancel_scope.cancel()
