"""The MCP stdio re-exec guard (stdio_guard.py).

A fake "code swap": the guard watches CONTORCH_MCP_GUARD_WATCH (a file the test
replaces with os.replace → new inode) instead of sys.executable. The client
here speaks raw JSON-RPC over the pipes, like Claude Code.
"""
import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"


class Client:
    def __init__(self, argv, env):
        self.p = subprocess.Popen(argv, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
        self.responses: dict = {}
        self.order: list = []
        self.cv = threading.Condition()
        self.stderr = bytearray()
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._read_err, daemon=True).start()
        self.next_id = 0

    def _read(self):
        for line in self.p.stdout:
            try:
                m = json.loads(line)
            except ValueError:
                continue
            if "id" in m and ("result" in m or "error" in m):
                with self.cv:
                    self.responses[m["id"]] = m
                    self.order.append(m["id"])
                    self.cv.notify_all()

    def _read_err(self):
        for line in self.p.stderr:
            self.stderr.extend(line)

    def send(self, *msgs):
        data = b"".join(json.dumps(m).encode() + b"\n" for m in msgs)
        self.p.stdin.write(data)
        self.p.stdin.flush()

    def request(self, method, params=None):
        self.next_id += 1
        m = {"jsonrpc": "2.0", "id": self.next_id, "method": method}
        if params is not None:
            m["params"] = params
        return self.next_id, m

    def wait(self, rid, timeout=60):
        with self.cv:
            ok = self.cv.wait_for(lambda: rid in self.responses, timeout=timeout)
        assert ok, f"no response to {rid}; stderr:\n{self.stderr.decode()[-3000:]}"
        return self.responses[rid]

    def call(self, method, params=None, timeout=60):
        rid, m = self.request(method, params)
        self.send(m)
        return self.wait(rid, timeout)

    def initialize(self):
        r = self.call("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                     "clientInfo": {"name": "test", "version": "0"}})
        assert "result" in r, r
        self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def close(self):
        try:
            self.p.stdin.close()
        except OSError:
            pass
        try:
            self.p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.p.kill()
            self.p.wait()


def _swap(marker: Path):
    tmp = marker.with_suffix(".new")
    tmp.write_text(str(time.time()))
    os.replace(tmp, marker)          # same path, new inode — like a venv rebuild


def _env(tmp_path, marker):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = dict(os.environ)
    env.update({"HOME": str(home), "CO_DB_PATH": str(home / "context.db"),
                "CO_CHROMA_PATH": str(home / "chroma"), "CO_EMBEDDING_MODEL": "none",
                "CONTORCH_MCP_GUARD_WATCH": str(marker),
                "PYTHONPATH": os.pathsep.join([str(SRC), env.get("PYTHONPATH", "")])})
    for k in ("CONTORCH_MCP_RESUMED", "CONTORCH_MCP_PENDING", "CONTORCH_MCP_PENDING_FILE",
              "CONTORCH_MCP_GUARD"):
        env.pop(k, None)
    return env


def _text(resp):
    return resp["result"]["content"][0]["text"]


def test_real_server_survives_a_swap_mid_stream(tmp_path):
    """contorch-mcp itself: 50 tool calls, the code swapped after call 20 and a
    pipelined burst right behind it — every call answered, no error, same PID,
    exactly one re-exec, the undispatched bytes replayed."""
    marker = tmp_path / "python"
    marker.write_text("v1")
    c = Client([sys.executable, "-m", "context_orchestrator.server"], _env(tmp_path, marker))
    try:
        c.initialize()
        pid = c.p.pid
        r = c.call("tools/call", {"name": "create_task", "arguments": {"name": "guard-task"}})
        assert "result" in r and not r["result"].get("isError"), r
        errors = 0
        for i in range(20):
            r = c.call("tools/call", {"name": "list_tasks", "arguments": {}})
            errors += "error" in r or r["result"].get("isError", False)
        _swap(marker)
        # 10 requests in ONE write: the guard reads them all, notices the swap
        # on the first, re-execs, and the new image must replay all ten.
        burst = [c.request("tools/call", {"name": "list_tasks", "arguments": {}}) for _ in range(10)]
        c.send(*[m for _rid, m in burst])
        for rid, _m in burst:
            r = c.wait(rid)
            errors += "error" in r or r["result"].get("isError", False)
            assert "guard-task" in _text(r)
        for i in range(20):
            r = c.call("tools/call", {"name": "list_tasks", "arguments": {}})
            errors += "error" in r or r["result"].get("isError", False)
            assert "guard-task" in _text(r)
        assert errors == 0
        assert c.p.pid == pid and c.p.poll() is None
    finally:
        c.close()
    err = c.stderr.decode()
    assert err.count("re-exec") == 1, err[-3000:]
    assert "resumed in a new image" in err and "replaying" in err


SLOW_SERVER = textwrap.dedent('''
    import os, sys
    import anyio
    from mcp.server.fastmcp import FastMCP
    from context_orchestrator import stdio_guard

    mcp = FastMCP("guard-test")
    RESUMED = os.environ.get("CONTORCH_MCP_RESUMED") == "1"

    @mcp.tool()
    async def slow(seconds: float) -> str:
        await anyio.sleep(seconds)
        return f"slow pid={os.getpid()} resumed={RESUMED}"

    @mcp.tool()
    def fast() -> str:
        return f"fast pid={os.getpid()} resumed={RESUMED}"

    if __name__ == "__main__":
        stdio_guard.run(mcp, argv=[sys.executable, __file__])
''')


def test_no_reexec_while_a_request_is_in_flight(tmp_path):
    marker = tmp_path / "python"
    marker.write_text("v1")
    script = tmp_path / "slow_server.py"
    script.write_text(SLOW_SERVER)
    c = Client([sys.executable, str(script)], _env(tmp_path, marker))
    try:
        c.initialize()
        assert "resumed=False" in _text(c.call("tools/call", {"name": "fast", "arguments": {}}))
        slow_id, slow = c.request("tools/call", {"name": "slow", "arguments": {"seconds": 1.5}})
        c.send(slow)
        time.sleep(0.3)                      # slow is now in flight in the old image
        _swap(marker)
        fast_id, fast = c.request("tools/call", {"name": "fast", "arguments": {}})
        c.send(fast)
        s = c.wait(slow_id)
        f = c.wait(fast_id)
        assert "resumed=False" in _text(s), "the in-flight request finished on the old image"
        assert "resumed=True" in _text(f), "the next one ran on the new image"
        assert c.order.index(slow_id) < c.order.index(fast_id)
        assert f"pid={c.p.pid}" in _text(f) and f"pid={c.p.pid}" in _text(s)
    finally:
        c.close()


def test_guard_off_is_plain_stdio(tmp_path):
    marker = tmp_path / "python"
    marker.write_text("v1")
    script = tmp_path / "slow_server.py"
    script.write_text(SLOW_SERVER)
    env = _env(tmp_path, marker)
    env["CONTORCH_MCP_GUARD"] = "0"
    c = Client([sys.executable, str(script)], env)
    try:
        c.initialize()
        _swap(marker)
        assert "resumed=False" in _text(c.call("tools/call", {"name": "fast", "arguments": {}}))
    finally:
        c.close()


def test_large_pending_buffer_goes_through_a_private_file(monkeypatch):
    from context_orchestrator import stdio_guard as g
    big = b"x" * (g._INLINE_LIMIT + 10)
    env = g._pending_env(big)
    assert set(env) == {g.ENV_PENDING_FILE}
    path = Path(env[g.ENV_PENDING_FILE])
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    monkeypatch.setenv(g.ENV_PENDING_FILE, str(path))
    monkeypatch.delenv(g.ENV_PENDING, raising=False)
    assert g._take_pending() == big and not path.exists()
    small = g._pending_env(b'{"a":1}\n')
    monkeypatch.setenv(g.ENV_PENDING, small[g.ENV_PENDING])
    assert g._take_pending() == b'{"a":1}\n'
