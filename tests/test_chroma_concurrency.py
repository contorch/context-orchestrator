"""Multi-process stress test of in-process Chroma (slow; `pytest -m slow`).

4 writers x 400 upserts (two of them also delete), 3 long-lived "MCP
servers" (reload + count + hybrid search every 0.3 s), and a stream of hook
processes, all on one folder at once; then a fresh process checks every id.
It must report 0 missing, 0 resurrected, 0 vector-less ids.

chromadb is pinned in pyproject.toml; a bump must pass this test.

The control (CO_STRESS_CONTROL=1) runs the same load through raw chromadb
clients held for each process's life — the pre-0.5 behaviour — to show the
loss the session lock prevents. It is report-only: the loss is real but not
deterministic (the lab saw 0, 32 and 147 lost of 2662 on three runs).

Knobs: CO_STRESS_WRITERS (4), CO_STRESS_DOCS (400), CO_STRESS_SERVERS (3),
CO_STRESS_HOOKS (15), CO_STRESS_REPORT (write the summary JSON there).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROLES = HERE / "chroma_stress.py"

pytestmark = pytest.mark.slow


def _env(tmp_path: Path) -> dict:
    home = tmp_path / "home"
    (home / ".context-orchestrator").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({
        "HOME": str(home),                      # never the real ~/.context-orchestrator
        "CO_CHROMA_PATH": str(home / ".context-orchestrator" / "chroma"),
        "CO_DB_PATH": str(home / ".context-orchestrator" / "context.db"),
        "CO_EMBEDDING_MODEL": "local",          # replaced by the hash EF in the roles
        "ANONYMIZED_TELEMETRY": "False",
        "PYTHONPATH": os.pathsep.join([str(HERE.parent / "src"), env.get("PYTHONPATH", "")]),
    })
    env.pop("CO_CHROMA_HOST", None)
    env.pop("CO_CHROMA_PORT", None)
    return env


def _spawn(env, *args) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, str(ROLES), *map(str, args)], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def run_stress(tmp_path: Path, unlocked: bool) -> dict:
    env = _env(tmp_path)
    W = int(os.environ.get("CO_STRESS_WRITERS", "4"))
    N = int(os.environ.get("CO_STRESS_DOCS", "400"))
    S = int(os.environ.get("CO_STRESS_SERVERS", "3"))
    H = int(os.environ.get("CO_STRESS_HOOKS", "15"))
    flag = ["--unlocked"] if unlocked else []
    out = tmp_path / "out"
    out.mkdir()
    done = tmp_path / "writers-done"
    t0 = time.monotonic()
    servers = [_spawn(env, "server", f"srv{s}", done, out / f"srv{s}.json", *flag) for s in range(S)]
    time.sleep(6)   # servers up and holding their client before the writes start
    writers = [_spawn(env, "writer", f"w{w}", N, 10 if w < 2 else 0, out / f"w{w}.json", *flag)
               for w in range(W)]
    hooks = _spawn(env, "hook", H, out / "hook.json")
    rcs = {}
    for name, p in [(f"w{i}", p) for i, p in enumerate(writers)] + [("hook", hooks)]:
        _o, e = p.communicate(timeout=1500)
        rcs[name] = (p.returncode, e.strip()[-800:])
    done.write_text("1")
    for i, p in enumerate(servers):
        _o, e = p.communicate(timeout=300)
        rcs[f"srv{i}"] = (p.returncode, e.strip()[-800:])
    elapsed = time.monotonic() - t0

    expected, gone, werr, op_ms = [], [], [], []
    for w in range(W):
        f = out / f"w{w}.json"
        r = json.loads(f.read_text()) if f.exists() else {"deleted": [], "errors": ["no output"], "op_ms": []}
        gone += r["deleted"]
        expected += [f"w{w}-{k}" for k in range(N) if f"w{w}-{k}" not in r["deleted"]]
        werr += r["errors"]
        op_ms += r["op_ms"]
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"expected": expected, "gone": gone}))
    v = subprocess.run([sys.executable, str(ROLES), "verify", spec], env=env,
                       capture_output=True, text=True, timeout=900)
    assert v.returncode == 0, v.stderr[-3000:]
    verdict = json.loads(v.stdout.strip().splitlines()[-1])

    srv = {}
    for s in range(S):
        f = out / f"srv{s}.json"
        r = json.loads(f.read_text()) if f.exists() else {"samples": [], "errors": ["no output"]}
        srv[f"srv{s}"] = {"samples": len(r["samples"]), "errors": len(r["errors"]),
                          "error_examples": r["errors"][:3],
                          "max_count": max((x["count"] for x in r["samples"]), default=None)}
    hk = json.loads((out / "hook.json").read_text()) if (out / "hook.json").exists() else {"runs": [], "errors": ["no output"]}
    hook_ms = sorted(x["ms"] for x in hk["runs"])
    op_ms.sort()

    def pct(xs, q):
        return xs[min(len(xs) - 1, int(len(xs) * q))] if xs else None
    summary = {
        "mode": "unlocked-control" if unlocked else "session-lock",
        "load": {"writers": W, "docs_per_writer": N, "servers": S, "hooks": H},
        "elapsed_s": round(elapsed, 1),
        "returncodes": {k: rc for k, (rc, _e) in rcs.items()},
        "nonzero": {k: e for k, (rc, e) in rcs.items() if rc != 0},
        "writer_errors": len(werr), "writer_error_examples": werr[:5],
        "write_op_ms_p50": pct(op_ms, 0.5), "write_op_ms_p95": pct(op_ms, 0.95),
        "servers": srv,
        "hook_runs": len(hook_ms), "hook_errors": len(hk["errors"]),
        "hook_error_examples": hk["errors"][:3],
        "hook_modes": {m: sum(1 for x in hk["runs"] if x.get("mode") == m) for m in ("vector", "keyword")},
        "hook_ms_p50": pct(hook_ms, 0.5), "hook_ms_max": hook_ms[-1] if hook_ms else None,
        "verify": verdict,
    }
    line = (f"[{summary['mode']}] vectorless={verdict['vectorless']} resurrected={verdict['resurrected']} "
            f"missing={verdict['missing']} knn_miss={verdict['knn_miss']} ids={verdict['ids']}/"
            f"{verdict['expected']} elapsed={summary['elapsed_s']}s")
    print("\n" + line)
    print(json.dumps(summary, indent=1))
    report = os.environ.get("CO_STRESS_REPORT")
    if report:
        Path(report).parent.mkdir(parents=True, exist_ok=True)
        with open(report, "a") as f:
            f.write(json.dumps(summary) + "\n")
    return summary


def test_session_lock_loses_nothing_under_concurrency(tmp_path):
    s = run_stress(tmp_path, unlocked=False)
    v = s["verify"]
    assert v["vectorless"] == 0, v
    assert v["resurrected"] == 0, v
    assert v["missing"] == 0, v
    assert v["dup_ids"] == 0, v
    assert v["knn_miss"] == 0, v
    assert s["nonzero"] == {}, s["nonzero"]
    assert s["writer_errors"] == 0, s["writer_error_examples"]
    assert all(x["errors"] == 0 for x in s["servers"].values()), s["servers"]
    assert s["hook_errors"] == 0, s["hook_error_examples"]
    # Servers kept seeing other processes' writes (no stale readers).
    assert all((x["max_count"] or 0) > 0 for x in s["servers"].values()), s["servers"]


@pytest.mark.skipif(os.environ.get("CO_STRESS_CONTROL") != "1",
                    reason="control run (no lock, pre-0.5 behaviour); set CO_STRESS_CONTROL=1")
def test_control_without_the_lock_reports_loss(tmp_path):
    """Report-only: shows what the lock prevents. Never fails on loss."""
    s = run_stress(tmp_path, unlocked=True)
    v = s["verify"]
    print(f"control: vectorless={v['vectorless']} resurrected={v['resurrected']} missing={v['missing']} "
          f"crashed={sorted(s['nonzero'])}")
