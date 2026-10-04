"""`contorch-hook` (context_orchestrator.hook.main): stdin JSON in, one JSON
object out, exit 0 whatever happens."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"


def _run(tmp_path, payload, extra_env=None, raw=None):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = dict(os.environ, HOME=str(home), CO_DB_PATH=str(tmp_path / "c.db"),
               CO_CHROMA_PATH=str(tmp_path / "chroma"), CO_EMBEDDING_MODEL="none",
               PYTHONPATH=os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")]))
    env.update(extra_env or {})
    t = time.monotonic()
    r = subprocess.run([sys.executable, "-c", "from context_orchestrator.hook import main; main()"], env=env,
                       input=raw if raw is not None else json.dumps(payload),
                       capture_output=True, text=True, timeout=60)
    return r, time.monotonic() - t


def _seed(tmp_path):
    from context_orchestrator.db import Database
    db = Database(db_path=tmp_path / "c.db")
    t = db.create_task("launch", project="p")
    db.add_source(t["id"], "text", "The zebra migration ships on Thursday behind the kiwi flag", notes="")


def test_memory_hits_and_git_state(tmp_path):
    _seed(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "trunk", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "commit",
                    "--allow-empty", "-qm", "first commit"], check=True)
    r, _ = _run(tmp_path, {"prompt": "when does the zebra migration ship?", "cwd": str(repo)})
    assert r.returncode == 0, r.stderr
    ctx = json.loads(r.stdout)["additionalContext"]
    assert ctx.startswith("**[auto-context]**")
    assert "zebra migration ships" in ctx and "(keyword)" in ctx
    assert "branch: `trunk`" in ctx and "first commit" in ctx
    beat = json.loads((tmp_path / "home" / ".context-orchestrator" / "auto-context-heartbeat.json").read_text())
    assert beat["injected_chars"] == len(ctx)


def test_trivial_slash_and_garbage_input_give_empty_context(tmp_path):
    for payload in ({"prompt": "ok thanks"}, {"prompt": "/compact please now"}):
        r, _ = _run(tmp_path, payload)
        assert r.returncode == 0 and json.loads(r.stdout) == {"additionalContext": ""}
    r, _ = _run(tmp_path, None, raw="not json at all")
    assert r.returncode == 0 and json.loads(r.stdout) == {"additionalContext": ""}


def test_a_broken_memory_still_answers(tmp_path):
    (tmp_path / "c.db").write_text("this is not a sqlite database")
    r, _ = _run(tmp_path, {"prompt": "when does the zebra migration ship?", "cwd": str(tmp_path)})
    assert r.returncode == 0 and json.loads(r.stdout) == {"additionalContext": ""}


def test_search_budget_bounds_the_hook(tmp_path, monkeypatch):
    """A search that never returns (e.g. a hung network call) costs at most the
    budget; the prompt goes on with whatever else there is."""
    from context_orchestrator import hook
    slow = tmp_path / "slowsite"
    slow.mkdir()
    (slow / "sitecustomize.py").write_text(
        "import time\nimport context_orchestrator.hook as h\n"
        "h.SEARCH_BUDGET_S = 1.0\n"
        "h.search_lines = lambda *a, **k: (time.sleep(30), ([], 'vector'))[1]\n")
    r, took = _run(tmp_path, {"prompt": "when does the zebra migration ship?", "cwd": str(tmp_path)},
                   extra_env={"PYTHONPATH": os.pathsep.join([str(slow), str(SRC)])})
    assert r.returncode == 0 and json.loads(r.stdout) == {"additionalContext": ""}
    assert took < 5, took
    assert hook.SEARCH_BUDGET_S < 10   # inside Claude Code's HOOK_TIMEOUT_S
