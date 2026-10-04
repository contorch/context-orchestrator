"""`contorch-memory claude install|uninstall|status` (claude_install.py).

Scratch CLAUDE_CONFIG_DIR (with a space in it), and a stub `claude` that
edits $CLAUDE_CONFIG_DIR/.claude.json the way `claude mcp add/remove/add-json`
do (shape checked against claude 2.1.280)."""
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from context_orchestrator import claude_install as ci
from context_orchestrator import settings

FAKE_CLAUDE = r'''#!{python}
import json, os, sys
cfg = os.environ.get("CLAUDE_CONFIG_DIR")
path = os.path.join(cfg, ".claude.json") if cfg else os.path.expanduser("~/.claude.json")
with open(os.environ["FAKE_CLAUDE_LOG"], "a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\n")
d = json.load(open(path)) if os.path.exists(path) else {{}}
servers = d.setdefault("mcpServers", {{}})
a = sys.argv[1:]
if os.environ.get("FAKE_CLAUDE_FAIL_ADD") and a[:2] == ["mcp", "add"]:
    sys.exit("boom")
if a[:2] == ["mcp", "remove"]:
    name = a[-1]
    if name not in servers:
        sys.exit(1)
    del servers[name]
elif a[:2] == ["mcp", "add-json"]:
    servers[a[-2]] = json.loads(a[-1])
elif a[:2] == ["mcp", "add"]:
    i = a.index("--")
    head, tail = a[2:i], a[i + 1:]
    env, name, k = {{}}, None, 0
    while k < len(head):
        if head[k] == "--scope":
            k += 2; continue
        if head[k] == "-e":
            key, _, val = head[k + 1].partition("="); env[key] = val; k += 2; continue
        name = head[k]; k += 1
    if name in servers:
        sys.exit(1)
    servers[name] = {{"type": "stdio", "command": tail[0], "args": tail[1:], "env": env}}
json.dump(d, open(path, "w"), indent=2)
'''


@pytest.fixture
def cc(tmp_path, monkeypatch):
    """Scratch Claude Code config (path with a space), stub claude, scratch HOME."""
    cfg = tmp_path / "claude config"
    cfg.mkdir()
    fake = tmp_path / "bin" / "claude"
    fake.parent.mkdir()
    fake.write_text(FAKE_CLAUDE.format(python=sys.executable))
    fake.chmod(0o755)
    log = tmp_path / "claude.log"
    log.write_text("")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(cfg))
    monkeypatch.setenv("CO_CLAUDE_BIN", str(fake))
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(log))
    monkeypatch.setenv("CO_CLAUDE_MANAGED_DIR", str(tmp_path / "managed"))
    monkeypatch.delenv("CONTORCH_CHANNEL", raising=False)
    # The install's own commands live in a directory with a space, too.
    bindir = tmp_path / "my apps" / "venv" / "bin"
    bindir.mkdir(parents=True)
    for name in ("contorch-mcp", "contorch-hook", "contorch-transcripts", "contorch-memory",
                 "context-orchestrator-chroma"):
        (bindir / name).write_text("#!/bin/sh\necho '{\"additionalContext\": \"hi\"}'\n")
        (bindir / name).chmod(0o755)
    monkeypatch.setattr(ci, "entry_points", lambda chan: {
        "mcp": str(bindir / "contorch-mcp"), "hook": str(bindir / "contorch-hook"),
        "transcripts": str(bindir / "contorch-transcripts"), "memory": str(bindir / "contorch-memory"),
        "chroma_cli": str(bindir / "context-orchestrator-chroma")})
    from context_orchestrator import chroma_daemon
    monkeypatch.setattr(chroma_daemon, "LAUNCHD_PLIST", tmp_path / "no-chroma-agent.plist")

    class C:
        pass
    c = C()
    c.cfg, c.bin, c.log, c.tmp = cfg, bindir, log, tmp_path
    c.paths = ci.ClaudePaths.detect()
    c.settings = cfg / "settings.json"
    c.md = cfg / "CLAUDE.md"
    c.skill = cfg / "skills" / "transcripts"
    c.claude_json = cfg / ".claude.json"
    c.calls = lambda: [json.loads(x) for x in log.read_text().splitlines()]
    return c


def _settings(c):
    return json.loads(c.settings.read_text())


def _hooks(c):
    return [h for b in _settings(c)["hooks"]["UserPromptSubmit"] for h in b["hooks"]]


def test_install_writes_the_guarded_quoted_hook_and_is_idempotent(cc):
    other = {"type": "command", "command": "/usr/local/bin/my-other-hook", "timeout": 5}
    cc.settings.write_text(json.dumps({"theme": "dark", "hooks": {
        "UserPromptSubmit": [{"matcher": "*", "hooks": [other]}],
        "Stop": [{"hooks": [{"type": "command", "command": "say done"}]}]}}))
    doc = ci.install("dev")
    assert doc["ok"] and doc["schema"] == "contorch-memory.claude/1"
    assert sorted(doc["changed"]) == ["claude_md", "hook", "mcp", "skill"]
    hooks = _hooks(cc)
    assert other in hooks, "other hooks are kept"
    ours = [h for h in hooks if "contorch-hook:dev" in h["command"]]
    assert len(ours) == 1 and ours[0]["timeout"] == ci.HOOK_TIMEOUT_S == 10
    p = str(cc.bin / "contorch-hook")
    assert ours[0]["command"] == f"[ -x '{p}' ] && '{p}'; exit 0 # contorch-hook:dev"
    s = _settings(cc)
    assert s["theme"] == "dark" and s["hooks"]["Stop"][0]["hooks"][0]["command"] == "say done"
    # Claude Code runs it with /bin/sh: present → output passes through; gone → silent, exit 0.
    r = subprocess.run(["/bin/sh", "-c", ours[0]["command"]], capture_output=True, text=True)
    assert r.returncode == 0 and "hi" in r.stdout
    (cc.bin / "contorch-hook").unlink()
    r = subprocess.run(["/bin/sh", "-c", ours[0]["command"]], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout == "" and r.stderr == ""
    (cc.bin / "contorch-hook").write_text("#!/bin/sh\necho '{}'\n")
    (cc.bin / "contorch-hook").chmod(0o755)

    before = {p: p.read_bytes() for p in (cc.settings, cc.md, cc.claude_json, cc.skill / "SKILL.md")}
    n_calls = len(cc.calls())
    again = ci.install("dev")
    assert again["ok"] and again["changed"] == []
    assert all(again[k]["matches"] for k in ("mcp", "hook", "claude_md", "skill"))
    assert {p: p.read_bytes() for p in before} == before
    assert len(cc.calls()) == n_calls, "no claude mcp calls when nothing changed"
    st = ci.status("dev")
    assert st["ok"] and st["changed"] == [] and st["action"] == "status"


def test_mcp_entry_uses_its_own_path_and_drops_stale_chroma_env(cc):
    cc.claude_json.write_text(json.dumps({"mcpServers": {"context-orchestrator": {
        "type": "stdio", "command": "/old/checkout/.venv/bin/python",
        "args": ["-m", "context_orchestrator.server"],
        "env": {"PYTHONPATH": "/old/checkout/src", "CO_EMBEDDING_MODEL": "local",
                "CO_CHROMA_HOST": "127.0.0.1", "CO_CHROMA_PORT": "8765", "GOOGLE_API_KEY": "secret"}}},
        "other": 1}))
    doc = ci.install("dev")
    srv = json.loads(cc.claude_json.read_text())["mcpServers"]["context-orchestrator"]
    assert srv["command"] == str(cc.bin / "contorch-mcp") and srv["args"] == []
    assert srv["env"] == {"CO_EMBEDDING_MODEL": "local"}, "only allowed CO_* keys; server keys dropped in-process"
    assert doc["mcp"] == {"present": True, "path": str(cc.bin / "contorch-mcp"), "matches": True,
                          "env_keys": ["CO_EMBEDDING_MODEL"]}
    calls = cc.calls()
    assert calls[0][:2] == ["mcp", "remove"]
    assert calls[1][:5] == ["mcp", "add", "--scope", "user", "context-orchestrator"]


def test_server_chroma_env_kept_when_the_chroma_agent_exists(cc, monkeypatch):
    from context_orchestrator import chroma_daemon
    plist = cc.tmp / "agent.plist"
    plist.write_text("x")
    monkeypatch.setattr(chroma_daemon, "LAUNCHD_PLIST", plist)
    cc.claude_json.write_text(json.dumps({"mcpServers": {"context-orchestrator": {
        "type": "stdio", "command": "/x/contorch-mcp", "args": [], "env": {"CO_CHROMA_PORT": "9000"}}}}))
    ci.install("dev")
    assert json.loads(cc.claude_json.read_text())["mcpServers"]["context-orchestrator"]["env"] == \
        {"CO_CHROMA_PORT": "9000"}


def test_failed_mcp_add_restores_the_previous_entry(cc, monkeypatch):
    prev = {"type": "stdio", "command": "/x/contorch-mcp", "args": [], "env": {}}
    cc.claude_json.write_text(json.dumps({"mcpServers": {"context-orchestrator": prev}}))
    monkeypatch.setenv("FAKE_CLAUDE_FAIL_ADD", "1")
    doc = ci.install("dev")
    assert not doc["ok"] and doc["error"]["code"] == "mcp_failed"
    assert json.loads(cc.claude_json.read_text())["mcpServers"]["context-orchestrator"] == prev


def test_no_claude_cli_is_a_todo_not_a_failure(cc, monkeypatch):
    monkeypatch.setenv("CO_CLAUDE_BIN", str(cc.tmp / "nope"))
    doc = ci.install("dev")
    assert doc["ok"] and not doc["mcp"]["present"]
    assert any("Install Claude Code" in t for t in doc["todo"])
    assert doc["hook"]["matches"] and doc["claude_md"]["matches"] and doc["skill"]["matches"]


# ---- what earlier installers left behind ------------------------------------

def _legacy_texts():
    repo = Path(__file__).resolve().parent.parent
    return {
        "template": (repo / "claude-md-template.md").read_text(),
        "snippet": (repo / "templates" / "claude-md-snippet.md").read_text(),
    }


def _shipped_hook_bytes():
    """Any shipped version of hooks/auto-context.py (from git), else skip."""
    repo = Path(__file__).resolve().parent.parent
    known = set(ci.legacy_hashes()["auto-context.py"])
    revs = subprocess.run(["git", "-C", str(repo), "log", "--all", "--format=%H", "--",
                           "hooks/auto-context.py"], capture_output=True, text=True).stdout.split()
    for r in revs:
        for rr in (r, r + "^"):
            b = subprocess.run(["git", "-C", str(repo), "show", f"{rr}:hooks/auto-context.py"],
                               capture_output=True).stdout
            if b and hashlib.sha256(b).hexdigest() in known:
                return b
    pytest.skip("no git history available")


def test_legacy_copy_replaced_only_by_hash(cc):
    hook_file = cc.cfg / "hooks" / "auto-context.py"
    hook_file.parent.mkdir()
    hook_file.write_bytes(_shipped_hook_bytes())
    hook_file.chmod(0o755)
    cc.settings.write_text(json.dumps({"hooks": {"UserPromptSubmit": [
        {"matcher": "*", "hooks": [{"type": "command", "command": str(hook_file), "timeout": 10}]}]}}))
    doc = ci.install("dev")
    assert doc["hook"]["legacy_copy"] == "replaced" and doc["hook"]["matches"]
    assert not hook_file.exists()
    assert [h["command"] for h in _hooks(cc)] == [ci.hook_command(str(cc.bin / "contorch-hook"), "dev")]


def test_custom_legacy_copy_is_kept_with_its_entry(cc):
    hook_file = cc.cfg / "hooks" / "auto-context.py"
    hook_file.parent.mkdir()
    hook_file.write_text("#!/usr/bin/env python3\n# my own tweaks\nprint('{}')\n")
    entry = {"type": "command", "command": str(hook_file), "timeout": 10}
    cc.settings.write_text(json.dumps({"hooks": {"UserPromptSubmit": [{"matcher": "*", "hooks": [entry]}]}}))
    doc = ci.install("dev")
    assert doc["hook"]["legacy_copy"] == "kept_custom" and not doc["hook"]["present"]
    assert hook_file.exists() and _hooks(cc) == [entry]
    assert any("customised" in t for t in doc["todo"])


def test_claude_md_known_legacy_blocks_become_one_marked_block(cc):
    t = _legacy_texts()
    cc.md.write_text("# My notes\n\nkeep this\n\n" + t["template"] + "\nmiddle text\n\n" + t["snippet"]
                     + "\n## Tail\nmine\n")
    doc = ci.install("dev")
    text = cc.md.read_text()
    assert doc["claude_md"]["matches"] and doc["claude_md"]["legacy_block"] == "none"
    assert text.count(ci.MD_BEGIN) == 1 and text.count(ci.MD_END) == 1
    assert "BEGIN CONTEXT-ORCHESTRATOR" not in text and "auto-context-section" not in text
    assert text.index("keep this") < text.index(ci.MD_BEGIN) < text.index("middle text") < text.index("## Tail")


def test_claude_md_edited_legacy_block_is_kept_and_no_duplicate_added(cc):
    edited = _legacy_texts()["template"].replace("call `get_task()`", "call `get_task()` please")
    cc.md.write_text(edited)
    doc = ci.install("dev")
    assert cc.md.read_text() == edited
    assert not doc["claude_md"]["present"] and doc["claude_md"]["legacy_block"] == "kept_custom"
    assert any("edited older Contorch block" in t for t in doc["todo"])


def test_claude_md_unmarked_exact_copy_is_adopted_other_guidance_left_alone(cc):
    block = ci.md_template()
    inner = block[len(ci.MD_BEGIN):-len(ci.MD_END)].strip()
    cc.md.write_text("intro\n\n" + inner + "\n\noutro\n")
    ci.install("dev")
    assert cc.md.read_text().count(ci.MD_BEGIN) == 1 and "outro" in cc.md.read_text()
    assert ci.claude_md_status(cc.paths)["matches"]

    cc.md.write_text("## Context Orchestrator\nmy own words about context-orchestrator\n")
    doc = ci.install("dev")
    assert cc.md.read_text() == "## Context Orchestrator\nmy own words about context-orchestrator\n"
    assert doc["claude_md"]["user_guidance"] and not doc["claude_md"]["present"]


def test_skill_rendered_with_its_own_path_symlink_and_curl_copy_replaced_user_copy_kept(cc):
    ci.install("dev")
    text = (cc.skill / "SKILL.md").read_text()
    assert f"CT='{cc.bin / 'contorch-transcripts'}'" in text and "{{" not in text
    stamp = json.loads((cc.skill / ci.SKILL_STAMP).read_text())
    assert stamp["owner"] == "context-orchestrator" and stamp["channel"] == "dev"

    # a symlink to some checkout (older setups)
    import shutil
    shutil.rmtree(cc.skill)
    target = cc.tmp / "checkout" / "skills" / "transcripts"
    target.mkdir(parents=True)
    (target / "SKILL.md").write_text("old")
    cc.skill.symlink_to(target)
    assert ci.install("dev")["skill"]["matches"] and not cc.skill.is_symlink()
    assert (target / "SKILL.md").read_text() == "old", "the link target is not touched"

    # a curl'ed copy of a version we shipped
    shutil.rmtree(cc.skill)
    cc.skill.mkdir()
    repo = Path(__file__).resolve().parent.parent
    old = subprocess.run(["git", "-C", str(repo), "show", "HEAD~3:skills/transcripts/SKILL.md"],
                         capture_output=True).stdout or subprocess.run(
        ["git", "-C", str(repo), "show", "origin/main:skills/transcripts/SKILL.md"], capture_output=True).stdout
    if hashlib.sha256(old).hexdigest() in ci.legacy_hashes()["transcripts_skill"]:
        (cc.skill / "SKILL.md").write_bytes(old)
        assert ci.install("dev")["skill"]["matches"]

    # the user's own skill
    shutil.rmtree(cc.skill)
    cc.skill.mkdir()
    (cc.skill / "SKILL.md").write_text("my skill")
    doc = ci.install("dev")
    assert (cc.skill / "SKILL.md").read_text() == "my skill" and doc["skill"]["user_copy"]


def test_uninstall_removes_only_what_is_ours(cc):
    other = {"type": "command", "command": "/usr/local/bin/my-other-hook"}
    cc.settings.write_text(json.dumps({"hooks": {"UserPromptSubmit": [{"hooks": [other]}]}}))
    cc.md.write_text("mine before\n")
    cc.claude_json.write_text(json.dumps({"mcpServers": {"someone-else": {"command": "x"}}}))
    ci.install("dev")
    (cc.cfg / "skills" / "mine").mkdir()
    doc = ci.uninstall("dev")
    assert sorted(doc["changed"]) == ["claude_md", "hook", "mcp", "skill"]
    assert _hooks(cc) == [other]
    assert cc.md.read_text() == "mine before\n"
    servers = json.loads(cc.claude_json.read_text())["mcpServers"]
    assert "context-orchestrator" not in servers and "someone-else" in servers
    assert not cc.skill.exists() and (cc.cfg / "skills" / "mine").exists()
    assert ci.uninstall("dev")["changed"] == []


def test_uninstall_leaves_a_foreign_mcp_entry_named_like_ours(cc):
    cc.claude_json.write_text(json.dumps({"mcpServers": {"context-orchestrator": {
        "type": "stdio", "command": "/somebody/else/server", "args": []}}}))
    doc = ci.uninstall("dev")
    assert "mcp" not in doc["changed"] and any("not Contorch" in t for t in doc["todo"])


def test_managed_settings_detected_and_respected(cc):
    managed = cc.tmp / "managed"
    managed.mkdir()
    (managed / "managed-settings.json").write_text(json.dumps({
        "allowManagedHooksOnly": True, "allowedMcpServers": [{"serverName": "github"}]}))
    doc = ci.install("dev")
    assert doc["blocked_by_managed_settings"]
    assert not doc["mcp"]["present"] and not doc["hook"]["present"]
    assert cc.calls() == [] and not cc.settings.exists()
    assert len(doc["managed_reasons"]) == 2
    assert doc["claude_md"]["matches"] and doc["skill"]["matches"]
    (managed / "managed-settings.json").write_text("{}")
    (managed / "managed-mcp.json").write_text("{}")
    assert ci.status("dev")["managed_reasons"] == [
        "mcp: managed-mcp.json takes exclusive control of MCP servers"]


def test_invalid_settings_json_is_never_overwritten(cc):
    cc.settings.write_text("{ not json")
    doc = ci.install("dev")
    assert cc.settings.read_text() == "{ not json"
    assert any("not valid JSON" in t for t in doc["todo"])


def test_backup_dir_gets_the_originals(cc):
    cc.settings.write_text('{"a": 1}')
    cc.md.write_text("hello\n")
    b = cc.tmp / "bk"
    ci.install("dev", backup_dir=b)
    assert (b / "settings.json").read_text() == '{"a": 1}' and (b / "CLAUDE.md").read_text() == "hello\n"


def test_channel_tag_follows_the_channel_and_switching_replaces_it(cc, monkeypatch):
    monkeypatch.setenv("CONTORCH_CHANNEL", "brew")
    ci.install()
    assert [h["command"].rsplit("#", 1)[1].strip() for h in _hooks(cc)] == ["contorch-hook:brew"]
    assert not ci.status("app")["hook"]["matches"]
    ci.install("app")
    assert [h["command"].rsplit("#", 1)[1].strip() for h in _hooks(cc)] == ["contorch-hook:app"]


def test_cli_json_is_one_document_and_text_mode_is_human(cc, capfd):
    assert settings.main(["claude", "install", "--channel", "dev", "--json"]) == 0
    out = capfd.readouterr()
    doc = json.loads(out.out)
    assert doc["schema"] == "contorch-memory.claude/1" and doc["ok"]
    assert set(doc) >= {"mcp", "hook", "claude_md", "skill", "blocked_by_managed_settings", "todo"}
    assert set(doc["mcp"]) == {"present", "path", "matches", "env_keys"}
    assert {"present", "matches", "legacy_copy"} <= set(doc["hook"])
    assert "Claude Code integration" in out.err
    assert settings.main(["claude", "status"]) == 0
    assert "Claude Code integration (dev, status)" in capfd.readouterr().out


def test_existing_text_commands_unchanged(cc, capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("CO_EMBEDDING_MODEL", "none")
    monkeypatch.setenv("CO_DB_PATH", str(tmp_path / "c.db"))
    monkeypatch.setenv("CO_CHROMA_PATH", str(tmp_path / "chroma"))
    assert settings.main(["embeddings"]) == 0
    assert capsys.readouterr().out.startswith("none — keyword")
    assert settings.main(["status"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0].startswith("embeddings:") and "full-text:   on" in out
