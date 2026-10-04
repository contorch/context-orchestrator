#!/usr/bin/env python3
"""Moved into the package: the hook is `contorch-hook` (context_orchestrator.hook),
installed by `contorch-memory claude install`. This shim keeps old settings
entries (a symlink to this file) working until that install replaces them."""
import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
for _hook in (os.path.join(_root, ".venv", "bin", "contorch-hook"),
              os.path.expanduser("~/.context-orchestrator/venv/bin/contorch-hook")):
    if os.access(_hook, os.X_OK):
        os.execv(_hook, [_hook] + sys.argv[1:])
print('{"additionalContext": ""}')
