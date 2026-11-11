#!/usr/bin/env python3
"""authorized_keys forced command: only lets `[.local/bin/]gpu_runner
submit|query|list|download|cancel ...` through. `make install` installs it as $(LIBDIR)/gate
with @RUNNER@ filled in; restrict a key with

    command="<LIBDIR>/gate",no-port-forwarding,no-X11-forwarding,no-agent-forwarding,no-pty ssh-ed25519 ...

The requested command is split with shlex (never by a shell), so quoting is honoured but
`;`, `$(...)` etc. are just characters in an argument.
"""
import os
import shlex
import sys

RUNNER = "@RUNNER@"
ALLOWED = {"submit", "query", "list", "download", "cancel"}

try:
    argv = shlex.split(os.environ.get("SSH_ORIGINAL_COMMAND", ""))
except ValueError:
    argv = []
if len(argv) < 2 or argv[0] not in ("gpu_runner", ".local/bin/gpu_runner") or argv[1] not in ALLOWED:
    sys.exit("gate: not allowed")
os.execv(RUNNER, [RUNNER, *argv[1:]])
