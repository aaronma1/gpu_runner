#!/bin/sh
# Only allow: <venv python> <gpu_runner.py> submit|query|list|download|cancel ...
# Template: `make install-mcp` fills in @PY@/@RUNNER@ and installs it on the ssh host as an
# authorized_keys forced command.
PY=@PY@
RUNNER=@RUNNER@
set -f                                    # no globbing while re-splitting
eval "set -- $SSH_ORIGINAL_COMMAND"       # split the requested command like a shell would
[ "$1" = "$PY" ] && [ "$2" = "$RUNNER" ] || { echo "gate: not allowed" >&2; exit 2; }
case "$3" in submit|query|list|download|cancel) ;; *) echo "gate: not allowed" >&2; exit 2;; esac
shift 2
exec "$PY" "$RUNNER" "$@"