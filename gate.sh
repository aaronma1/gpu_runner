#!/bin/sh
# Only allow: <venv python> <gpu_runner.py> submit|query|list ...
PY=/home/a84456917/Projects/gpu_runner/.venv/bin/python
RUNNER=/home/a84456917/Projects/gpu_runner/gpu_runner.py
set -f                                    # no globbing while re-splitting
eval "set -- $SSH_ORIGINAL_COMMAND"       # split the requested command like a shell would
[ "$1" = "$PY" ] && [ "$2" = "$RUNNER" ] || { echo "gate: not allowed" >&2; exit 2; }
case "$3" in submit|query|list) ;; *) echo "gate: not allowed" >&2; exit 2;; esac
shift 2
exec "$PY" "$RUNNER" "$@"