# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp>=1.2,<2"]
# ///
"""MCP (stdio) server exposing gpu_runner.py to agents.

Run:  uv run --script gpu_runner_mcp.py

Talks to the daemon through the gpu_runner.py CLI. With SSH_HOST = None the CLI runs on this
machine (the daemon's rpyc port only listens on localhost); set SSH_HOST to an ssh alias to
reach a daemon on another machine.
"""
import io
import os
import shlex
import subprocess
import tarfile
import time
import uuid

from mcp.server.mcpserver import MCPServer

# ---------------------------------------------------------------- hardcoded for now
GPURUN = "/home/a84456917/Projects/gpu_runner/gpu_runner.py"
PYTHON = "/home/a84456917/Projects/gpu_runner/.venv/bin/python"   # env that has rpyc
DAEMON_WORKDIR = "/home/a84456917/.gpu_runner"                    # where job dirs live
SSH_HOST = "gpu"                                                     # e.g. "gpu" to go over ssh

TERMINAL = {"COMPLETED", "FAILED"}
EXCLUDE_DIRS = {".git", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache",
                ".ipynb_checkpoints", "node_modules", "wandb"}
MAX_UPLOAD_MB = 500
SUBMIT_RETRIES = 3

mcp = MCPServer("gpu_runner")


class RunnerError(RuntimeError):
    pass


def run_cli(args, stdin=None, timeout=120):
    """Run `gpu_runner.py <args>`; returns (stdout, stderr). Raises RunnerError on failure."""
    cmd = [PYTHON, GPURUN, *args]
    if SSH_HOST:
        cmd = ["ssh", "-o", "BatchMode=yes", SSH_HOST, shlex.join(cmd)]
    try:
        p = subprocess.run(cmd, input=stdin, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RunnerError(f"gpu_runner {args[0]} timed out after {timeout}s") from None
    out, err = p.stdout.decode(errors="replace"), p.stderr.decode(errors="replace")
    if p.returncode != 0:
        raise RunnerError(err.strip() or out.strip() or f"gpu_runner exited {p.returncode}")
    return out, err


def tar_dir(path):
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isdir(path):
        raise RunnerError(f"not a directory: {path}")

    def skip(ti):
        parts = ti.name.split("/")
        return None if any(p in EXCLUDE_DIRS for p in parts) or ti.name.endswith((".pyc", ".pyo")) else ti

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(path, arcname=".", filter=skip)
    data = buf.getvalue()
    if len(data) > MAX_UPLOAD_MB * 2**20:
        raise RunnerError(f"upload is {len(data) / 2**20:.0f} MB (> {MAX_UPLOAD_MB} MB); "
                          "download datasets inside the job instead of uploading them")
    return data


def parse_status(line):
    """`FAILED 7 (failed: exit 7)` / `COMPLETED 0 ` / `RUNNING  ` -> dict."""
    status, _, rest = line.strip().partition(" ")
    rest = rest.strip()
    code, failed = None, None
    if rest:
        head, _, tail = rest.partition(" ")
        if head.lstrip("-").isdigit():
            code, rest = int(head), tail.strip()
        if rest.startswith("(failed: ") and rest.endswith(")"):
            failed = rest[len("(failed: "):-1]
    return {"status": status, "exit_code": code, "failed_command": failed}


def get_status(job_uuid):
    out, _ = run_cli(["query", job_uuid, "--status"], timeout=60)
    st = parse_status(out.splitlines()[0] if out else "")
    st["job_uuid"] = job_uuid
    st["job_dir"] = f"{DAEMON_WORKDIR}/{job_uuid}"
    return st


def tail(text, lines):
    if not lines:
        return text
    parts = text.splitlines(keepends=True)
    cut = len(parts) - lines
    return (f"[... {cut} earlier lines omitted ...]\n" if cut > 0 else "") + "".join(parts[-lines:])


# ---------------------------------------------------------------- tools
@mcp.tool()
def gpu_submit(local_dir: str, commands: list[str]) -> dict:
    """Queue a job on the GPU box: upload a directory, then run shell commands in it.

    Uploads `local_dir` as a gzipped tarball (excluding .git/.venv/caches, max 500 MB) and
    runs each string in `commands` in order, as a shell command, inside that directory.
    The first command that exits non-zero stops the job and it is marked FAILED; the
    remaining commands do not run. Commands get no stdin. Jobs run one at a time and each
    gets all the runner's GPUs (CUDA_VISIBLE_DEVICES is set). Write results to files in the
    job directory (e.g. out/) — that is how they can be read back with gpu_output.

    There is no cancel and no time limit, so make sure the job terminates.

    Args:
        local_dir: local directory to upload as the job's working tree.
        commands: non-empty list of shell commands, run in order.

    Returns:
        {"job_uuid": str, "job_dir": str} on success, or {"error": str} on failure.

    Note: idempotent per call — transport retries reuse one message_id, so a retried submit
    returns the same job rather than queueing a duplicate.
    """
    if not commands:
        return {"error": "commands must be a non-empty list"}
    try:
        data = tar_dir(local_dir)
    except (RunnerError, OSError) as e:
        return {"error": str(e)}
    message_id = str(uuid.uuid4())  # same id on every retry -> the daemon dedups
    last = None
    for attempt in range(SUBMIT_RETRIES):
        try:
            out, _ = run_cli(["submit", "--message_id", message_id, "-", *commands],
                             stdin=data, timeout=600)
            job_uuid = out.strip().splitlines()[-1]
            return {"job_uuid": job_uuid, "job_dir": f"{DAEMON_WORKDIR}/{job_uuid}"}
        except RunnerError as e:
            last = e
            if "no daemon" in str(e):
                break  # retrying won't help
            time.sleep(2 * (attempt + 1))
    return {"error": str(last)}


@mcp.tool()
def gpu_status(job_uuid: str) -> dict:
    """Report one job's current state.

    Args:
        job_uuid: the job's uuid (from gpu_submit or gpu_list).

    Returns:
        {"status": "QUEUED" | "RUNNING" | "COMPLETED" | "FAILED",
        "exit_code": int | None,          # exit code of the last command; None until done
        "failed_command": str | None,     # the failing command when FAILED, else None
        "job_uuid": str, "job_dir": str}
        or {"error": str} (e.g. unknown job) on failure.
    """
    try:
        return get_status(job_uuid)
    except RunnerError as e:
        return {"error": str(e)}


@mcp.tool()
def gpu_output(job_uuid: str, tail_lines: int = 200) -> dict:
    """Return a job's stdout and stderr so far. Works while the job is still running.

    Args:
        job_uuid: the job's uuid.
        tail_lines: return only the last N lines of each stream; 0 returns everything.

    Returns:
        {"stdout": str, "stderr": str}, or {"error": str} on failure.
    """
    try:
        out, err = run_cli(["query", job_uuid, "--stdout", "--stderr"], timeout=120)
        return {"stdout": tail(out, tail_lines), "stderr": tail(err, tail_lines)}
    except RunnerError as e:
        return {"error": str(e)}


@mcp.tool()
def gpu_wait(job_uuid: str, max_wait_s: int = 600, poll_s: int = 10) -> dict:
    """Wait until the job finishes or max_wait_s passes (keep it below the MCP call timeout).
    Returns the final status, or the current one with still_running=true. The job keeps running
    either way; call again or do other work. Never resubmit just because a wait timed out."""
    deadline = time.time() + max_wait_s
    try:
        while True:
            st = get_status(job_uuid)
            if st["status"] in TERMINAL:
                return st
            if time.time() + poll_s > deadline:
                return {**st, "still_running": True}
            time.sleep(poll_s)
    except RunnerError as e:
        return {"error": str(e)}


@mcp.tool()
def gpu_list() -> str:
    """List every job the daemon knows about since it last started.

    Returns:
        Text, one line per job: "<job_uuid>  <status>  <first command>".
        "(no jobs)" when the queue is empty, or "error: <message>" on failure.

    Job state is held in memory, so restarting the daemon forgets all jobs.
    """
    try:
        out, _ = run_cli(["list"], timeout=60)
        return out or "(no jobs)"
    except RunnerError as e:
        return f"error: {e}"


if __name__ == "__main__":
    mcp.run()