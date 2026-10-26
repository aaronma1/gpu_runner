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
import shutil
import subprocess
import tarfile
import tempfile
import time
import uuid

from mcp.server.mcpserver import MCPServer

# ---------------------------------------------------------------- hardcoded for now
GPURUN = "/home/a84456917/Projects/gpu_runner/gpu_runner.py"
PYTHON = "/home/a84456917/Projects/gpu_runner/.venv/bin/python"   # env that has rpyc
DAEMON_WORKDIR = "/home/a84456917/.gpu_runner"                    # where job dirs live
SSH_HOST = "gpu"                                                     # e.g. "gpu" to go over ssh

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED"}
EXCLUDE_DIRS = {".git", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache",
                ".ipynb_checkpoints", "node_modules", "wandb"}
MAX_UPLOAD_MB = 500
SUBMIT_RETRIES = 3

mcp = MCPServer("gpu_runner")


class RunnerError(RuntimeError):
    pass


def run_cli(args, stdin=None, timeout=120, binary=False):
    """Run `gpu_runner.py <args>`; returns (stdout, stderr), stdout as bytes if `binary`.
    Raises RunnerError on failure."""
    cmd = [PYTHON, GPURUN, *args]
    if SSH_HOST:
        cmd = ["ssh", "-o", "BatchMode=yes", SSH_HOST, shlex.join(cmd)]
    try:
        p = subprocess.run(cmd, input=stdin, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RunnerError(f"gpu_runner {args[0]} timed out after {timeout}s") from None
    err = p.stderr.decode(errors="replace")
    if p.returncode != 0:
        msg = err.strip() or ("" if binary else p.stdout.decode(errors="replace").strip())
        raise RunnerError(msg or f"gpu_runner exited {p.returncode}")
    return (p.stdout if binary else p.stdout.decode(errors="replace")), err


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


def status_dict(job_uuid, out):
    """Status dict from the CLI's status line (`query --status` / `cancel` output)."""
    st = parse_status(out.splitlines()[0] if out else "")
    st["job_uuid"] = job_uuid
    st["job_dir"] = f"{DAEMON_WORKDIR}/{job_uuid}"
    return st


def get_status(job_uuid):
    out, _ = run_cli(["query", job_uuid, "--status"], timeout=60)
    return status_dict(job_uuid, out)


def tail(text, lines):
    if not lines:
        return text
    parts = text.splitlines(keepends=True)
    cut = len(parts) - lines
    return (f"[... {cut} earlier lines omitted ...]\n" if cut > 0 else "") + "".join(parts[-lines:])


def _safe_extract(data, dest):
    """Extract a tar.gz built from a job's directory (untrusted names) into dest."""
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(dest, filter="data")
            return
        root = os.path.realpath(dest)
        members = []
        for m in tar.getmembers():
            target = os.path.realpath(os.path.join(dest, m.name))
            if target != root and not target.startswith(root + os.sep):
                raise RunnerError(f"unsafe path in download: {m.name}")
            if m.issym() or m.islnk() or m.isdev():
                continue
            members.append(m)
        tar.extractall(dest, members=members)


def _place(src, dest, job_path, overwrite):
    """Copy extracted `src` to `dest`. A directory merges into dest; a file goes to dest, or
    inside dest if dest is an existing directory or ends with '/'. Returns the final path."""
    if os.path.isdir(src):
        if os.path.exists(dest) and not os.path.isdir(dest):
            raise RunnerError(f"{job_path} is a directory but {dest} is a file")
        if not overwrite:
            for dp, _, fs in os.walk(src):
                for f in fs:
                    target = os.path.join(dest, os.path.relpath(os.path.join(dp, f), src))
                    if os.path.lexists(target):
                        raise RunnerError(f"{target} already exists (overwrite=false)")
        shutil.copytree(src, dest, dirs_exist_ok=True, symlinks=True)
        return dest
    if dest.endswith(os.sep) or os.path.isdir(dest):
        dest = os.path.join(dest, os.path.basename(os.path.normpath(job_path)))
    if os.path.lexists(dest) and not overwrite:
        raise RunnerError(f"{dest} already exists (overwrite=false)")
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    shutil.copy2(src, dest)
    return dest


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

    GPUs: the job can use only the GPUs it was given, renumbered from 0 inside the job (use
    cuda:0, cuda:1, ...; torch.cuda.device_count() says how many). nvidia-smi ignores
    CUDA_VISIBLE_DEVICES and lists every GPU on the machine, including ones that belong to other
    people; that is expected and not a choice you need to make. Use all the GPUs the job sees,
    never pick GPUs from nvidia-smi output, and never set or change CUDA_VISIBLE_DEVICES. You
    don't need to check whether GPUs are free before submitting: jobs queue and get their GPUs
    when they start.

    Jobs get a clean interpreter environment: the daemon's own virtualenv is stripped from
    VIRTUAL_ENV and PATH, so a bare `python` is the system interpreter and `uv` has no venv to
    fall back to. Create your own and address it explicitly, or nothing gets installed where
    you expect:

        uv venv
        uv pip install --python .venv/bin/python numpy
        .venv/bin/python script.py

    If local_dir has a pyproject.toml, build the environment from it with uv instead. .venv is
    never uploaded, so every job builds a fresh one, but uv installs from the GPU box's shared
    cache (~/.cache/uv), so once a package version has been downloaded by any job, later jobs
    reuse it in seconds instead of downloading it again (the first torch install is slow, later
    ones aren't). Pin versions in uv.lock so jobs hit the cache and stay reproducible:

        commands = ["uv sync --locked",          # plain `uv sync` if there is no uv.lock
                    "uv run python train.py"]    # runs in ./.venv, no activation needed

    `uv run` uses the project's ./.venv. Keep heavy dependencies in pyproject.toml rather than
    installing them with pip in a command, so they come from the cache.

    There is no time limit, so make sure the job terminates; stop a job that won't (or that
    you no longer need) with gpu_cancel.

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
        {"status": "QUEUED" | "RUNNING" | "COMPLETED" | "FAILED" | "CANCELLED",
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
def gpu_cancel(job_uuid: str) -> dict:
    """Cancel a job. A queued job is removed from the queue and never runs. A running job has
    all its processes killed (SIGTERM, then SIGKILL) and its remaining commands skipped; the
    call returns once they are gone, so the GPUs are free. Either way the job ends CANCELLED,
    and its output and files stay available (gpu_output, download_job_files). Cancelling a job
    that already finished changes nothing and returns its final status.

    Args:
        job_uuid: the job's uuid.

    Returns:
        The job's status after the cancel, in the same form as gpu_status, or {"error": str}
        on failure (e.g. unknown job).
    """
    try:
        out, _ = run_cli(["cancel", job_uuid], timeout=120)
        return status_dict(job_uuid, out)
    except RunnerError as e:
        return {"error": str(e)}


@mcp.tool()
def download_job_files(job_uuid: str, files: dict[str, str], overwrite: bool = True) -> dict:
    """Copy files from a job's directory on the GPU box to this machine.

    Args:
        job_uuid: the job's uuid.
        files: maps paths in the job directory to absolute local destinations, e.g.
            {"out/metrics.json": "/home/me/exp1/metrics.json", "out/plots": "/home/me/exp1/plots"}.
            Job paths are relative to the job directory; "." is everything. A directory is merged
            into its destination; a file is written to its destination, or inside it if the
            destination is an existing directory or ends with "/".
        overwrite: replace existing local files (default true); if false, an existing file is
            an error.

    Nothing is written if any job path is missing. Total size is capped (1 GB by default), so
    fetch metrics, logs and plots rather than every checkpoint. Works while the job is running,
    but its files may then be incomplete.

    Returns:
        {"downloaded": {job_path: local_path}, "bytes": int, "job_status": str,
        "warning": str (only if the job was still running)}, or {"error": str} on failure.
    """
    if not files:
        return {"error": "files must map at least one job path to a local destination"}
    targets = {}
    for job_path, dest in files.items():
        dest = os.path.expanduser(dest)
        if not os.path.isabs(dest):
            return {"error": f"local destination must be an absolute path: {dest!r}"}
        targets[job_path] = dest
    try:
        # Status before the download decides completeness (a job that was already finished can't
        # change its files mid-download); status after is what we report.
        before = get_status(job_uuid)["status"]
        data, _ = run_cli(["download", job_uuid, "--", *targets], binary=True, timeout=1800)
        status = get_status(job_uuid)
        placed = {}
        with tempfile.TemporaryDirectory(prefix="gpu_runner_dl_") as tmp:
            _safe_extract(data, tmp)
            for job_path, dest in targets.items():
                src = os.path.join(tmp, os.path.normpath(job_path))
                if not os.path.lexists(src):
                    raise RunnerError(f"{job_path} missing from the downloaded archive")
                placed[job_path] = _place(src, dest, job_path, overwrite)
    except (RunnerError, OSError, tarfile.TarError) as e:
        return {"error": str(e)}
    result = {"downloaded": placed, "bytes": len(data), "job_status": status["status"]}
    if before not in TERMINAL:
        result["warning"] = "job was still running during the download; files may be incomplete"
    return result


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