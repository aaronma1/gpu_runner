# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp>=1.2,<2"]
# ///
"""MCP server for gpu_runner: lets agents submit jobs, check them, and download their files.

Every tool shells out to the `gpu_runner.py` CLI (locally, or over ssh when SSH_HOST is set).
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

from mcp.server.mcpserver import MCPServer as MCP

# ---------------------------------------------------------------- hardcoded for now (env overrides)
GPURUN = os.environ.get("GPU_RUNNER_MCP_GPURUN", "/home/a84456917/Projects/gpu_runner/gpu_runner.py")
PYTHON = os.environ.get("GPU_RUNNER_MCP_PYTHON",
                        "/home/a84456917/Projects/gpu_runner/.venv/bin/python")  # env with rpyc
DAEMON_WORKDIR = os.environ.get("GPU_RUNNER_MCP_WORKDIR", "/home/a84456917/.gpu_runner")
SSH_HOST = os.environ.get("GPU_RUNNER_MCP_SSH_HOST") or "gpu"  # e.g. "gpu" to go over ssh

TERMINAL = {"COMPLETED", "FAILED"}
EXCLUDE_DIRS = {".git", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache",
                ".ipynb_checkpoints", "node_modules", "wandb"}
MAX_UPLOAD_MB = 500
SUBMIT_RETRIES = 3

mcp = MCP("gpu_runner")


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
    """Queue a job on the GPU box. Uploads local_dir (minus .git/.venv/caches, max 500 MB) and
    runs `commands` in it, in order, as shell commands; the first failing command stops the job.
    Jobs run one at a time and each gets all the runner's GPUs (CUDA_VISIBLE_DEVICES is set).
    Commands get no stdin. Write results to files in the job directory (e.g. out/) and fetch
    them with download_job_files.
    There is no cancel and no time limit, so make sure the job terminates.
    Returns the job_uuid. Safe to call once per job: transport retries reuse one message_id."""
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
    """Status of one job: QUEUED, RUNNING, COMPLETED or FAILED, its exit code, the command that
    failed (if any) and job_dir, the job's working directory on the GPU box."""
    try:
        return get_status(job_uuid)
    except RunnerError as e:
        return {"error": str(e)}


@mcp.tool()
def gpu_output(job_uuid: str, tail_lines: int = 200) -> dict:
    """stdout and stderr of a job (so far, if it is still running): the last `tail_lines` lines
    of each (0 = everything)."""
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
def download_job_files(job_uuid: str, files: dict[str, str], overwrite: bool = True) -> dict:
    """Download files from a job's directory on the GPU box to this machine.
    `files` maps job paths to local destinations, e.g.
      {"out/metrics.json": "/home/me/exp1/metrics.json", "out/plots": "/home/me/exp1/plots"}.
    Job paths are relative to the job's directory ("." = everything). Local destinations must be
    absolute. A directory merges into its destination; a file is written to its destination, or
    inside it if the destination is an existing directory or ends with "/".
    Nothing is written if any job path is missing. Total size is capped (1 GB by default), so
    fetch metrics, logs and plots rather than every checkpoint. Works on a running job, but its
    files may be incomplete; the result includes the job's status."""
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
    """All jobs the daemon knows (since it last started): uuid, status, first command."""
    try:
        out, _ = run_cli(["list"], timeout=60)
        return out or "(no jobs)"
    except RunnerError as e:
        return f"error: {e}"


if __name__ == "__main__":
    mcp.run()