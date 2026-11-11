#!/usr/bin/env python3
"""gpu_runner — single-queue GPU job daemon with an rpyc interface (localhost only).

    gpu_runner.py daemon --gpu 0 1                       # run jobs one at a time on GPUs 0,1
    gpu_runner.py submit job.tar "python train.py" "python eval.py" [--message_id ID]
    tar -czf - -C exp . | gpu_runner.py submit - "python train.py"   # tarball from stdin
    gpu_runner.py query JOB_UUID --status --stdout --stderr [--wait]
    gpu_runner.py download JOB_UUID out/metrics.json out/plots -o results.tar.gz
    gpu_runner.py cancel JOB_UUID                        # dequeue it, or kill it if running

Over ssh from another machine (the rpyc port only listens on localhost):
    ssh gpu gpu_runner.py submit - "python train.py" --message_id $(uuidgen) < job.tar
    ssh gpu gpu_runner.py download JOB_UUID out > results.tar.gz

A job is a tarball (extracted into the job's directory) plus shell commands run in order;
the first failing command stops the job (FAILED). Commands get no stdin. Output goes to
<workdir>/<uuid>.stdout/.stderr and can be queried while the job runs. Processes a job
leaves running in the background are killed when the job ends. Cancelling a queued job
removes it from the queue; cancelling a running one kills all its processes and skips its
remaining commands (CANCELLED). Submitting twice with the
same message_id returns the same job instead of queueing a duplicate. Job state is in
memory: restarting the daemon forgets all jobs.

Settings are read at startup from the [runner] table of $GPURUN_CONFIG_DIR/config.toml
(default ~/.config/gpu_runner/config.toml); GPU_RUNNER_<KEY> environment variables override it:
    [runner]
    workdir = "~/data/.gpu_runner"                  # job directories and their stdout/stderr
    job_uv_cache = "~/data/.cache/gpu_runner/uv"    # UV_CACHE_DIR for jobs
    job_cache = "~/data/.cache/gpu_runner"          # XDG_CACHE_HOME for jobs
    port = 18861                                    # rpyc port (localhost only)
"""
import argparse
import os
import signal
import subprocess
import sys
import threading
import time
import uuid

import rpyc

from gpu_runner_utils import TERMINAL, is_within, load_config, safe_extract, tar_gz


def setting(config, key, default):
    """GPU_RUNNER_<KEY> from the environment, else the config value, else `default`."""
    return os.environ.get(f"GPU_RUNNER_{key.upper()}", config.get(key, default))


CONFIG = load_config("runner", "gpu_runner")
DAEMON_WORKDIR = os.path.expanduser(setting(CONFIG, "workdir", "~/data/.gpu_runner"))
JOB_UV_CACHE = os.path.expanduser(setting(CONFIG, "job_uv_cache", "~/data/.cache/gpu_runner/uv"))
JOB_CACHE = os.path.expanduser(setting(CONFIG, "job_cache", "~/data/.cache/gpu_runner"))

PORT = int(setting(CONFIG, "port", 18861))
MAX_DOWNLOAD_BYTES = 512 * 1024 * 1024  # 512 MiB; downloads are built in memory, so cap them


def job_env(gpus):
    """The environment a job's commands run in: the daemon's, minus the daemon's own venv.

    The daemon is normally started from an activated virtualenv, so VIRTUAL_ENV and that venv's
    bin (first on PATH) would otherwise leak into every job. Tools that resolve the environment
    themselves then pick the wrong one: `uv pip install` installs into the daemon's venv instead
    of the job's ./.venv and still exits 0, and a bare `python` runs the daemon's interpreter.
    Both fail confusingly much later, so strip the venv and let jobs use their own.
    """
    env = dict(os.environ)  # keep PATH, HOME, ...; only override the GPU selection
    venv = env.pop("VIRTUAL_ENV", None)
    env.pop("VIRTUAL_ENV_PROMPT", None)
    if venv:
        venv_bin = os.path.join(os.path.realpath(venv), "bin")
        env["PATH"] = os.pathsep.join(p for p in env.get("PATH", "").split(os.pathsep)
                                      if p and os.path.realpath(p) != venv_bin)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so indices match nvidia-smi
    env["UV_CACHE_DIR"] = JOB_UV_CACHE
    env["XDG_CACHE_HOME"] = JOB_CACHE
    
    return env


class Daemon(rpyc.Service):

    def __init__(self, workdir, gpus=None, poll=1.0, verbose=False):
        self.workdir = workdir
        self.gpus = list(gpus or [])
        self.poll = poll

        self.mutex = threading.Lock()
        self.messages_received = {}  # message_id -> job_uuid       (mutex protected)
        self.jobs = {}               # job_uuid -> Job              (mutex protected)
        self.ready = []              # queued jobs, FIFO            (mutex protected)
        self.running = None          # the one running job, or None (mutex protected)

        self.verbose = verbose

    # ------------------------------------------------------------ job execution
    def _setup_job(self, job):
        """Create the job's directory and extract its tarball into it (safely)."""
        if self.verbose:
            print(f"[gpu_runner] setting up job {job.job_uuid} in {job.workdir}", flush=True)
        os.makedirs(job.workdir)
        if job.tarfile:
            safe_extract(job.tarfile, job.workdir)
        job.tarfile = None  # free the upload's memory

    def _run_job(self, job):
        """Runs a RUNNING job in its own thread and always moves it to a terminal state. Never
        raises, so the daemon can always move on to the next job.

        The terminal state is entered only after the job's leftover processes are dead, so a
        terminal status means the GPUs are free: _loop starts the next job on it."""
        outcome, failed_command = "FAILED", None
        try:
            self._setup_job(job)
            outcome, failed_command = self._run_commands(job)
        except Exception as e:
            with open(job.stderr_path, "a") as err:
                err.write(f"\n[gpu_runner] job setup/run error: {type(e).__name__}: {e}\n")
        finally:
            try:
                self._kill_leftovers(job)
            finally:
                with job.mutex:
                    job.failed_command = failed_command
                    job.transition(outcome)

    def _run_commands(self, job):
        """Run the job's commands in order; returns (outcome, failed_command) where outcome is
        the terminal state the job should end in. Updates exit_code/pid/pgids as it goes."""
        env = job_env(self.gpus)
        for command in job.commands:
            # Output goes to files, not pipes: with pipes, a background child that inherits
            # them keeps the job "running" until it exits. stdin is /dev/null so a command
            # that reads input gets EOF instead of blocking on the daemon's stdin.
            # Spawned under the job's mutex so a cancel either sees this process group or
            # stops it from being started.
            with job.mutex:
                if job.cancel_requested:
                    return "CANCELLED", None
                with open(job.stdout_path, "ab") as out, open(job.stderr_path, "ab") as err:
                    process = subprocess.Popen(command, cwd=job.workdir, shell=True, env=env,
                                               stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                               start_new_session=True)
                job.pid = process.pid
                job.pgids.append(process.pid)  # its own process group (new session)
            process.wait()
            with job.mutex:
                job.exit_code = process.returncode
                if process.returncode != 0:
                    return ("CANCELLED", None) if job.cancel_requested else ("FAILED", command)
        return "COMPLETED", None

    @staticmethod
    def _kill_leftovers(job):
        """Kill background processes the job's commands left behind (they could hold a GPU).
        Commands may start helpers used by later commands, so this runs once, at job end."""
        for sig in (signal.SIGTERM, signal.SIGKILL):
            alive = False
            for pgid in job.pgids:
                try:
                    os.killpg(pgid, sig)
                    alive = True
                except (ProcessLookupError, PermissionError):
                    pass
            if not alive:
                return
            time.sleep(1)

    def _loop(self):
        while True:
            with self.mutex:
                if self.running is not None:
                    with self.running.mutex:
                        done = self.running.status in TERMINAL
                        status = self.running.status
                    if done:
                        print(f"Job {self.running.job_uuid} {status}.", flush=True)
                        self.running = None
                if self.running is None and self.ready:
                    job = self.ready.pop(0)
                    with job.mutex:
                        job.transition("RUNNING")
                    self.running = job
                    print(f"Job {job.job_uuid} started.", flush=True)
                    threading.Thread(target=self._run_job, args=(job,), daemon=True).start()
            time.sleep(self.poll)

    # ------------------------------------------------------------ rpyc interface
    def exposed_receive_job(self, message_id, tarfile_bytes, commands):
        # rpyc passes str/bytes by value but lists by reference (a proxy to the client's list,
        # dead once the client disconnects), so copy everything into local objects now.
        if self.verbose:
            print(f"[gpu_runner] received job with message_id {message_id} and commands {commands}", flush=True)
        message_id = str(message_id)
        commands = [str(c) for c in commands]
        tarfile_bytes = bytes(tarfile_bytes) if tarfile_bytes else None
        if not commands:
            raise ValueError("at least one command is required")
        with self.mutex:
            if message_id in self.messages_received:
                print(f"Message {message_id} has already been processed. Returning its job.", flush=True)
                return self.messages_received[message_id]
            job_uuid = str(uuid.uuid4())
            job = Job(job_uuid, tarfile_bytes, commands, os.path.join(self.workdir, job_uuid))
            self.messages_received[message_id] = job_uuid
            self.jobs[job_uuid] = job
            self.ready.append(job)
        return job_uuid

    def _job(self, job_uuid):
        with self.mutex:
            job = self.jobs.get(str(job_uuid))
        if job is None:
            raise KeyError(f"no such job {job_uuid}")
        return job

    def exposed_get_job_out(self, job_uuid):
        """(stdout, stderr) so far; works while the job is running."""
        job = self._job(job_uuid)
        return _read(job.stdout_path), _read(job.stderr_path)  # str tuple: passed by value

    def exposed_get_job_status(self, job_uuid):
        job = self._job(job_uuid)
        with job.mutex:
            return job.status, job.exit_code, job.failed_command

    def exposed_download_job_files(self, job_uuid, files):
        """tar.gz (bytes) of `files`, paths relative to the job's directory ('.' = everything).
        Directories are included recursively; each entry is stored under its normalized
        requested name. All or nothing: FileNotFoundError lists every missing path; ValueError
        for paths outside the job directory (including via symlinks) or over MAX_DOWNLOAD_BYTES.
        Works on a running job (files may be mid-write). Runs in the caller's connection
        thread, so it never blocks the job loop."""
        job = self._job(job_uuid)
        files = [str(f) for f in files]  # copy the client's list (a netref) while connected
        if not files:
            raise ValueError("no files requested")
        root = os.path.realpath(job.workdir)
        if not os.path.isdir(root):
            with job.mutex:
                status = job.status
            raise FileNotFoundError(f"job {job_uuid} has no directory yet (status {status})")

        entries, missing = [], []
        for f in files:
            arc = os.path.normpath(f)
            if os.path.isabs(f) or arc == ".." or arc.startswith(".." + os.sep):
                raise ValueError(f"{f!r} is not a path inside the job directory")
            real = os.path.realpath(os.path.join(root, arc))  # follows symlinks
            if not is_within(real, root):
                raise ValueError(f"{f!r} resolves outside the job directory")
            if os.path.exists(real):
                entries.append((real, arc))
            else:
                missing.append(f)
        if missing:
            raise FileNotFoundError(f"not found in job {job_uuid}: {', '.join(missing)}")

        total = sum(_tree_size(real) for real, _ in entries)  # before building it in memory
        if total > MAX_DOWNLOAD_BYTES:
            raise ValueError(f"requested files total {total / 2**20:.1f} MB, over the "
                             f"{MAX_DOWNLOAD_BYTES / 2**20:.0f} MB download limit")

        def drop_escaping_links(ti):
            # A symlink inside a requested directory is stored as a link, not followed. One that
            # points outside the job dir would let a job smuggle an absolute/../ link into the
            # client's extraction, so leave it out.
            if ti.issym() or ti.islnk():
                target = os.path.realpath(os.path.join(root, os.path.dirname(ti.name), ti.linkname))
                if not is_within(target, root):
                    return None
            return ti

        # arcnames are relative to root, so the filter can resolve links against it
        return tar_gz(entries, filter=drop_escaping_links)  # bytes: passed by value

    def exposed_cancel_job(self, job_uuid, timeout=30):
        """Cancel a job; returns (status, exit_code, failed_command) like get_job_status.
        A queued job is dropped from the queue and never runs. A running job has all its
        process groups killed (SIGTERM, then SIGKILL) and its remaining commands skipped; this
        waits up to `timeout` s for it to wind down, so its GPUs are free on return. Either
        way it ends CANCELLED. A job that already finished is left as it is."""
        job = self._job(job_uuid)
        with self.mutex:  # same lock order as _loop: daemon, then job
            with job.mutex:
                running = job.status == "RUNNING"
                if job.status == "QUEUED":
                    self.ready.remove(job)
                    job.transition("CANCELLED")
                    print(f"Job {job.job_uuid} CANCELLED (queued).", flush=True)
                elif running:
                    job.cancel_requested = True  # _run_commands turns this into CANCELLED
        if running:
            self._kill_leftovers(job)  # no new groups can appear once cancel_requested is set
            job.finished.wait(timeout)
        with job.mutex:
            return job.status, job.exit_code, job.failed_command

    def exposed_list_jobs(self):
        with self.mutex:
            jobs = list(self.jobs.values())
        out = []
        for j in jobs:
            with j.mutex:
                out.append((j.job_uuid, j.status, j.commands[0]))
        return tuple(out)


def _tree_size(path):
    """Bytes in a file, or in all regular files under a directory (symlinks not followed)."""
    if not os.path.isdir(path):
        return os.path.getsize(path)
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for name in filenames:
            try:
                st = os.lstat(os.path.join(dirpath, name))
            except FileNotFoundError:
                continue  # deleted mid-walk (running job)
            if not (st.st_mode & 0o170000 == 0o120000):  # skip symlinks
                total += st.st_size
    return total


def _read(path):
    try:
        with open(path, "rb") as f:
            return f.read().decode(errors="replace")
    except FileNotFoundError:
        return ""


class Job:
    """A job and its lifecycle, a state machine changed only through transition():

        QUEUED --> RUNNING --> COMPLETED | FAILED | CANCELLED
           \\--> CANCELLED                (cancelled before it started)

    Entered by: RUNNING from Daemon._loop; CANCELLED from QUEUED by cancel_job; the terminal
    states from RUNNING by Daemon._run_job, after the job's processes are dead. A cancel of a
    RUNNING job only sets cancel_requested; _run_job turns it into CANCELLED."""

    TRANSITIONS = {"QUEUED": {"RUNNING", "CANCELLED"}, "RUNNING": TERMINAL}

    def __init__(self, job_uuid, tarfile, commands, workdir):

        # immutable attributes
        self.job_uuid = job_uuid
        self.commands = commands
        self.workdir = workdir
        self.stdout_path = workdir + ".stdout"   # next to the job dir, not inside it
        self.stderr_path = workdir + ".stderr"

        self.mutex = threading.Lock()
        self.status = "QUEUED"       # see TRANSITIONS; change only via transition() (mutex protected)
        self.tarfile = tarfile       # bytes; dropped after extraction or on finishing (mutex protected)
        self.pid = None              # current command's pid while RUNNING (mutex protected)
        self.exit_code = None        # last command's exit code (mutex protected)
        self.failed_command = None   # set on entering FAILED (mutex protected)
        self.pgids = []              # process groups of its commands (mutex protected)
        self.cancel_requested = False  # set once by cancel_job while RUNNING (mutex protected)
        self.finished = threading.Event()  # set on entering a terminal state
        self.started = self.ended = None   # set on entering RUNNING / a terminal state

    def transition(self, new):
        """Move to state `new`; the caller holds self.mutex. Raises RuntimeError for a move
        TRANSITIONS doesn't allow. Entering a terminal state records the end, drops the upload
        and wakes finished.wait()ers; callers must have killed the job's processes first."""
        if new not in self.TRANSITIONS.get(self.status, ()):
            raise RuntimeError(f"job {self.job_uuid}: illegal transition {self.status} -> {new}")
        self.status = new
        if new == "RUNNING":
            self.started = time.time()
        else:
            self.ended = time.time()
            self.tarfile = None
            self.pid = None
            self.finished.set()


# ------------------------------------------------------------ client
def connect():
    try:
        return rpyc.connect("localhost", PORT, config={"sync_request_timeout": 600})
    except ConnectionRefusedError:
        sys.exit(f"gpu_runner: no daemon listening on localhost:{PORT}")


def submit_job(tarfile_path, commands, message_id):
    if tarfile_path == "-":
        data = sys.stdin.buffer.read()
    elif tarfile_path:
        try:
            with open(tarfile_path, "rb") as f:
                data = f.read()
        except OSError as e:
            sys.exit(f"gpu_runner: cannot read tarball: {e}")
    else:
        data = b""
    conn = connect()
    try:
        return conn.root.receive_job(message_id, data, list(commands))
    finally:
        conn.close()


def print_status(status, exit_code, failed):
    print(status, "" if exit_code is None else exit_code, f"(failed: {failed})" if failed else "")


def main():
    parser = argparse.ArgumentParser(description="Single-queue GPU job daemon.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__.split("\n\n", 1)[1])
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_daemon = subparsers.add_parser("daemon", help="Start the GPU daemon.")
    parser_daemon.add_argument("--gpu", type=int, nargs="+", default=[0],
                               help="GPU indices every job gets (default: 0).")
    parser_daemon.add_argument("--cpu", action="store_true",
                               help="Run jobs with no GPUs (CUDA_VISIBLE_DEVICES empty); overrides --gpu.")
    parser_daemon.add_argument("--workdir", default=DAEMON_WORKDIR)
    parser_daemon.add_argument("--verbose", action="store_true", help="Print verbose output.")

    parser_submit = subparsers.add_parser("submit", help="Submit a job; prints its UUID.")
    parser_submit.add_argument("tarfile", help="Tarball of the job's working directory, '-' for stdin, "
                                               "or '' for none.")
    parser_submit.add_argument("commands", nargs="+", help="Shell commands, run in order.")
    parser_submit.add_argument("--message_id", default=None,
                               help="Idempotency key: resubmitting with the same id returns the same "
                                    "job. Default: a fresh random id (no dedup across retries).")

    parser_query = subparsers.add_parser("query", help="Show a job's status and output.")
    parser_query.add_argument("job_uuid")
    parser_query.add_argument("--status", action="store_true", help="Print status and exit code.")
    parser_query.add_argument("--stdout", action="store_true", help="Print stdout.")
    parser_query.add_argument("--stderr", action="store_true", help="Print stderr.")
    parser_query.add_argument("--wait", action="store_true",
                              help="Block until the job finishes; exit 0 iff COMPLETED.")

    parser_download = subparsers.add_parser(
        "download", help="Write a tar.gz of files from a job's directory.")
    parser_download.add_argument("job_uuid")
    parser_download.add_argument("files", nargs="+",
                                 help="Paths relative to the job directory ('.' = everything).")
    parser_download.add_argument("-o", "--output", default="-",
                                 help="Output file ('-' = stdout, the default; use over ssh).")

    parser_cancel = subparsers.add_parser(
        "cancel", help="Cancel a job (dequeue it, or kill it if running); prints its status.")
    parser_cancel.add_argument("job_uuid")

    subparsers.add_parser("list", help="List all jobs.")

    args = parser.parse_args()

    if args.command == "daemon":
        os.makedirs(args.workdir, exist_ok=True)
        daemon = Daemon(args.workdir, [] if args.cpu else args.gpu, verbose=args.verbose)
        server = rpyc.ThreadedServer(daemon, hostname="localhost", port=PORT,  # NOT 0.0.0.0
                                     protocol_config={"sync_request_timeout": 600})
        threading.Thread(target=server.start, daemon=True).start()
        print(f"GPU daemon on localhost:{PORT}, gpus={[] if args.cpu else args.gpu}, workdir={args.workdir}", flush=True)
        try:
            daemon._loop()  # only returns by raising; anything but Ctrl-C prints its traceback
        except KeyboardInterrupt:
            print("GPU daemon exiting.", flush=True)
        finally:
            server.close()

    elif args.command == "submit":
        job_uuid = submit_job(args.tarfile, args.commands, args.message_id or str(uuid.uuid4()))
        print(job_uuid)

    elif args.command == "query":
        conn = connect()
        try:
            status, exit_code, failed = conn.root.get_job_status(args.job_uuid)
            while args.wait and status not in TERMINAL:
                time.sleep(1)
                status, exit_code, failed = conn.root.get_job_status(args.job_uuid)
            stdout, stderr = conn.root.get_job_out(args.job_uuid)
        except KeyError:
            sys.exit(f"gpu_runner: no such job {args.job_uuid}")
        finally:
            conn.close()
        show_all = not (args.status or args.stdout or args.stderr)
        if args.status or show_all or args.wait:
            print_status(status, exit_code, failed)
        if args.stdout or show_all:
            sys.stdout.write(stdout)
        if args.stderr or show_all:
            sys.stderr.write(stderr)
        if args.wait:
            sys.exit(0 if status == "COMPLETED" else 1)

    elif args.command == "download":
        if args.output == "-" and sys.stdout.isatty():
            sys.exit("gpu_runner: refusing to write a tar.gz to a terminal; use -o FILE or redirect")
        conn = connect()
        try:
            data = conn.root.download_job_files(args.job_uuid, list(args.files))
        except (KeyError, FileNotFoundError, ValueError) as e:
            sys.exit(f"gpu_runner: {e.args[0] if e.args else e}")
        finally:
            conn.close()
        if args.output == "-":
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
        else:
            try:
                with open(args.output, "wb") as f:
                    f.write(data)
            except OSError as e:
                sys.exit(f"gpu_runner: cannot write {args.output}: {e}")

    elif args.command == "cancel":
        conn = connect()
        try:
            print_status(*conn.root.cancel_job(args.job_uuid))
        except KeyError:
            sys.exit(f"gpu_runner: no such job {args.job_uuid}")
        finally:
            conn.close()

    elif args.command == "list":
        conn = connect()
        try:
            for jid, status, first_cmd in conn.root.list_jobs():
                print(f"{jid}  {status:<9}  {first_cmd}")
        finally:
            conn.close()


if __name__ == "__main__":
    main()
