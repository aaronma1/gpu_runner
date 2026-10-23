#!/usr/bin/env python3
"""gpu_runner — single-queue GPU job daemon with an rpyc interface (localhost only).

    gpu_runner.py daemon --gpu 0 1                       # run jobs one at a time on GPUs 0,1
    gpu_runner.py submit job.tar "python train.py" "python eval.py" [--message_id ID]
    tar -czf - -C exp . | gpu_runner.py submit - "python train.py"   # tarball from stdin
    gpu_runner.py query JOB_UUID --status --stdout --stderr [--wait]

Over ssh from another machine (the rpyc port only listens on localhost):
    ssh gpu gpu_runner.py submit - "python train.py" --message_id $(uuidgen) < job.tar

A job is a tarball (extracted into the job's directory) plus shell commands run in order;
the first failing command stops the job (FAILED). Commands get no stdin. Output goes to
<workdir>/<uuid>.stdout/.stderr and can be queried while the job runs. Processes a job
leaves running in the background are killed when the job ends. Submitting twice with the
same message_id returns the same job instead of queueing a duplicate. Job state is in
memory: restarting the daemon forgets all jobs.
"""
import argparse
import io
import os
import signal
import subprocess
import sys
import tarfile
import threading
import time
import uuid

import rpyc

DAEMON_WORKDIR = os.path.expanduser(os.environ.get("GPU_RUNNER_WORKDIR", "~/.gpu_runner"))
PORT = int(os.environ.get("GPU_RUNNER_PORT", 18861))
TERMINAL = {"COMPLETED", "FAILED"}


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
            with tarfile.open(fileobj=io.BytesIO(job.tarfile), mode="r:*") as tar:
                if hasattr(tarfile, "data_filter"):
                    tar.extractall(job.workdir, filter="data")  # rejects ../ and absolute paths
                else:
                    root = os.path.realpath(job.workdir)
                    for m in tar.getmembers():
                        target = os.path.realpath(os.path.join(job.workdir, m.name))
                        if target != root and not target.startswith(root + os.sep):
                            raise ValueError(f"unsafe path in tarball: {m.name}")
                    tar.extractall(job.workdir)
        job.tarfile = None  # free the upload's memory

    def _run_job(self, job):
        """Runs in its own thread. Never raises: any error marks the job FAILED, so the
        daemon can always move on to the next job."""
        try:
            self._setup_job(job)
            env = dict(os.environ)  # keep PATH, HOME, ...; only override the GPU selection
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpus))
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so indices match nvidia-smi
            for command in job.commands:
                # Output goes to files, not pipes: with pipes, a background child that inherits
                # them keeps the job "running" until it exits. stdin is /dev/null so a command
                # that reads input gets EOF instead of blocking on the daemon's stdin.
                with open(job.stdout_path, "ab") as out, open(job.stderr_path, "ab") as err:
                    process = subprocess.Popen(command, cwd=job.workdir, shell=True, env=env,
                                               stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                               start_new_session=True)
                with job.mutex:
                    job.pid = process.pid
                    job.pgids.append(process.pid)  # its own process group (new session)
                process.wait()
                with job.mutex:
                    job.exit_code = process.returncode
                if process.returncode != 0:
                    with job.mutex:
                        job.failed_command = command
                        job.status = "FAILED"
                    return
            with job.mutex:
                job.status = "COMPLETED"
        except Exception as e:
            with open(job.stderr_path, "a") as err:
                err.write(f"\n[gpu_runner] job setup/run error: {type(e).__name__}: {e}\n")
            with job.mutex:
                job.status = "FAILED"
        finally:
            self._kill_leftovers(job)
            with job.mutex:
                job.pid = None
                job.ended = time.time()

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
                        job.status = "RUNNING"
                        job.started = time.time()
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

    def exposed_list_jobs(self):
        with self.mutex:
            jobs = list(self.jobs.values())
        out = []
        for j in jobs:
            with j.mutex:
                out.append((j.job_uuid, j.status, j.commands[0]))
        return tuple(out)


def _read(path):
    try:
        with open(path, "rb") as f:
            return f.read().decode(errors="replace")
    except FileNotFoundError:
        return ""


class Job:
    def __init__(self, job_uuid, tarfile, commands, workdir):

        # immutable attributes
        self.job_uuid = job_uuid
        self.commands = commands
        self.workdir = workdir
        self.stdout_path = workdir + ".stdout"   # next to the job dir, not inside it
        self.stderr_path = workdir + ".stderr"

        self.mutex = threading.Lock()
        self.tarfile = tarfile       # bytes; dropped after extraction (mutex protected)
        self.status = "QUEUED"       # QUEUED -> RUNNING -> COMPLETED | FAILED (mutex protected)
        self.pid = None              # mutex protected
        self.exit_code = None        # mutex protected
        self.failed_command = None   # mutex protected
        self.pgids = []              # process groups of its commands (mutex protected)
        self.started = self.ended = None


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


def main():
    parser = argparse.ArgumentParser(description="Single-queue GPU job daemon.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__.split("\n\n", 1)[1])
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_daemon = subparsers.add_parser("daemon", help="Start the GPU daemon.")
    parser_daemon.add_argument("--gpu", type=int, nargs="+", default=[0],
                               help="GPU indices every job gets (default: 0).")
                            
    parser_daemon.add_argument("--cpu", action="store_true", help="CPU indices every job gets (default: 0).")
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
            daemon._loop()
        except:
            print("GPU daemon exiting.", flush=True)
            server.close()
            sys.exit(1)

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
            print(status, "" if exit_code is None else exit_code, f"(failed: {failed})" if failed else "")
        if args.stdout or show_all:
            sys.stdout.write(stdout)
        if args.stderr or show_all:
            sys.stderr.write(stderr)
        if args.wait:
            sys.exit(0 if status == "COMPLETED" else 1)

    elif args.command == "list":
        conn = connect()
        try:
            for jid, status, first_cmd in conn.root.list_jobs():
                print(f"{jid}  {status:<9}  {first_cmd}")
        finally:
            conn.close()


if __name__ == "__main__":
    main()