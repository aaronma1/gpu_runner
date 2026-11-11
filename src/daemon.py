"""Daemon: the job queue and its rpyc interface (localhost only). Runs jobs one at a time,
each through a JobRunner."""
import os
import threading
import time
import uuid

import rpyc

from config import PORT
from job import Job
from runner import JobRunner
from utils import is_within, tar_gz

MAX_DOWNLOAD_BYTES = 512 * 1024 * 1024  # 512 MiB; downloads are built in memory, so cap them


class Daemon(rpyc.Service):
    """The job queue and rpyc interface: accepts jobs, runs them one at a time (each through a
    JobRunner), and answers status/output/download/cancel requests."""

    def __init__(self, workdir, gpus=None, poll=1.0, verbose=False):
        self.workdir = workdir
        self.gpus = list(gpus or [])
        self.poll = poll

        self.mutex = threading.Lock()
        self.messages_received = {}  # message_id -> job_uuid                 (mutex protected)
        self.jobs = {}               # job_uuid -> Job                        (mutex protected)
        self.ready = []              # queued jobs, FIFO                      (mutex protected)
        self.running = None          # JobRunner of the running job, or None (mutex protected)

        self.verbose = verbose

    # ------------------------------------------------------------ scheduling
    def _loop(self):
        while True:
            with self.mutex:
                if self.running is not None and self.running.done():
                    job = self.running.job
                    print(f"Job {job.job_uuid} {job.status}.", flush=True)
                    self.running = None
                if self.running is None and self.ready:
                    job = self.ready.pop(0)
                    self.running = JobRunner(job, self.gpus, verbose=self.verbose)
                    self.running.start()
                    print(f"Job {job.job_uuid} started.", flush=True)
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
        runner = None
        with self.mutex:  # same lock order as _loop: daemon, then job
            with job.mutex:
                if job.status == "QUEUED":
                    self.ready.remove(job)
                    job.transition("CANCELLED")
                    print(f"Job {job.job_uuid} CANCELLED (queued).", flush=True)
            if self.running is not None and self.running.job is job:
                runner = self.running
        if runner is not None:
            runner.cancel(timeout)  # outside the daemon lock: it waits for the job to end
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


def serve(workdir, gpus, verbose=False):
    """Run the daemon in the foreground until Ctrl-C."""
    os.makedirs(workdir, exist_ok=True)
    daemon = Daemon(workdir, gpus, verbose=verbose)
    server = rpyc.ThreadedServer(daemon, hostname="localhost", port=PORT,  # NOT 0.0.0.0
                                 protocol_config={"sync_request_timeout": 600})
    threading.Thread(target=server.start, daemon=True).start()
    print(f"GPU daemon on localhost:{PORT}, gpus={gpus}, workdir={workdir}", flush=True)
    try:
        daemon._loop()  # only returns by raising; anything but Ctrl-C prints its traceback
    except KeyboardInterrupt:
        print("GPU daemon exiting.", flush=True)
    finally:
        server.close()
