"""JobRunner: runs a single Job on the daemon's GPUs, from RUNNING to a terminal state.
BwrapJobRunner does the same with each command inside a bubblewrap sandbox."""
import os
import shutil
import signal
import subprocess
import threading
import time

from config import JOB_CACHE, JOB_UV_CACHE
from utils import TERMINAL, safe_extract


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


class JobRunner:
    """Runs one Job on a set of GPUs and owns its lifecycle from RUNNING to a terminal state:
    sets up its directory, runs its commands in order, kills whatever they leave behind, and
    handles cancellation of the running job. Knows nothing about the queue; the Daemon decides
    which job runs when.

        runner = JobRunner(job, gpus); runner.start()   # QUEUED -> RUNNING, runs in a thread
        runner.cancel(timeout)                          # kill it; returns once it's terminal
        runner.done()                                   # True once terminal (GPUs are free)
    """

    def __init__(self, job, gpus, verbose=False):
        self.job = job
        self.gpus = list(gpus)
        self.verbose = verbose

    def start(self):
        """Move the job to RUNNING and run it in a background thread."""
        with self.job.mutex:
            self.job.transition("RUNNING")
        threading.Thread(target=self._run, daemon=True).start()

    def done(self):
        with self.job.mutex:
            return self.job.status in TERMINAL

    def cancel(self, timeout=30):
        """Kill all the job's process groups (SIGTERM, then SIGKILL) and skip its remaining
        commands; waits up to `timeout` s for it to end CANCELLED, so its GPUs are free on
        return. No-op for a job that already finished."""
        with self.job.mutex:
            if self.job.status != "RUNNING":
                return
            self.job.cancel_requested = True  # _run_commands turns this into CANCELLED
        self._kill_leftovers()  # no new groups can appear once cancel_requested is set
        self.job.finished.wait(timeout)

    # ------------------------------------------------------------ execution
    def _setup(self):
        """Create the job's directory and extract its tarball into it (safely)."""
        job = self.job
        if self.verbose:
            print(f"[gpu_runner] setting up job {job.job_uuid} in {job.workdir}", flush=True)
        os.makedirs(job.workdir)
        if job.tarfile:
            safe_extract(job.tarfile, job.workdir)
        job.tarfile = None  # free the upload's memory

    def _run(self):
        """Runs the RUNNING job (in its own thread) and always moves it to a terminal state.
        Never raises, so the daemon can always move on to the next job.

        The terminal state is entered only after the job's leftover processes are dead, so a
        terminal status means the GPUs are free: the daemon starts the next job on it."""
        job = self.job
        outcome, failed_command = "FAILED", None
        try:
            self._setup()
            outcome, failed_command = self._run_commands()
        except Exception as e:
            with open(job.stderr_path, "a") as err:
                err.write(f"\n[gpu_runner] job setup/run error: {type(e).__name__}: {e}\n")
        finally:
            try:
                self._kill_leftovers()
                self._cleanup()
            except Exception as e:
                with open(job.stderr_path, "a") as err:
                    err.write(f"\n[gpu_runner] job cleanup error: {type(e).__name__}: {e}\n")
            finally:
                with job.mutex:
                    job.failed_command = failed_command
                    job.transition(outcome)

    def _cleanup(self):
        """Hook: undo _setup's extras once the job's processes are dead (the job dir stays)."""

    def _spawn(self, command, env, out, err):
        """Hook: start one command in its own session (so its process group is killable)."""
        return subprocess.Popen(command, cwd=self.job.workdir, shell=True, env=env,
                                stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                start_new_session=True)

    def _run_commands(self):
        """Run the job's commands in order; returns (outcome, failed_command) where outcome is
        the terminal state the job should end in. Updates exit_code/pid/pgids as it goes."""
        job = self.job
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
                    process = self._spawn(command, env, out, err)
                job.pid = process.pid
                job.pgids.append(process.pid)  # its own process group (new session)
            process.wait()
            with job.mutex:
                job.exit_code = process.returncode
                if process.returncode != 0:
                    return ("CANCELLED", None) if job.cancel_requested else ("FAILED", command)
        return "COMPLETED", None

    def _kill_leftovers(self):
        """Kill background processes the job's commands left behind (they could hold a GPU).
        Commands may start helpers used by later commands, so this runs once, at job end."""
        for sig in (signal.SIGTERM, signal.SIGKILL):
            alive = False
            for pgid in self.job.pgids:
                try:
                    os.killpg(pgid, sig)
                    alive = True
                except (ProcessLookupError, PermissionError):
                    pass
            if not alive:
                return
            time.sleep(1)


class BwrapJobRunner(JobRunner):
    """A JobRunner that runs each command inside a bubblewrap (bwrap) sandbox:

      * the whole filesystem is read-only, except the job's own directory, the job caches
        (UV_CACHE_DIR / XDG_CACHE_HOME) and a private /tmp that lives for the whole job
        (so files left in /tmp by one command are seen by the next);
      * the daemon's workdir is hidden behind an empty tmpfs, so a job can't see other jobs;
      * /dev is passed through (the GPUs), the network is shared, and the sandbox dies with
        the daemon.

    No new pid namespace or session: the command stays in the process group the runner kills,
    so leftover processes and cancellation behave exactly as with the plain JobRunner (and
    helpers a command starts in the background survive for later commands)."""

    BWRAP = "bwrap"

    @classmethod
    def check(cls):
        """Raise RuntimeError unless bwrap is installed and can create a sandbox here
        (unprivileged user namespaces can be disabled by the kernel or an LSM)."""
        if shutil.which(cls.BWRAP) is None:
            raise RuntimeError(f"{cls.BWRAP} not found on PATH")
        p = subprocess.run([cls.BWRAP, "--ro-bind", "/", "/", "--dev-bind", "/dev", "/dev",
                            "true"], capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"{cls.BWRAP} can't create a sandbox: {p.stderr.strip()}")

    @property
    def tmpdir(self):
        return self.job.workdir + ".tmp"  # next to the job dir, like its .stdout/.stderr

    def _setup(self):
        super()._setup()
        os.makedirs(self.tmpdir)
        for d in (JOB_UV_CACHE, JOB_CACHE):
            os.makedirs(d, exist_ok=True)  # bwrap can only bind directories that exist

    def _cleanup(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def bwrap_argv(self, command):
        # Mounts apply in order, each on top of the earlier ones (sources are host paths), so
        # /tmp goes first: the daemon workdir or caches may themselves be under /tmp.
        workdir = self.job.workdir
        args = [self.BWRAP,
                "--ro-bind", "/", "/",
                "--dev-bind", "/dev", "/dev",
                "--bind", self.tmpdir, "/tmp",
                "--tmpfs", os.path.dirname(workdir),  # hide the other jobs
                "--bind", workdir, workdir]
        for d in (JOB_UV_CACHE, JOB_CACHE):
            args += ["--bind", d, d]
        return args + ["--die-with-parent", "--chdir", workdir, "/bin/sh", "-c", command]

    def _spawn(self, command, env, out, err):
        return subprocess.Popen(self.bwrap_argv(command), env=env, stdin=subprocess.DEVNULL,
                                stdout=out, stderr=err, start_new_session=True)


RUNNERS = {"plain": JobRunner, "bwrap": BwrapJobRunner}
