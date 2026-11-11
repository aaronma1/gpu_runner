"""Job: one submitted job and its state machine."""
import threading
import time

from utils import TERMINAL


class Job:
    """A job and its lifecycle, a state machine changed only through transition():

        QUEUED --> RUNNING --> COMPLETED | FAILED | CANCELLED
           \\--> CANCELLED                (cancelled before it started)

    Entered by: RUNNING from JobRunner.start; CANCELLED from QUEUED by Daemon.cancel_job; the
    terminal states from RUNNING by JobRunner._run, after the job's processes are dead. A
    cancel of a RUNNING job (JobRunner.cancel) only sets cancel_requested; _run turns it into
    CANCELLED."""

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
