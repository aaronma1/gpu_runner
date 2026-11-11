"""Client side of the CLI: each command connects to the daemon on localhost:PORT over rpyc."""
import sys
import time

import rpyc

from config import PORT
from utils import TERMINAL


def connect():
    try:
        return rpyc.connect("localhost", PORT, config={"sync_request_timeout": 600})
    except ConnectionRefusedError:
        sys.exit(f"gpu_runner: no daemon listening on localhost:{PORT}")


def print_status(status, exit_code, failed):
    print(status, "" if exit_code is None else exit_code, f"(failed: {failed})" if failed else "")


def submit(tarfile_path, commands, message_id):
    """Queue a job; returns its UUID. `tarfile_path` '-' reads stdin, '' sends no tarball."""
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


def query(job_uuid, status=False, stdout=False, stderr=False, wait=False):
    """Print a job's status and/or output (all of them if none is asked for). With `wait`,
    block until it finishes and exit 0 iff it COMPLETED."""
    conn = connect()
    try:
        st, exit_code, failed = conn.root.get_job_status(job_uuid)
        while wait and st not in TERMINAL:
            time.sleep(1)
            st, exit_code, failed = conn.root.get_job_status(job_uuid)
        out, err = conn.root.get_job_out(job_uuid)
    except KeyError:
        sys.exit(f"gpu_runner: no such job {job_uuid}")
    finally:
        conn.close()
    show_all = not (status or stdout or stderr)
    if status or show_all or wait:
        print_status(st, exit_code, failed)
    if stdout or show_all:
        sys.stdout.write(out)
    if stderr or show_all:
        sys.stderr.write(err)
    if wait:
        sys.exit(0 if st == "COMPLETED" else 1)


def download(job_uuid, files, output="-"):
    """Write a tar.gz of `files` from the job's directory to `output` ('-' = stdout)."""
    if output == "-" and sys.stdout.isatty():
        sys.exit("gpu_runner: refusing to write a tar.gz to a terminal; use -o FILE or redirect")
    conn = connect()
    try:
        data = conn.root.download_job_files(job_uuid, list(files))
    except (KeyError, FileNotFoundError, ValueError) as e:
        sys.exit(f"gpu_runner: {e.args[0] if e.args else e}")
    finally:
        conn.close()
    if output == "-":
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
    else:
        try:
            with open(output, "wb") as f:
                f.write(data)
        except OSError as e:
            sys.exit(f"gpu_runner: cannot write {output}: {e}")


def cancel(job_uuid):
    conn = connect()
    try:
        print_status(*conn.root.cancel_job(job_uuid))
    except KeyError:
        sys.exit(f"gpu_runner: no such job {job_uuid}")
    finally:
        conn.close()


def list_jobs():
    conn = connect()
    try:
        for jid, status, first_cmd in conn.root.list_jobs():
            print(f"{jid}  {status:<9}  {first_cmd}")
    finally:
        conn.close()
