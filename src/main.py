#!/usr/bin/env python3
"""gpu_runner — single-queue GPU job daemon with an rpyc interface (localhost only).

    gpu_runner daemon --gpu 0 1                       # run jobs one at a time on GPUs 0,1
    gpu_runner daemon --gpu 0 --sandbox bwrap         # ... each command in a bubblewrap sandbox
    gpu_runner submit job.tar "python train.py" "python eval.py" [--message_id ID]
    tar -czf - -C exp . | gpu_runner submit - "python train.py"   # tarball from stdin
    gpu_runner query JOB_UUID --status --stdout --stderr [--wait]
    gpu_runner download JOB_UUID out/metrics.json out/plots -o results.tar.gz
    gpu_runner cancel JOB_UUID                        # dequeue it, or kill it if running

Over ssh from another machine (the rpyc port only listens on localhost):
    ssh gpu gpu_runner submit - "python train.py" --message_id $(uuidgen) < job.tar
    ssh gpu gpu_runner download JOB_UUID out > results.tar.gz

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
    sandbox = "plain"                               # job runner: "plain", or "bwrap" (bubblewrap)
"""
import argparse
import sys
import uuid

import client
import daemon
from config import DAEMON_WORKDIR, SANDBOX
from runner import RUNNERS


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
    parser_daemon.add_argument("--sandbox", choices=sorted(RUNNERS), default=SANDBOX,
                               help="How to run jobs: 'plain' processes, or 'bwrap' to run each "
                                    "command in a bubblewrap sandbox (read-only filesystem except "
                                    f"the job's directory, caches and /tmp). Default: {SANDBOX}.")

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
        runner_cls = RUNNERS.get(args.sandbox)
        if runner_cls is None:  # a bad `sandbox` config value (argparse doesn't check defaults)
            sys.exit(f"gpu_runner: unknown sandbox {args.sandbox!r}; one of {sorted(RUNNERS)}")
        try:
            if hasattr(runner_cls, "check"):
                runner_cls.check()
        except RuntimeError as e:
            sys.exit(f"gpu_runner: --sandbox {args.sandbox}: {e}")
        daemon.serve(args.workdir, [] if args.cpu else args.gpu, verbose=args.verbose,
                     runner_cls=runner_cls)
        return

    if args.command == "submit":
        print(client.submit(args.tarfile, args.commands, args.message_id or str(uuid.uuid4())))
    elif args.command == "query":
        client.query(args.job_uuid, args.status, args.stdout, args.stderr, args.wait)
    elif args.command == "download":
        client.download(args.job_uuid, args.files, args.output)
    elif args.command == "cancel":
        client.cancel(args.job_uuid)
    elif args.command == "list":
        client.list_jobs()


if __name__ == "__main__":
    main()
