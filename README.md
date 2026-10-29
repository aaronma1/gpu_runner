# GPU RUNNER

A single-queue GPU job daemon. A job is a tarball (extracted into the job's own directory) plus a
list of shell commands, run in order on the daemon's GPUs. Jobs run one at a time; each gets all
the daemon's GPUs through `CUDA_VISIBLE_DEVICES`. An MCP server (`gpu_runner_mcp.py`) lets
agents such as Claude Code submit and manage jobs, either on this machine or on a GPU box reached
over ssh.

## Install

On the GPU machine:

```sh
make install        # ~/.local/bin/gpu_runner, on its own venv in ~/.local/share/gpu_runner/venv
gpu_runner daemon --gpu 0 1
```

`make install` copies `gpu_runner.py`, so re-run it after changing that file. `make uninstall`
removes the command and its venv. Set `PREFIX`, `BINDIR` or `LIBDIR` to install elsewhere.

On the machine where Claude Code runs (this can be the same machine):

```sh
make install-mcp    # asks which host runs the daemon, writes the [mcp] config, registers the server
```

The installer checks that `gpu_runner list` works, over `ssh HOST .local/bin/gpu_runner list` for
a remote host. That needs key-based ssh (`BatchMode=yes`) and the daemon running. `make
uninstall-mcp` unregisters the server and removes the script.

### Restricting the ssh key

`make install` also installs `~/.local/share/gpu_runner/gate`, a forced command for
`authorized_keys` that only lets `gpu_runner submit|query|list|download|cancel` through. It
splits the command with `shlex` and never passes it to a shell:

```
command="/home/you/.local/share/gpu_runner/gate",no-port-forwarding,no-X11-forwarding,no-agent-forwarding,no-pty ssh-ed25519 AAAA...
```

## CLI

```sh
gpu_runner daemon --gpu 0 1 [--workdir DIR] [--verbose]   # --cpu: run with no GPUs
gpu_runner submit job.tar "python train.py" "python eval.py" [--message_id ID]
tar -czf - -C exp . | gpu_runner submit - "python train.py"  # tarball from stdin
gpu_runner query JOB_UUID [--status] [--stdout] [--stderr] [--wait]
gpu_runner download JOB_UUID out/metrics.json out/plots -o results.tar.gz
gpu_runner cancel JOB_UUID
gpu_runner list
```

From another machine (the rpyc port only listens on localhost):

```sh
ssh gpu .local/bin/gpu_runner submit - "python train.py" --message_id $(uuidgen) < job.tar
ssh gpu .local/bin/gpu_runner download JOB_UUID out > results.tar.gz
```

### Job semantics

- Commands run in the job directory with no stdin. The first command that fails stops the job
  and marks it `FAILED`. Otherwise the job ends `COMPLETED`.
- stdout and stderr go to `<workdir>/<uuid>.stdout` and `.stderr`. You can query them while the
  job runs.
- When a job ends, any background processes it left running are killed.
- `cancel` removes a queued job from the queue. For a running job, it kills all the job's
  processes, skips the remaining commands, and marks the job `CANCELLED`.
- Submitting twice with the same `--message_id` returns the existing job instead of queueing a
  duplicate.
- Jobs run without the daemon's virtualenv, so `python` and `uv` use the job's own environment.
  `UV_CACHE_DIR` and `XDG_CACHE_HOME` point at shared caches.
- Job state is kept in memory, so restarting the daemon forgets all jobs.

## Configuration

Both programs read `$GPURUN_CONFIG_DIR/config.toml` (default `~/.config/gpu_runner/config.toml`).

```toml
[runner]                                        # the daemon; GPU_RUNNER_<KEY> env vars override
workdir = "~/data/.gpu_runner"                  # job directories and their stdout/stderr
job_uv_cache = "~/data/.cache/gpu_runner/uv"    # UV_CACHE_DIR for jobs
job_cache = "~/data/.cache/gpu_runner"          # XDG_CACHE_HOME for jobs
port = 18861                                    # rpyc port (localhost only)

[mcp]                                           # written by `make install-mcp`
ssh_host = "gpu"                                # ssh alias of the GPU box; "" = this machine
daemon_workdir = "/home/me/data/.gpu_runner"    # local daemon only
# gpurun = ".../gpu_runner.py"                  # local only: override which CLI to run
# python = ".../.venv/bin/python"               # and its interpreter ("" runs gpurun directly)
```

## MCP tools

| tool | what it does |
| --- | --- |
| `gpu_submit` | uploads a local directory (excluding `.git`, `.venv` and caches; max 500 MB) and queues commands |
| `gpu_status` | gets a job's status and exit code |
| `gpu_output` | gets the tail of a job's stdout and stderr |
| `gpu_wait` | waits for a job to finish, up to a timeout |
| `gpu_cancel` | cancels a job |
| `download_job_files` | copies files from a job directory to local paths |
| `gpu_list` | lists all jobs |

## Development

```sh
uv sync
uv run pytest -q test_gpu_runner.py                 # everything (~1-2 min)
uv run pytest -q test_gpu_runner.py -k "not slow"   # skip the throughput/size tests
```
