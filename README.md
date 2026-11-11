# GPU RUNNER

A single-queue GPU job daemon. A job is a tarball (extracted into the job's own directory) plus a
list of shell commands, run in order on the daemon's GPUs. Jobs run one at a time; each gets all
the daemon's GPUs through `CUDA_VISIBLE_DEVICES`. An MCP server (`src/mcp_server.py`) lets
agents such as Claude Code submit and manage jobs, either on this machine or on a GPU box reached
over ssh.

## Install

On the GPU machine:

```sh
make install        # ~/.local/bin/gpu_runner, on its own venv in ~/.local/share/gpu_runner/venv
gpu_runner daemon --gpu 0 1
```

`make install` copies `src/main.py` and the modules it imports into `~/.local/share/gpu_runner`
(linked from `~/.local/bin/gpu_runner`), so re-run it after changing any of them. `make uninstall`
removes the command and its venv. Set `PREFIX`, `BINDIR` or `LIBDIR` to install elsewhere.

On the machine where Claude Code runs (this can be the same machine):

```sh
make install-mcp    # asks which host runs the daemon, writes the [mcp] config, registers the server
```

The installer checks that `gpu_runner list` works, over `ssh HOST .local/bin/gpu_runner list` for
a remote host. That needs key-based ssh (`BatchMode=yes`) and the daemon running. `make
uninstall-mcp` unregisters the server and removes the script.

### jiuwenswarm web UI

The same MCP server works as a jiuwenswarm connector (stdio transport). On the machine where
jiuwenswarm runs:

1. Install and configure the server. Pass `MCP_FLAGS=--no-register` if you don't also want it
   registered with Claude Code:

   ```sh
   make install-mcp MCP_FLAGS=--no-register
   ```

   At the end it prints the JSON for step 3, with your paths filled in.

2. In the web UI, open **Extensions**, then the **Connector** tab, then **Create** →
   **Configure connector**.
3. Paste the printed JSON into **Add JSON config**. It fills in the name, type (stdio), startup
   command and arguments. With the default `PREFIX` it looks like this:

   ```json
   {
     "mcpServers": {
       "gpu_runner": {
         "command": "/home/you/.local/share/gpu_runner/venv/bin/python",
         "args": ["/home/you/.local/share/gpu_runner/mcp_server.py"]
       }
     }
   }
   ```

   Use absolute paths, since the form has no working-directory field. If you set
   `GPURUN_CONFIG_DIR`, add it under **Environment variables** (the printed JSON already includes
   it). If your ssh key to the GPU host is held by an ssh agent rather than a key file, add
   `SSH_AUTH_SOCK` under **Env passthrough**. A stdio MCP subprocess gets only a minimal
   environment.
4. Confirm. The tools (`gpu_submit`, `gpu_status`, ...) show up under the `gpu_runner`
   connector. To change the connector later, use **Edit** on its detail page.

Instead of using the web UI, you can add the server to the `mcp.servers` list in
`~/.jiuwenswarm/config/config.yaml`:

```yaml
mcp:
  servers:
    - name: gpu_runner
      enabled: true
      transport: stdio
      command: /home/you/.local/share/gpu_runner/venv/bin/python
      args: ["/home/you/.local/share/gpu_runner/mcp_server.py"]
```

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
gpu_runner daemon --gpu 0 --sandbox bwrap                 # each command in a bubblewrap sandbox
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

### Sandboxed jobs (`--sandbox bwrap`)

With `--sandbox bwrap` (or `sandbox = "bwrap"` in `[runner]`), each command runs under
[bubblewrap](https://github.com/containers/bubblewrap). The daemon refuses to start if `bwrap` is
missing or can't create a sandbox, for example when unprivileged user namespaces are disabled.
Inside the sandbox:

- The whole filesystem is read-only, except the job's directory, the two job caches, and a
  private `/tmp`. The private `/tmp` lasts for the whole job, so later commands see files that
  earlier ones wrote there. It is deleted when the job ends.
- The daemon's workdir is hidden, so a job can't see other jobs' directories or output.
- `/dev` (and with it the GPUs) and the network are shared with the host.
- Process handling is the same as without a sandbox. Background helpers survive until the job
  ends, and cancel kills everything.

The sandbox keeps a job from changing the host or reading other jobs. It doesn't hide readable
files outside the job, such as those in your home directory.

## Configuration

Both programs read `$GPURUN_CONFIG_DIR/config.toml` (default `~/.config/gpu_runner/config.toml`).

```toml
[runner]                                        # the daemon; GPU_RUNNER_<KEY> env vars override
workdir = "~/data/.gpu_runner"                  # job directories and their stdout/stderr
job_uv_cache = "~/data/.cache/gpu_runner/uv"    # UV_CACHE_DIR for jobs
job_cache = "~/data/.cache/gpu_runner"          # XDG_CACHE_HOME for jobs
port = 18861                                    # rpyc port (localhost only)
sandbox = "plain"                               # job runner: "plain", or "bwrap" (bubblewrap)

[mcp]                                           # written by `make install-mcp`
ssh_host = "gpu"                                # ssh alias of the GPU box; "" = this machine
daemon_workdir = "/home/me/data/.gpu_runner"    # local daemon only
# gpurun = ".../src/main.py"                    # local only: override which CLI to run
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

| file | contents |
| --- | --- |
| `src/main.py` | CLI entrypoint (`gpu_runner`): parses arguments, starts the daemon or runs a client command |
| `src/daemon.py` | `Daemon`: the job queue and rpyc interface; `serve()` runs it |
| `src/runner.py` | `JobRunner`: runs one job on the GPUs, from start to a terminal state, and cancels it; `BwrapJobRunner` does the same inside bubblewrap |
| `src/job.py` | `Job`: a job's data and state machine |
| `src/client.py` | the CLI's client commands (submit, query, download, cancel, list) |
| `src/config.py` | `[runner]` settings |
| `src/utils.py` | helpers shared with the MCP server (config loading, tar handling) |
| `src/mcp_server.py` | the MCP server |
| `src/install_mcp.py` | interactive setup for the MCP server, run by `make install-mcp` |
| `src/gate.py` | the ssh forced command installed by `make install` |

```sh
uv sync
uv run pytest -q test_gpu_runner.py                 # everything (~1-2 min)
uv run pytest -q test_gpu_runner.py -k "not slow"   # skip the throughput/size tests
```
