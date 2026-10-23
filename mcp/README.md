# gpurun MCP — tool reference

The `gpurun` MCP server exposes the `gpu_runner` job queue as five tools for queueing jobs on
a GPU box, tracking them, and reading their output:

`gpu_submit`, `gpu_status`, `gpu_output`, `gpu_wait`, `gpu_list`.

Jobs run **one at a time**, in submission order, each with the runner's full GPU set exposed
via `CUDA_VISIBLE_DEVICES`. There is **no cancel** and **no time limit**.

Every tool returns a JSON object. On failure a tool returns `{"error": "<message>"}` instead of
raising — always check for an `error` key. Typical errors: `Host key verification failed.`,
`Permission denied (publickey,password).`,
`gpu_runner: no daemon listening on localhost:18861`, `gate: not allowed`.

---

## `gpu_submit`

**Signature:** `gpu_submit(local_dir: str, commands: list[str]) -> dict`

Queue a job. Uploads `local_dir` as a gzipped tarball, extracts it into the job's directory,
then runs each entry of `commands` in that directory, in order, as a shell command.

**Parameters**

| Name        | Type        | Required | Description                                          |
| ----------- | ----------- | -------- | ---------------------------------------------------- |
| `local_dir` | `str`       | yes      | Local directory to upload as the job's working tree. |
| `commands`  | `list[str]` | yes      | Shell commands, run in order. Must be non-empty.     |

**Behavior**

- The **first command that exits non-zero stops the job**, which is marked `FAILED`; the
  remaining commands do not run.
- Commands receive **no stdin** (`/dev/null`) — anything that reads input gets EOF.
- Write results to **files inside the job directory** (e.g. `out/`). That is how output
  survives the job and can be read back with `gpu_output`.
- Upload excludes these directories anywhere in the tree: `.git`, `.venv`, `venv`,
  `__pycache__`, `.mypy_cache`, `.pytest_cache`, `.ipynb_checkpoints`, `node_modules`,
  `wandb`; and skips `*.pyc` / `*.pyo`.
- Upload size is capped at **500 MB**; larger uploads return an error. Download large datasets
  *inside* the job rather than uploading them.
- **No cancel, no time limit** — ensure the commands terminate on their own.
- **Idempotent.** Each submission carries an internal id; transport retries reuse it, so a
  retried submit returns the same job instead of queueing a duplicate.

**Returns**

- Success: `{"job_uuid": "<uuid>", "job_dir": "<job directory on the GPU box>"}`
- Failure: `{"error": "<message>"}`

**Example**

```python
gpu_submit(
    local_dir="/path/to/project",
    commands=[
        "uv venv",
        "uv pip install --python .venv/bin/python numpy",
        ".venv/bin/python main.py",
    ],
)
```

> When installing into a freshly created virtualenv, pin the interpreter
> (`uv pip install --python .venv/bin/python <pkg>`): a bare `uv pip install` can resolve
> against a different environment and leave the job's `.venv` missing the package.

---

## `gpu_status`

**Signature:** `gpu_status(job_uuid: str) -> dict`

Report a job's current state.

**Parameters**

| Name       | Type  | Required | Description                                   |
| ---------- | ----- | -------- | --------------------------------------------- |
| `job_uuid` | `str` | yes      | The job's uuid (from `gpu_submit`/`gpu_list`). |

**Returns**

- Success: `{"status", "exit_code", "failed_command", "job_uuid", "job_dir"}`
  - `status` ∈ `QUEUED` | `RUNNING` | `COMPLETED` | `FAILED`.
  - `exit_code` — exit code of the last command run (`null` until the job finishes).
  - `failed_command` — the command that failed, only when `FAILED`; otherwise `null`.
- Failure: `{"error": "<message>"}` (e.g. `no such job`).

---

## `gpu_output`

**Signature:** `gpu_output(job_uuid: str, tail_lines: int = 200) -> dict`

Return the job's stdout and stderr. Works **while the job is still running** (partial output).

**Parameters**

| Name         | Type  | Required | Default | Description                                            |
| ------------ | ----- | -------- | ------- | ------------------------------------------------------ |
| `job_uuid`   | `str` | yes      | —       | The job's uuid.                                        |
| `tail_lines` | `int` | no       | `200`   | Return only the last N lines of each stream. **`0` = everything.** |

**Returns**

- Success: `{"stdout": "<text>", "stderr": "<text>"}`
- Failure: `{"error": "<message>"}`

---

## `gpu_wait`

**Signature:** `gpu_wait(job_uuid: str, max_wait_s: int = 600, poll_s: int = 10) -> dict`

Block until the job reaches a terminal state (`COMPLETED` or `FAILED`) or `max_wait_s`
elapses.

**Parameters**

| Name         | Type  | Required | Default | Description                                               |
| ------------ | ----- | -------- | ------- | --------------------------------------------------------- |
| `job_uuid`   | `str` | yes      | —       | The job's uuid.                                           |
| `max_wait_s` | `int` | no       | `600`   | Maximum seconds to wait. Keep below the MCP call timeout. |
| `poll_s`     | `int` | no       | `10`    | Seconds between status polls.                             |

**Returns**

- Terminal: the same fields as `gpu_status` (`status`, `exit_code`, `failed_command`,
  `job_uuid`, `job_dir`).
- Timed out: the current status plus `"still_running": true`.
- Failure: `{"error": "<message>"}`

> The job keeps running on the GPU box either way. If a wait times out, call `gpu_wait` again —
> **never resubmit** a job just because a wait expired.

---

## `gpu_list`

**Signature:** `gpu_list() -> str`

List every job the daemon knows about since it last started.

**Parameters:** none.

**Returns:** plain text, one line per job:

```
<job_uuid>  <status>   <first command>
```

Returns `(no jobs)` when the queue is empty, or `error: <message>` on failure.

> Job state lives in memory on the daemon: restarting the daemon forgets all jobs.
