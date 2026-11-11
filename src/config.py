"""Daemon/CLI settings, read at startup from the [runner] table of
$GPURUN_CONFIG_DIR/config.toml (default ~/.config/gpu_runner/config.toml); GPU_RUNNER_<KEY>
environment variables override it:

    [runner]
    workdir = "~/data/.gpu_runner"                  # job directories and their stdout/stderr
    job_uv_cache = "~/data/.cache/gpu_runner/uv"    # UV_CACHE_DIR for jobs
    job_cache = "~/data/.cache/gpu_runner"          # XDG_CACHE_HOME for jobs
    port = 18861                                    # rpyc port (localhost only)
    sandbox = "plain"                               # job runner: "plain", or "bwrap" (bubblewrap)
"""
import os

from utils import load_config


def setting(config, key, default):
    """GPU_RUNNER_<KEY> from the environment, else the config value, else `default`."""
    return os.environ.get(f"GPU_RUNNER_{key.upper()}", config.get(key, default))


CONFIG = load_config("runner", "gpu_runner")
DAEMON_WORKDIR = os.path.expanduser(setting(CONFIG, "workdir", "~/data/.gpu_runner"))
JOB_UV_CACHE = os.path.expanduser(setting(CONFIG, "job_uv_cache", "~/data/.cache/gpu_runner/uv"))
JOB_CACHE = os.path.expanduser(setting(CONFIG, "job_cache", "~/data/.cache/gpu_runner"))

PORT = int(setting(CONFIG, "port", 18861))
SANDBOX = setting(CONFIG, "sandbox", "plain")  # a key of runner.RUNNERS
