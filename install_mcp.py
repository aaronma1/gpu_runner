"""Interactive setup for gpu_runner_mcp.py, run by `make install-mcp`.

Asks which machine runs the gpu_runner daemon, checks that `gpu_runner list` answers there (over
`ssh HOST .local/bin/gpu_runner list` for a remote one, exactly as the MCP server calls it), writes the
[mcp] table of $GPURUN_CONFIG_DIR/config.toml (default ~/.config/gpu_runner/config.toml; other
tables are kept), and registers the server with Claude Code. Nothing is written to the remote
machine: `make install` there provides gpu_runner (and its ssh gate).

    python install_mcp.py --python VENV_PYTHON --script INSTALLED_MCP_SCRIPT
"""
import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tomllib

NAME = "gpu_runner"
REMOTE_GPURUN = ".local/bin/gpu_runner"  # as in gpu_runner_mcp.py: `make install`'s, from $HOME
HERE = os.path.dirname(os.path.abspath(__file__))


def config_path():
    config_dir = os.path.expanduser(os.environ.get("GPURUN_CONFIG_DIR", "~/.config/gpu_runner"))
    return os.path.join(config_dir, "config.toml")


def ask(prompt, default=""):
    answer = input(f"{prompt} [{default}]: " if default else f"{prompt}: ").strip()
    return answer or default


def run(cmd, host=None, timeout=30):
    """Run `cmd` here, or on `host` over ssh; (returncode, stdout+stderr)."""
    if host:
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, shlex.join(cmd)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired) as e:
        return 1, str(e)
    return p.returncode, (p.stdout + p.stderr).strip()


def load_mcp_table(path):
    try:
        with open(path, "rb") as f:
            return tomllib.load(f).get("mcp", {})
    except FileNotFoundError:
        return {}
    except tomllib.TOMLDecodeError as e:
        sys.exit(f"install_mcp: bad config file {path}: {e}")


def write_mcp_table(path, values):
    """Replace the [mcp] table of `path` with `values`, keeping everything else."""
    try:
        with open(path) as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        lines = []
    kept, in_mcp = [], False
    for line in lines:
        header = line.split("#")[0].strip()
        if header.startswith("["):
            in_mcp = header == "[mcp]"
        if not in_mcp:
            kept.append(line)
    table = "[mcp]\n" + "".join(f"{k} = {json.dumps(v)}\n" for k, v in values.items())
    text = "\n".join(kept).rstrip()
    text = f"{text}\n\n{table}" if text else table
    tomllib.loads(text)  # never write a file the servers can't read
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def configure(path):
    old = load_mcp_table(path)
    print("Which machine runs the gpu_runner daemon? Give its ssh alias (from ~/.ssh/config),")
    print("or `local` for this machine.")
    host = ask("ssh host", old.get("ssh_host") or "local")
    host = "" if host == "local" else host
    values = {"ssh_host": host}
    if not host:  # a remote daemon's workdir is that machine's business, not the MCP config's
        home = os.path.expanduser("~")
        default = old.get("daemon_workdir") if not old.get("ssh_host") else None
        values["daemon_workdir"] = ask("daemon workdir on this machine",
                                       default or f"{home}/data/.gpu_runner")

    code, out = check_list(host)
    if code == 0:
        print("gpu_runner answered:\n  " + (out.replace("\n", "\n  ") or "(no jobs)"))
    else:
        where = host or "this machine"
        print(f"gpu_runner list on {where} failed:\n  {out}")
        if host:
            print(f"(needs key-based `ssh -o BatchMode=yes {host}`, `make install` on {host} "
                  "and the daemon running)")
        if ask("save these settings anyway? [y/N]").lower() != "y":
            sys.exit(1)

    write_mcp_table(path, values)
    print(f"wrote [mcp] to {path}")


def check_list(host):
    """Run `gpu_runner list` the way the MCP server will; (returncode, output)."""
    if host:
        return run([REMOTE_GPURUN, "list"], host, timeout=60)
    gpurun = shutil.which("gpu_runner")
    if gpurun:
        return run([gpurun, "list"], timeout=60)
    return run([os.path.join(HERE, ".venv", "bin", "python"),
                os.path.join(HERE, "gpu_runner.py"), "list"], timeout=60)


def register(python, script):
    cmd = ["claude", "mcp", "add", "-s", "user", NAME]
    if "GPURUN_CONFIG_DIR" in os.environ:
        cmd += ["-e", f"GPURUN_CONFIG_DIR={os.environ['GPURUN_CONFIG_DIR']}"]
    cmd += ["--", python, script]
    if not shutil.which("claude"):
        print("claude not found; register the server yourself with:\n  " + shlex.join(cmd))
        return
    subprocess.run(["claude", "mcp", "remove", "-s", "user", NAME], capture_output=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--python", required=True, help="python of the MCP server's venv")
    ap.add_argument("--script", required=True, help="installed gpu_runner_mcp.py")
    ap.add_argument("--no-register", action="store_true", help="only write the config")
    args = ap.parse_args()
    configure(config_path())
    if not args.no_register:
        register(args.python, args.script)


if __name__ == "__main__":
    main()
