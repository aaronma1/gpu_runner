"""Interactive setup for gpu_runner_mcp.py, run by `make install-mcp`.

Asks which machine runs the gpu_runner daemon and where gpu_runner lives there, checks that the
CLI answers, writes the [mcp] table of $GPURUN_CONFIG_DIR/config.toml (default
~/.config/gpu_runner/config.toml; other tables are kept), and registers the server with Claude Code.

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
HERE = os.path.dirname(os.path.abspath(__file__))
# prints $HOME, then the gpu_runner command if installed (non-interactive ssh may lack ~/.local/bin)
PROBE = ('echo "$HOME"; command -v gpu_runner || '
         '{ [ -x "$HOME/.local/bin/gpu_runner" ] && echo "$HOME/.local/bin/gpu_runner"; }; true')
# the venv `make install` creates (rpyc) and `make install-mcp` reuses (mcp), relative to $HOME
VENV_PYTHON = ".local/share/gpu_runner/venv/bin/python"


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
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           stdin=subprocess.DEVNULL)
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

    code, out = run(["sh", "-c", PROBE], host)
    if code != 0:
        sys.exit(f"install_mcp: can't reach {host or 'this machine'}: {out}\n"
                 "(the MCP server uses `ssh -o BatchMode=yes`, so key-based login must work)")
    home, *found = out.splitlines()
    where = host or "this machine"

    if found:
        print(f"found gpu_runner on {where}: {found[0]}")
        gpurun_default = found[0]
    elif not host:
        gpurun_default = os.path.join(HERE, "gpu_runner.py")
    else:
        print(f"no gpu_runner command on {host} (`make install` there provides one)")
        gpurun_default = f"{home}/Projects/gpu_runner/gpu_runner.py"

    def path_on_host(p):  # paths are passed quoted over ssh, so expand ~ here
        return home + p[1:] if p == "~" or p.startswith("~/") else p

    gpurun = path_on_host(ask(f"gpu_runner on {where}", old.get("gpurun", gpurun_default)))
    python = f"{home}/{VENV_PYTHON}"
    workdir = path_on_host(ask(f"daemon workdir on {where}",
                               old.get("daemon_workdir", f"{home}/data/.gpu_runner")))

    code, out = run([python, gpurun, "list"] if python else [gpurun, "list"], host, timeout=60)
    if code == 0:
        print("gpu_runner answered:\n  " + (out.replace("\n", "\n  ") or "(no jobs)"))
    else:
        print(f"gpu_runner list failed (is the daemon running?):\n  {out}")
        if ask("save these settings anyway? [y/N]").lower() != "y":
            sys.exit(1)

    write_mcp_table(path, {"ssh_host": host, "gpurun": gpurun, "python": python,
                           "daemon_workdir": workdir})
    print(f"wrote [mcp] to {path}")


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
