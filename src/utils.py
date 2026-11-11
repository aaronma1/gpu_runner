"""Helpers shared by the daemon, the CLI (main.py) and mcp_server.py.

Installed next to both scripts (`make install` / `make install-mcp` put it in $(LIBDIR)); keep it
stdlib-only so either venv can import it.
"""
import io
import os
import sys
import tarfile
import tomllib

TERMINAL = frozenset({"COMPLETED", "FAILED", "CANCELLED"})


def load_config(table, prog):
    """The [`table`] table of $GPURUN_CONFIG_DIR/config.toml (default ~/.config/gpu_runner);
    {} if the file doesn't exist. Exits with a `prog:`-prefixed message if it doesn't parse."""
    config_dir = os.path.expanduser(os.environ.get("GPURUN_CONFIG_DIR", "~/.config/gpu_runner"))
    path = os.path.join(config_dir, "config.toml")
    try:
        with open(path, "rb") as f:
            return tomllib.load(f).get(table, {})
    except FileNotFoundError:
        return {}
    except tomllib.TOMLDecodeError as e:
        sys.exit(f"{prog}: bad config file {path}: {e}")


def is_within(path, root):
    """True if `path` is `root` or below it. Both should already be realpath'd."""
    return path == root or path.startswith(root + os.sep)


def tar_gz(entries, filter=None):
    """tar.gz (bytes) of `entries`, an iterable of (path, arcname); directories are added
    recursively. `filter` is passed to TarFile.add (return None to drop a member)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path, arcname in entries:
            tar.add(path, arcname=arcname, filter=filter)
    return buf.getvalue()


def safe_extract(data, dest):
    """Extract tarball bytes (any compression) with untrusted member names into `dest`.

    Raises tarfile.TarError for members that would land outside dest, links that point outside
    it, and device files. Uses tarfile's "data" filter where available (it also strips setuid
    bits); otherwise checks members by hand."""
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(dest, filter="data")
            return
        root = os.path.realpath(dest)
        for m in tar.getmembers():
            if not is_within(os.path.realpath(os.path.join(root, m.name)), root):
                raise tarfile.TarError(f"unsafe path in tarball: {m.name}")
            if m.isdev():
                raise tarfile.TarError(f"device file in tarball: {m.name}")
            if m.issym() or m.islnk():
                base = os.path.dirname(m.name) if m.issym() else ""  # hard links: archive-relative
                if not is_within(os.path.realpath(os.path.join(root, base, m.linkname)), root):
                    raise tarfile.TarError(f"link outside the tarball's root: {m.name}")
        tar.extractall(dest)
