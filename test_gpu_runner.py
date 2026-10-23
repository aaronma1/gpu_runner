"""Test suite for gpu_runner.py.

    pytest -q test_gpu_runner.py                 # everything (~1-2 min)
    pytest -q test_gpu_runner.py -k "not slow"   # skip the throughput/size tests
    GPU_RUNNER_TEST_GPUS=3 pytest -q ...         # GPU ids to expect in CUDA_VISIBLE_DEVICES

Layers:
  * Daemon        — the Daemon class in-process: exposed_* called directly, _loop in a thread.
  * RPC           — a real `gpu_runner.py daemon` subprocess, driven with rpyc.connect.
  * CLI           — the same daemon, driven with `gpu_runner.py submit/query/list`.
No GPU is needed; GPU ids are only checked as CUDA_VISIBLE_DEVICES.
"""
import io
import os
import re
import socket
import subprocess
import sys
import tarfile
import threading
import time
import uuid

import pytest
import rpyc

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gpu_runner  # noqa: E402

RUNNER = [sys.executable, os.path.join(HERE, "gpu_runner.py")]
GPUS = [int(g) for g in os.environ.get("GPU_RUNNER_TEST_GPUS", "0,1").split(",")]
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


# ====================================================================== helpers
def make_tar(files=None, mode="w:gz", extra=None):
    """files: {name: str|bytes|(content, filemode)}; extra: callable(tarfile) for odd members."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode=mode) as t:
        for name, spec in (files or {}).items():
            content, fmode = spec if isinstance(spec, tuple) else (spec, 0o644)
            data = content.encode() if isinstance(content, str) else content
            ti = tarfile.TarInfo(name)
            ti.size, ti.mode = len(data), fmode
            t.addfile(ti, io.BytesIO(data))
        if extra:
            extra(t)
    return buf.getvalue()


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_port(port, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.1)
    return False


def wait_status(get_status, jid, timeout=30, states=gpu_runner.TERMINAL):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = get_status(jid)
        if st[0] in states:
            return st
        time.sleep(0.05)
    raise AssertionError(f"job {jid} still {get_status(jid)[0]} after {timeout}s")


def pids_alive(pattern):
    out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.split()
    return [p for p in out if int(p) != os.getpid()]


# ====================================================================== fixtures
@pytest.fixture
def daemon(tmp_path):
    """In-process Daemon with a fast poll, its loop running in a background thread."""
    d = gpu_runner.Daemon(str(tmp_path / "work"), GPUS, poll=0.05)
    os.makedirs(d.workdir)
    threading.Thread(target=d._loop, daemon=True).start()
    d.submit = lambda cmds, tar=b"", mid=None: d.exposed_receive_job(mid or str(uuid.uuid4()), tar, cmds)
    d.status = d.exposed_get_job_status
    d.out = d.exposed_get_job_out
    d.wait = lambda jid, timeout=30: wait_status(d.status, jid, timeout)
    return d


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    """A real daemon subprocess on a free port. stdin is an open pipe on purpose: a job must
    never be able to read (or block on) the daemon's stdin."""
    tmp = tmp_path_factory.mktemp("srv")
    port = free_port()
    env = dict(os.environ, GPU_RUNNER_PORT=str(port), GPU_RUNNER_WORKDIR=str(tmp / "work"),
               GPU_RUNNER_TEST_MARKER="inherited-ok")
    log = open(tmp / "daemon.log", "w")
    proc = subprocess.Popen(RUNNER + ["daemon", "--gpu", *map(str, GPUS)], env=env,
                            stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT)
    assert wait_for_port(port), f"daemon didn't start, see {tmp / 'daemon.log'}"
    yield {"port": port, "env": env, "tmp": tmp, "proc": proc, "log": tmp / "daemon.log"}
    proc.terminate()
    proc.wait(timeout=10)


@pytest.fixture
def conn(server):
    c = rpyc.connect("localhost", server["port"], config={"sync_request_timeout": 120})
    yield c
    c.close()


def cli(server, *args, stdin=None, timeout=60):
    return subprocess.run(RUNNER + list(args), input=stdin, capture_output=True,
                          env=server["env"], timeout=timeout)


def rpc_submit(conn, cmds, tar=b"", mid=None):
    return conn.root.receive_job(mid or str(uuid.uuid4()), tar, cmds)


def rpc_wait(conn, jid, timeout=60):
    return wait_status(conn.root.get_job_status, jid, timeout)


# ====================================================================== Daemon: submission
class TestSubmission:
    def test_returns_uuid4_string(self, daemon):
        jid = daemon.submit(["true"])
        assert isinstance(jid, str) and UUID_RE.match(jid)

    def test_ids_are_unique(self, daemon):
        ids = {daemon.submit(["true"]) for _ in range(50)}
        assert len(ids) == 50

    def test_new_job_is_queued_with_inputs_copied(self, daemon):
        cmds = ["echo a", "echo b"]
        daemon.poll = 3600  # keep the loop from starting it (first sleep already in progress)
        jid = daemon.submit(cmds)
        cmds.append("echo mutated-after-submit")
        job = daemon.jobs[jid]
        assert job.commands == ["echo a", "echo b"]
        assert job.commands is not cmds

    def test_empty_command_list_rejected(self, daemon):
        with pytest.raises(ValueError):
            daemon.submit([])

    def test_commands_coerced_to_str(self, daemon):
        jid = daemon.submit([42])  # `42` is not a command, but the type must not break the daemon
        assert daemon.jobs[jid].commands == ["42"]
        assert daemon.wait(jid)[0] == "FAILED"

    def test_dedup_returns_same_job(self, daemon):
        a = daemon.submit(["echo once"], mid="m-1")
        b = daemon.submit(["echo once"], mid="m-1")
        assert a == b
        assert sum(1 for j in daemon.jobs.values() if j.commands == ["echo once"]) == 1

    def test_dedup_ignores_payload_of_duplicate(self, daemon):
        a = daemon.submit(["echo first"], mid="m-2")
        b = daemon.submit(["echo second"], mid="m-2")
        assert a == b and daemon.jobs[a].commands == ["echo first"]

    def test_message_id_compared_as_string(self, daemon):
        assert daemon.exposed_receive_job(7, b"", ["true"]) == daemon.exposed_receive_job("7", b"", ["true"])

    def test_distinct_message_ids_distinct_jobs(self, daemon):
        assert daemon.submit(["true"], mid="x") != daemon.submit(["true"], mid="y")

    def test_concurrent_submits_same_message_id_make_one_job(self, daemon):
        results, barrier = [], threading.Barrier(20)

        def go():
            barrier.wait()
            results.append(daemon.submit(["echo race"], mid="race-1"))
        threads = [threading.Thread(target=go) for _ in range(20)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert len(set(results)) == 1
        assert sum(1 for j in daemon.jobs.values() if j.commands == ["echo race"]) == 1

    def test_concurrent_submits_distinct_all_run(self, daemon):
        ids, lock = [], threading.Lock()

        def go(i):
            jid = daemon.submit([f"echo job{i}"])
            with lock:
                ids.append(jid)
        threads = [threading.Thread(target=go, args=(i,)) for i in range(15)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert len(set(ids)) == 15
        for jid in ids:
            assert daemon.wait(jid)[0] == "COMPLETED"

    def test_unknown_job_raises_keyerror(self, daemon):
        with pytest.raises(KeyError):
            daemon.status(str(uuid.uuid4()))
        with pytest.raises(KeyError):
            daemon.out("not-a-uuid")


# ====================================================================== Daemon: lifecycle & scheduling
class TestLifecycle:
    def test_status_transitions(self, daemon):
        jid = daemon.submit(["sleep 1"])
        seen = []
        deadline = time.time() + 15
        while time.time() < deadline:
            st = daemon.status(jid)[0]
            if not seen or seen[-1] != st:
                seen.append(st)
            if st in gpu_runner.TERMINAL:
                break
            time.sleep(0.02)
        assert seen[-2:] == ["RUNNING", "COMPLETED"]
        assert set(seen) <= {"QUEUED", "RUNNING", "COMPLETED"}

    def test_timestamps_recorded(self, daemon):
        jid = daemon.submit(["true"])
        daemon.wait(jid)
        job = daemon.jobs[jid]
        assert job.started and job.ended and job.ended >= job.started

    def test_fifo_order(self, daemon, tmp_path):
        order = tmp_path / "order.txt"
        ids = [daemon.submit([f"echo {i} >> {order}"]) for i in range(6)]
        for jid in ids:
            daemon.wait(jid)
        assert order.read_text().split() == [str(i) for i in range(6)]

    def test_never_two_running(self, daemon):
        ids = [daemon.submit(["sleep 0.3"]) for _ in range(5)]
        max_running = 0
        while not all(daemon.status(j)[0] in gpu_runner.TERMINAL for j in ids):
            running = sum(daemon.status(j)[0] == "RUNNING" for j in ids)
            max_running = max(max_running, running)
            time.sleep(0.01)
        assert max_running == 1

    def test_queue_drains_after_failures(self, daemon):
        ids = [daemon.submit(["exit 1"]), daemon.submit(["true"]), daemon.submit(["exit 2"]),
               daemon.submit(["true"])]
        assert [daemon.wait(j)[0] for j in ids] == ["FAILED", "COMPLETED", "FAILED", "COMPLETED"]

    def test_running_pointer_cleared(self, daemon):
        daemon.wait(daemon.submit(["true"]))
        time.sleep(0.2)
        assert daemon.running is None and daemon.ready == []

    def test_tarball_memory_released(self, daemon):
        jid = daemon.submit(["true"], tar=make_tar({"big.bin": b"x" * 1_000_000}))
        daemon.wait(jid)
        assert daemon.jobs[jid].tarfile is None

    def test_list_jobs(self, daemon):
        a = daemon.submit(["echo listed-a"])
        daemon.wait(a)
        rows = {r[0]: r for r in daemon.exposed_list_jobs()}
        assert rows[a][1] == "COMPLETED" and rows[a][2] == "echo listed-a"


# ====================================================================== Daemon: execution
class TestExecution:
    def test_commands_run_in_order_in_job_dir(self, daemon):
        jid = daemon.submit(["pwd", "echo hi > f.txt", "cat f.txt"])
        assert daemon.wait(jid)[0] == "COMPLETED"
        out, _ = daemon.out(jid)
        lines = out.split()
        assert lines[0] == os.path.join(daemon.workdir, jid) and lines[1] == "hi"

    def test_each_command_is_a_fresh_shell(self, daemon):
        jid = daemon.submit(["cd /; export FOO=1", "pwd; echo \"foo=$FOO\""])
        daemon.wait(jid)
        out, _ = daemon.out(jid)
        assert os.path.join(daemon.workdir, jid) in out and "foo=\n" in out

    def test_shell_features(self, daemon):
        jid = daemon.submit(["printf 'b\\na\\n' | sort | tr -d '\\n'; echo", "for i in 1 2; do echo n$i; done"])
        daemon.wait(jid)
        assert daemon.out(jid)[0] == "ab\nn1\nn2\n"

    def test_multiline_command(self, daemon):
        jid = daemon.submit(["x=3\nif [ $x -eq 3 ]; then\n  echo three\nfi"])
        daemon.wait(jid)
        assert daemon.out(jid)[0] == "three\n"

    def test_stdout_stderr_separated(self, daemon):
        jid = daemon.submit(["echo to-out; echo to-err >&2"])
        daemon.wait(jid)
        assert daemon.out(jid) == ("to-out\n", "to-err\n")

    def test_output_accumulates_across_commands(self, daemon):
        jid = daemon.submit(["echo 1", "echo 2 >&2", "echo 3"])
        daemon.wait(jid)
        assert daemon.out(jid) == ("1\n3\n", "2\n")

    def test_unicode_and_invalid_utf8(self, daemon):
        jid = daemon.submit(["printf 'h\\303\\251 \\360\\237\\232\\200\\n'", "printf 'bad:\\377\\n'"])
        daemon.wait(jid)
        out = daemon.out(jid)[0]
        assert "hé 🚀" in out and "bad:�" in out

    @pytest.mark.slow
    def test_large_output_no_deadlock(self, daemon):
        jid = daemon.submit(["python3 -c \"import sys; sys.stdout.write('x' * 20_000_000); "
                             "sys.stderr.write('y' * 5_000_000)\""])
        assert daemon.wait(jid, timeout=60)[0] == "COMPLETED"
        out, err = daemon.out(jid)
        assert len(out) == 20_000_000 and len(err) == 5_000_000

    def test_job_cannot_read_daemon_stdin(self, daemon):
        # `cat` with no input must see EOF immediately rather than block on whatever the daemon's
        # stdin is (a terminal, or here pytest's stdin).
        jid = daemon.submit(["cat; echo after-cat"])
        st = daemon.wait(jid, timeout=10)
        assert st[0] == "COMPLETED" and "after-cat" in daemon.out(jid)[0]

    def test_background_process_does_not_hang_job(self, daemon):
        tag = f"bgsleep-{uuid.uuid4().hex[:8]}"
        jid = daemon.submit([f"bash -c 'exec -a {tag} sleep 30' & echo started"])
        t0 = time.time()
        st = daemon.wait(jid, timeout=25)
        assert st[0] == "COMPLETED" and time.time() - t0 < 10, "job waited for its background child"

    def test_no_processes_left_after_job(self, daemon):
        tag = f"leftover-{uuid.uuid4().hex[:8]}"
        jid = daemon.submit([f"(bash -c 'exec -a {tag} sleep 60' &) ; echo spawned"])
        daemon.wait(jid, timeout=25)
        time.sleep(0.5)
        assert pids_alive(tag) == [], "a background process outlived its job (it could hold a GPU)"


# ====================================================================== Daemon: failures
class TestFailures:
    def test_first_failure_stops_job(self, daemon):
        jid = daemon.submit(["echo one", "exit 3", "echo three"])
        st, code, failed = daemon.wait(jid)
        assert (st, code, failed) == ("FAILED", 3, "exit 3")
        assert daemon.out(jid)[0] == "one\n"

    @pytest.mark.parametrize("code", [1, 2, 127, 255])
    def test_exit_codes_preserved(self, daemon, code):
        assert daemon.wait(daemon.submit([f"exit {code}"]))[:2] == ("FAILED", code)

    def test_command_not_found(self, daemon):
        jid = daemon.submit(["definitely-not-a-command-xyz"])
        st, code, _ = daemon.wait(jid)
        assert (st, code) == ("FAILED", 127) and "not found" in daemon.out(jid)[1]

    def test_killed_by_signal(self, daemon):
        st, code, _ = daemon.wait(daemon.submit(["kill -9 $$"]))
        assert st == "FAILED" and code == -9

    def test_success_records_exit_zero(self, daemon):
        assert daemon.wait(daemon.submit(["true"]))[:3] == ("COMPLETED", 0, None)

    def test_bad_tarball_fails_job_with_reason(self, daemon):
        jid = daemon.submit(["echo never"], tar=b"definitely not a tarball")
        st = daemon.wait(jid)
        assert st[0] == "FAILED" and "setup/run error" in daemon.out(jid)[1]
        assert "never" not in daemon.out(jid)[0]

    def test_truncated_tarball(self, daemon):
        good = make_tar({"a.txt": "x" * 100_000}, mode="w")
        st = daemon.wait(daemon.submit(["true"], tar=good[: len(good) // 2]))
        assert st[0] == "FAILED"


# ====================================================================== Daemon: tarballs
class TestTarballs:
    @pytest.mark.parametrize("mode", ["w", "w:gz", "w:bz2", "w:xz"])
    def test_compression_formats(self, daemon, mode):
        jid = daemon.submit(["cat a.txt"], tar=make_tar({"a.txt": "fmt-ok"}, mode=mode))
        assert daemon.wait(jid)[0] == "COMPLETED" and daemon.out(jid)[0] == "fmt-ok"

    def test_no_tarball_still_gets_empty_dir(self, daemon):
        jid = daemon.submit(["ls -A | wc -l"])
        daemon.wait(jid)
        assert daemon.out(jid)[0].strip() == "0"

    def test_nested_dirs_and_exec_bit(self, daemon):
        tar = make_tar({"src/pkg/mod.py": "print('nested-ok')",
                        "run.sh": ("#!/bin/sh\npython3 src/pkg/mod.py\n", 0o755)})
        jid = daemon.submit(["./run.sh"], tar=tar)
        assert daemon.wait(jid)[0] == "COMPLETED" and daemon.out(jid)[0] == "nested-ok\n"

    def test_relative_symlink_inside_is_allowed(self, daemon):
        def link(t):
            ti = tarfile.TarInfo("link.txt")
            ti.type, ti.linkname = tarfile.SYMTYPE, "real.txt"
            t.addfile(ti)
        jid = daemon.submit(["cat link.txt"], tar=make_tar({"real.txt": "via-link"}, extra=link))
        assert daemon.wait(jid)[0] == "COMPLETED" and daemon.out(jid)[0] == "via-link"

    @pytest.mark.parametrize("member,linkname,kind", [
        ("../../escape.txt", None, "file"),
        ("evil", "/etc/passwd", "symlink"),
        ("evil2", "../../../etc", "symlink"),
        ("hard", "/etc/passwd", "hardlink"),
    ])
    def test_unsafe_members_rejected(self, daemon, tmp_path, member, linkname, kind):
        def odd(t):
            ti = tarfile.TarInfo(member)
            if kind == "file":
                ti.size = 3
                t.addfile(ti, io.BytesIO(b"bad"))
            else:
                ti.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
                ti.linkname = linkname
                t.addfile(ti)
        jid = daemon.submit(["true"], tar=make_tar(extra=odd))
        assert daemon.wait(jid)[0] == "FAILED"
        assert not (tmp_path / "escape.txt").exists()

    def test_absolute_member_lands_inside_job_dir(self, daemon):
        name = f"/abs-{uuid.uuid4().hex[:8]}.txt"
        jid = daemon.submit([f"cat .{name}"], tar=make_tar({name: "contained"}))
        assert daemon.wait(jid)[0] == "COMPLETED" and daemon.out(jid)[0] == "contained"
        assert not os.path.exists(name)

    def test_device_file_rejected(self, daemon):
        def dev(t):
            ti = tarfile.TarInfo("null")
            ti.type, ti.devmajor, ti.devminor = tarfile.CHRTYPE, 1, 3
            t.addfile(ti)
        assert daemon.wait(daemon.submit(["true"], tar=make_tar(extra=dev)))[0] == "FAILED"

    def test_setuid_bit_stripped(self, daemon):
        jid = daemon.submit(["stat -c %a s.sh"], tar=make_tar({"s.sh": ("#!/bin/sh\n", 0o4755)}))
        daemon.wait(jid)
        assert daemon.out(jid)[0].strip() == "755"

    def test_jobs_have_isolated_dirs(self, daemon):
        a = daemon.submit(["echo A > shared.txt"])
        daemon.wait(a)
        b = daemon.submit(["cat shared.txt"])
        assert daemon.wait(b)[0] == "FAILED"


# ====================================================================== Daemon: environment
class TestEnvironment:
    def test_cuda_visible_devices(self, daemon):
        jid = daemon.submit(['echo "$CUDA_VISIBLE_DEVICES|$CUDA_DEVICE_ORDER"'])
        daemon.wait(jid)
        assert daemon.out(jid)[0].strip() == ",".join(map(str, GPUS)) + "|PCI_BUS_ID"

    def test_no_gpus_means_empty_cvd(self, tmp_path):
        d = gpu_runner.Daemon(str(tmp_path / "w"), [], poll=0.05)
        threading.Thread(target=d._loop, daemon=True).start()
        jid = d.exposed_receive_job("m", b"", ['echo "[$CUDA_VISIBLE_DEVICES]"'])
        wait_status(d.exposed_get_job_status, jid)
        assert d.exposed_get_job_out(jid)[0].strip() == "[]"

    def test_environment_inherited(self, daemon, monkeypatch):
        monkeypatch.setenv("GPU_RUNNER_TEST_INHERIT", "yes")
        jid = daemon.submit(['echo "$GPU_RUNNER_TEST_INHERIT $HOME"; command -v python3'])
        daemon.wait(jid)
        out = daemon.out(jid)[0]
        assert out.startswith("yes /") and "python3" in out

    def test_cvd_not_leaked_into_daemon(self, daemon):
        before = os.environ.get("CUDA_VISIBLE_DEVICES")
        daemon.wait(daemon.submit(["true"]))
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == before


# ====================================================================== RPC layer (real daemon)
class TestRPC:
    def test_round_trip(self, conn):
        jid = rpc_submit(conn, ["echo rpc-ok"], tar=make_tar({"a": "b"}))
        assert rpc_wait(conn, jid)[0] == "COMPLETED"
        assert conn.root.get_job_out(jid)[0] == "rpc-ok\n"

    def test_results_are_values_not_proxies(self, server, conn):
        jid = rpc_submit(conn, ["echo by-value"])
        rpc_wait(conn, jid)
        out, err = conn.root.get_job_out(jid)
        status = conn.root.get_job_status(jid)
        rows = conn.root.list_jobs()
        conn.close()  # everything must still be usable without the connection
        assert out == "by-value\n" and err == "" and status[0] == "COMPLETED"
        assert any(r[0] == jid for r in rows) and isinstance(rows, tuple)

    def test_job_survives_client_disconnect(self, server):
        c = rpyc.connect("localhost", server["port"])
        jid = rpc_submit(c, ["sleep 1", "echo survived"])
        c.close()  # client gone while the job is queued/running
        c2 = rpyc.connect("localhost", server["port"])
        try:
            assert rpc_wait(c2, jid)[0] == "COMPLETED" and c2.root.get_job_out(jid)[0] == "survived\n"
        finally:
            c2.close()

    def test_remote_errors_keep_their_type(self, conn):
        with pytest.raises(KeyError):
            conn.root.get_job_status(str(uuid.uuid4()))
        with pytest.raises(ValueError):
            conn.root.receive_job(str(uuid.uuid4()), b"", [])

    def test_dedup_across_connections(self, server):
        mid = str(uuid.uuid4())
        ids = []
        for _ in range(3):  # a client retrying after reconnecting
            c = rpyc.connect("localhost", server["port"])
            ids.append(rpc_submit(c, ["echo retry"], mid=mid))
            c.close()
        assert len(set(ids)) == 1

    def test_many_parallel_clients(self, server):
        ids, errors = [], []

        def client(i):
            try:
                c = rpyc.connect("localhost", server["port"])
                ids.append(rpc_submit(c, [f"echo parallel-{i}"]))
                c.close()
            except Exception as e:  # noqa: BLE001
                errors.append(e)
        threads = [threading.Thread(target=client, args=(i,)) for i in range(20)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert not errors and len(set(ids)) == 20

    @pytest.mark.slow
    def test_large_upload(self, conn):
        payload = os.urandom(40 * 1024 * 1024)  # incompressible 40 MB
        jid = rpc_submit(conn, ["sha256sum blob.bin | cut -c1-16"], tar=make_tar({"blob.bin": payload}, mode="w"))
        import hashlib
        assert rpc_wait(conn, jid, timeout=120)[0] == "COMPLETED"
        assert conn.root.get_job_out(jid)[0].strip() == hashlib.sha256(payload).hexdigest()[:16]

    def test_garbage_on_port_does_not_kill_daemon(self, server, conn):
        with socket.create_connection(("127.0.0.1", server["port"])) as s:
            s.sendall(os.urandom(4096))
        with socket.create_connection(("127.0.0.1", server["port"])) as s:
            pass  # connect and hang up immediately
        time.sleep(0.3)
        assert server["proc"].poll() is None
        jid = rpc_submit(conn, ["echo still-up"])
        assert rpc_wait(conn, jid)[0] == "COMPLETED"

    def test_job_ignores_daemon_stdin(self, tmp_path):
        # Own daemon, with an open never-written stdin pipe: if a job could read the daemon's
        # stdin it would hang forever, and with one job at a time that would block every later
        # test that shares the module daemon.
        port = free_port()
        env = dict(os.environ, GPU_RUNNER_PORT=str(port), GPU_RUNNER_WORKDIR=str(tmp_path))
        proc = subprocess.Popen(RUNNER + ["daemon"], env=env, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            assert wait_for_port(port)
            c = rpyc.connect("localhost", port)
            jid = rpc_submit(c, ["cat; echo stdin-closed"])
            assert rpc_wait(c, jid, timeout=10)[0] == "COMPLETED"
            c.close()
        finally:
            proc.kill()
            proc.wait()

    def test_daemon_env_inherited(self, conn):
        jid = rpc_submit(conn, ["echo $GPU_RUNNER_TEST_MARKER"])
        rpc_wait(conn, jid)
        assert conn.root.get_job_out(jid)[0] == "inherited-ok\n"

    def test_listens_on_localhost_only(self, server):
        ips = subprocess.run(["hostname", "-I"], capture_output=True, text=True).stdout.split()
        ext = [ip for ip in ips if not ip.startswith("127.") and ":" not in ip]
        if not ext:
            pytest.skip("no non-loopback IPv4 address to test against")
        with socket.socket() as s:
            s.settimeout(2)
            assert s.connect_ex((ext[0], server["port"])) != 0

    def test_only_exposed_methods_reachable(self, conn):
        with pytest.raises(AttributeError):
            conn.root._run_job
        with pytest.raises(AttributeError):
            conn.root.jobs


# ====================================================================== CLI
class TestCLI:
    def test_submit_prints_only_uuid(self, server, tmp_path):
        tar = tmp_path / "j.tar.gz"
        tar.write_bytes(make_tar({"x.txt": "cli"}))
        p = cli(server, "submit", str(tar), "cat x.txt")
        assert p.returncode == 0 and UUID_RE.match(p.stdout.decode().strip()), p
        assert p.stdout.decode().count("\n") == 1

    def test_submit_from_stdin(self, server):
        p = cli(server, "submit", "-", "cat s.txt", stdin=make_tar({"s.txt": "from-stdin"}))
        jid = p.stdout.decode().strip()
        q = cli(server, "query", jid, "--wait", "--stdout")
        assert q.returncode == 0 and "from-stdin" in q.stdout.decode()

    def test_submit_without_tarball(self, server):
        p = cli(server, "submit", "", "echo no-tar")
        jid = p.stdout.decode().strip()
        assert cli(server, "query", jid, "--wait").returncode == 0

    def test_submit_multiple_commands(self, server):
        jid = cli(server, "submit", "", "echo c1", "echo c2").stdout.decode().strip()
        q = cli(server, "query", jid, "--wait", "--stdout")
        assert q.stdout.decode().endswith("c1\nc2\n")

    def test_submit_dedup_via_flag(self, server):
        mid = str(uuid.uuid4())
        a = cli(server, "submit", "", "echo d", "--message_id", mid).stdout
        b = cli(server, "submit", "", "echo d", "--message_id", mid).stdout
        assert a == b

    def test_submit_missing_tarfile_is_clean_error(self, server):
        p = cli(server, "submit", "/nonexistent/job.tar", "true")
        assert p.returncode != 0 and b"Traceback" not in p.stderr, p.stderr.decode()

    def test_submit_requires_a_command(self, server):
        assert cli(server, "submit", "").returncode == 2  # argparse usage error

    def test_query_wait_exit_codes(self, server):
        ok = cli(server, "submit", "", "true").stdout.decode().strip()
        bad = cli(server, "submit", "", "exit 4").stdout.decode().strip()
        assert cli(server, "query", ok, "--wait").returncode == 0
        q = cli(server, "query", bad, "--wait")
        assert q.returncode == 1 and q.stdout.decode().startswith("FAILED 4")

    def test_query_flags_select_streams(self, server):
        jid = cli(server, "submit", "", "echo O; echo E >&2").stdout.decode().strip()
        cli(server, "query", jid, "--wait")
        out_only = cli(server, "query", jid, "--stdout")
        err_only = cli(server, "query", jid, "--stderr")
        status_only = cli(server, "query", jid, "--status")
        everything = cli(server, "query", jid)
        assert out_only.stdout == b"O\n" and out_only.stderr == b""
        assert err_only.stderr == b"E\n" and err_only.stdout == b""
        assert status_only.stdout.decode().startswith("COMPLETED 0")
        assert b"COMPLETED" in everything.stdout and b"O\n" in everything.stdout and everything.stderr == b"E\n"

    def test_query_unknown_job(self, server):
        p = cli(server, "query", str(uuid.uuid4()))
        assert p.returncode != 0 and b"no such job" in p.stderr and b"Traceback" not in p.stderr

    def test_list(self, server):
        jid = cli(server, "submit", "", "echo listed").stdout.decode().strip()
        cli(server, "query", jid, "--wait")
        lines = cli(server, "list").stdout.decode().splitlines()
        assert any(l.startswith(jid) and "COMPLETED" in l and "echo listed" in l for l in lines)

    def test_no_subcommand_is_usage_error(self, server):
        assert cli(server).returncode == 2

    def test_no_daemon_is_clean_error(self, server):
        env = dict(server["env"], GPU_RUNNER_PORT=str(free_port()))
        for args in (["submit", "", "true"], ["query", str(uuid.uuid4())], ["list"]):
            p = subprocess.run(RUNNER + args, capture_output=True, env=env, timeout=30)
            assert p.returncode != 0 and b"no daemon" in p.stderr and b"Traceback" not in p.stderr, args


# ====================================================================== daemon process
class TestDaemonProcess:
    def test_second_daemon_on_same_port_exits(self, server):
        p = subprocess.Popen(RUNNER + ["daemon"], env=server["env"], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        try:
            rc = p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
            pytest.fail("second daemon kept running although the port was taken")
        assert rc != 0

    def test_workdir_created(self, tmp_path):
        env = dict(os.environ, GPU_RUNNER_PORT=str(free_port()),
                   GPU_RUNNER_WORKDIR=str(tmp_path / "deep" / "new" / "dir"))
        p = subprocess.Popen(RUNNER + ["daemon"], env=env, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        try:
            assert wait_for_port(int(env["GPU_RUNNER_PORT"]))
            assert (tmp_path / "deep" / "new" / "dir").is_dir()
        finally:
            p.terminate()
            p.wait(timeout=10)

    def test_sigterm_stops_daemon(self, tmp_path):
        env = dict(os.environ, GPU_RUNNER_PORT=str(free_port()), GPU_RUNNER_WORKDIR=str(tmp_path))
        p = subprocess.Popen(RUNNER + ["daemon"], env=env, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        assert wait_for_port(int(env["GPU_RUNNER_PORT"]))
        p.terminate()
        assert p.wait(timeout=10) is not None

    @pytest.mark.slow
    def test_throughput(self, conn):
        t0 = time.time()
        ids = [rpc_submit(conn, ["true"]) for _ in range(10)]
        for jid in ids:
            rpc_wait(conn, jid, timeout=60)
        elapsed = time.time() - t0
        assert elapsed < 30, f"10 trivial jobs took {elapsed:.1f}s"