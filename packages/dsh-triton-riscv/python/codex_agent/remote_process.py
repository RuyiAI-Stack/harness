"""Trusted supervisor-owned cancellation, with Linux child-reaping proof.

No caller-supplied PIDs. pidfds pin the actual process; orphaned descendants
are adopted by this single-threaded subreaper before confirming shutdown.
"""
import ctypes
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def enable_subreaper():
    if sys.platform != "linux" or not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        return False
    library = ctypes.CDLL(None, use_errno=True)
    library.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    library.prctl.restype = ctypes.c_int
    return library.prctl(36, 1, 0, 0, 0) == 0  # PR_SET_CHILD_SUBREAPER


def child_ids():
    return [int(value) for value in Path(f"/proc/self/task/{os.getpid()}/children").read_text().split()]


def signal_owned_child(pid, signum):
    try:
        descriptor = os.pidfd_open(pid)
    except ProcessLookupError:
        return
    try:
        # Recheck after pinning: never signal a PID that was reused elsewhere.
        if pid in child_ids():
            signal.pidfd_send_signal(descriptor, signum)
    except ProcessLookupError:
        pass
    finally:
        os.close(descriptor)


def drain_children(timeout=3):
    deadline = time.monotonic() + timeout
    while True:
        children = child_ids()
        if not children:
            return
        for pid in children:
            signal_owned_child(pid, signal.SIGKILL)
        while True:
            try:
                pid, _ = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                break
            if pid == 0:
                break
        if time.monotonic() >= deadline:
            raise RuntimeError("descendant shutdown not confirmed; preserve the capacity reservation")
        time.sleep(0.02)


def run_controlled(argv, *, stdin, stdout, cancelled):
    supported = enable_subreaper()
    process = subprocess.Popen(argv, stdin=stdin, stdout=stdout, stderr=subprocess.STDOUT,
                               close_fds=True, start_new_session=True)
    signalled_at = None
    while True:
        try:
            code = process.wait(timeout=0.1)
            break
        except subprocess.TimeoutExpired:
            if supported and cancelled():
                if signalled_at is None:
                    signal_owned_child(process.pid, signal.SIGTERM)
                    signalled_at = time.monotonic()
                elif time.monotonic() - signalled_at >= 2:
                    signal_owned_child(process.pid, signal.SIGKILL)
    # Reap even naturally completed commands: detached descendants must not
    # outlive a released admission slot. A drain failure produces no terminal proof.
    if supported:
        drain_children()
    return {"returncode": code, "cancel_applied": signalled_at is not None,
            "descendants_stopped": supported, "cancellation_supported": supported}
