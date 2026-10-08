"""Rootless Linux validation launcher using installed util-linux/coreutils.

This standalone file is transferred by the trusted executor, never supplied by
the model. No candidate Python is imported before namespaces/root/caps are set.
The policy is not a VM, seccomp filter or aggregate resource quota.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys


POLICY = "rootless-namespace-v1"
EXIT_SETUP_FAILED = 79
SYSTEM_PATH = "/usr/sbin:/usr/bin:/sbin:/bin"
TOOLS = ("unshare", "mount", "chroot", "setpriv")
RUNTIME_KEYS = (
    "TRITON_DIR", "BUILD_DIR", "LLVM_SYSPATH", "LLVM_BINARY_DIR",
    "BUDDY_MLIR_BINARY_DIR", "TRITON_SHARED_OPT_PATH",
)


def launcher_digest() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def namespaces() -> dict[str, str]:
    return {name: os.readlink(f"/proc/self/ns/{name}") for name in ("user", "mnt", "net", "pid", "ipc")}


def installed_tools() -> dict[str, str]:
    if platform.system() != "Linux":
        raise RuntimeError("rootless namespace validation requires Linux")
    result = {name: shutil.which(name, path=SYSTEM_PATH) for name in TOOLS}
    missing = [name for name, value in result.items() if value is None]
    if missing:
        raise RuntimeError("missing isolation tools: " + ", ".join(missing))
    return result


def allowed_runtime(path: str) -> Path:
    source = Path(path)
    if not source.is_absolute() or ".." in source.parts:
        raise ValueError("runtime paths must be absolute and traversal-free")
    broad = {Path(p) for p in ("/", "/home", "/root", "/etc", "/tmp", "/var", "/proc", "/sys", "/run", "/dev")}
    broad.add(Path.home())
    broad |= {p.resolve() for p in broad}
    if source in broad:
        raise ValueError(f"refusing broad runtime mount: {source}")
    source = source.resolve(strict=True)
    # Runtime configuration is host-owned, but reject common accidental broad grants.
    if source in broad:
        raise ValueError(f"refusing broad runtime mount: {source}")
    if not source.is_file() and not source.is_dir():
        raise ValueError(f"runtime mount must be a file or directory: {source}")
    return source


def build_policy(root: Path, stage: Path, environ: dict[str, str], command: list[str]) -> dict:
    root, stage = root.resolve(strict=True), stage.resolve(strict=True)
    if not command or any("\0" in arg for arg in command):
        raise ValueError("validation command is missing or invalid")
    if not (stage / "workspace").is_dir():
        raise ValueError("a prepared source snapshot is required")
    venv = allowed_runtime(str(root / ".venv"))
    python = venv / "bin/python"
    if not python.is_file():
        raise ValueError("repository virtualenv Python is missing")
    mounts = {str(venv)}
    runtime = {}
    for key in RUNTIME_KEYS:
        if not environ.get(key):
            continue
        value = allowed_runtime(environ[key])
        runtime[key] = str(value)
        if key == "TRITON_DIR":
            # Only the Python package is needed, not the entire source/history tree.
            mounts.add(str(allowed_runtime(str(value / "python"))))
        else:
            mounts.add(str(value))
    backend = root / "backend"
    if backend.is_dir():
        mounts.add(str(allowed_runtime(str(backend))))
    # Libraries alongside Buddy executables may be needed by the dynamic loader.
    if runtime.get("BUDDY_MLIR_BINARY_DIR"):
        library = Path(runtime["BUDDY_MLIR_BINARY_DIR"]).parent / "lib"
        if library.is_dir():
            mounts.add(str(allowed_runtime(str(library))))
    path = [str(venv / "bin")]
    path += [runtime[key] for key in ("LLVM_BINARY_DIR", "BUDDY_MLIR_BINARY_DIR") if key in runtime]
    env = {
        **runtime,
        "PATH": ":".join([*path, SYSTEM_PATH]),
        "HOME": "/tmp/home", "TMPDIR": "/tmp", "XDG_CACHE_HOME": "/tmp/cache",
        "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "USER": "sandbox", "LOGNAME": "sandbox",
        "VIRTUAL_ENV": str(venv), "PYTHONPATH": f"/work:{root}",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTEST_ADDOPTS": "-p no:cacheprovider",
        "TRITON_RISCV_DIR": str(root), "TRITON_PLUGIN_DIRS": str(root),
        "TRITON_VENV": str(venv), "TRITON_HOME": "/tmp/home",
        "TRITON_RUNTIME_ROOT": "/tmp/triton", "TRITON_CACHE_DIR": "/tmp/triton/cache",
        "TRITON_DUMP_DIR": "/tmp/triton/dump", "TRITON_OVERRIDE_DIR": "/tmp/triton/override",
        "TRITON_SHARED_DUMP_PATH": "/tmp/triton/dump/shared",
        "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
    }
    # These are behavior settings, not arbitrary ambient environment inheritance.
    for key in ("TRITON_RISCV_LOWERING_MODE", "TRITON_RISCV_USE_IME", "TRITON_SHARED_SANITIZER_TYPE"):
        if key in environ:
            env[key] = environ[key]
    argv = [str(python) if command[0] in {"python", "python3"} else command[0], *command[1:]]
    return {"policy": POLICY, "stage": str(stage), "mounts": sorted(mounts),
            "environment": env, "command": argv, "before": namespaces()}


def _run(*args: str) -> None:
    subprocess.run(list(args), check=True, stdin=subprocess.DEVNULL, close_fds=True)


def _exec(argv: list[str], environment: dict[str, str]) -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    with open("/dev/null", "rb") as handle:
        os.dup2(handle.fileno(), 0)
    # No inherited directory fd/socket can expose the original root or server.
    for name in list(Path("/proc/self/fd").iterdir()):
        fd = int(name.name)
        if fd > 2:
            try:
                os.close(fd)
            except OSError:
                pass
    os.execve(argv[0], argv, environment)


def enter(policy: dict) -> None:
    tools = installed_tools()
    if any(namespaces()[key] == value for key, value in policy["before"].items()):
        raise RuntimeError("required namespaces were not established")
    if os.getpid() != 1:
        raise RuntimeError("sandbox supervisor must own PID namespace init")
    stage = Path(policy["stage"])
    rootfs = stage / "rootfs"
    rootfs.mkdir(mode=0o700)
    mount = tools["mount"]
    _run(mount, "--make-rprivate", "/")
    _run(mount, "-t", "tmpfs", "-o", "size=16m,mode=0755,nosuid,nodev", "tmpfs", str(rootfs))

    def bind(source: str, target: str, *, device: bool = False) -> None:
        origin = Path(source)
        destination = rootfs / target.lstrip("/")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if origin.is_dir():
            destination.mkdir(exist_ok=True)
        else:
            destination.touch()
        # Nonrecursive bind excludes nested host mounts. All input stays readonly.
        _run(mount, "--bind", str(origin), str(destination))
        options = "remount,bind,ro,nosuid" + ("" if device else ",nodev")
        _run(mount, "-o", options, str(destination))

    for name in ("usr", "bin", "sbin", "lib", "lib64"):
        source = Path("/") / name
        if source.is_symlink():
            (rootfs / name).symlink_to(os.readlink(source))
        elif source.is_dir():
            bind(str(source), str(source))
    # Some distributions route /usr/bin/ld through this one alternatives entry.
    for name in ("/etc/ld.so.cache", "/etc/ld.so.conf", "/etc/ld.so.conf.d", "/etc/alternatives/ld"):
        if Path(name).exists():
            bind(name, name)
    mounted = [Path("/usr"), Path("/bin"), Path("/sbin"), Path("/lib"), Path("/lib64")]
    for name in sorted(policy["mounts"], key=lambda p: len(Path(p).parts)):
        source = Path(name)
        if any(source.is_relative_to(parent) for parent in mounted):
            continue
        bind(name, name)
        mounted.append(source)
    bind(str(stage / "workspace"), "/work")
    for name in ("null", "zero", "random", "urandom"):
        bind(f"/dev/{name}", f"/dev/{name}", device=True)
    for name, size in (("tmp", "1g"), ("dev/shm", "64m")):
        destination = rootfs / name
        destination.mkdir(parents=True, exist_ok=True)
        _run(mount, "-t", "tmpfs", "-o", f"size={size},mode=1777,nosuid,nodev", "tmpfs", str(destination))
    (rootfs / "tmp/home").mkdir()
    for name in ("cache", "dump/shared", "override"):
        (rootfs / "tmp/triton" / name).mkdir(parents=True, exist_ok=True)
    (rootfs / "proc").mkdir()
    # This /proc sees only the new PID namespace, never host processes/credentials.
    _run(mount, "-t", "proc", "-o", "nosuid,nodev,noexec", "proc", str(rootfs / "proc"))
    # Change this mount's flags, not the tmpfs superblock owned by the user namespace.
    _run(mount, "-o", "remount,bind,ro,nosuid,nodev", str(rootfs))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
    resource.setrlimit(resource.RLIMIT_FSIZE, (256 * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3,) * 2)
    print("SANDBOX_POLICY=" + POLICY, flush=True)
    print("SANDBOX_READONLY=" + json.dumps(policy["mounts"]), flush=True)
    # No fd to the original root or policy is passed to candidate code. chroot
    # changes cwd to /; all capabilities are dropped before executing the program.
    argv = [tools["chroot"], str(rootfs), tools["setpriv"], "--no-new-privs",
            "--bounding-set=-all", "--inh-caps=-all", "--ambient-caps=-all",
            "/bin/sh", "-c", 'cd /work && exec "$@"', "sandbox", *policy["command"]]
    _exec(argv, policy["environment"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--stage", type=Path)
    parser.add_argument("--enter", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        if args.enter:
            enter(json.loads(args.enter.read_text()))
        tools = installed_tools()
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if args.repo is None or args.stage is None:
            raise ValueError("repo and stage are required")
        policy = build_policy(args.repo, args.stage, dict(os.environ), command)
        path = args.stage / "sandbox-policy.json"
        with path.open("x", encoding="utf-8") as handle:
            json.dump(policy, handle)
        # The only child of unshare is namespace PID 1. Killing it tears down its
        # descendants, including detached subprocesses. stdin is never the SSH pipe.
        _exec([
            tools["unshare"], "--user", "--map-root-user", "--mount", "--net", "--ipc",
            "--pid", "--fork", "--kill-child=KILL", "--mount-proc",
            sys.executable, "-I", str(Path(__file__).resolve()), "--enter", str(path),
        ], {"PATH": SYSTEM_PATH, "LANG": "C.UTF-8"})
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"SANDBOX_ERROR={error}", file=sys.stderr, flush=True)
        return EXIT_SETUP_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
