"""Start the local Triton-RISCV Agent web workbench."""

from __future__ import annotations

import argparse
import os

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Triton-RISCV Agent platform.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--repo-root", help="Triton-RISCV target checkout (not the plugin source)")
    parser.add_argument("--state-dir", help="Directory for session and memory databases")
    args = parser.parse_args()
    if args.repo_root:
        os.environ["TRITON_RISCV_REPO_ROOT"] = args.repo_root
    if args.state_dir:
        os.environ["TRITON_RISCV_STATE_DIR"] = args.state_dir
    uvicorn.run(
        "codex_agent.platform.api:create_app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        factory=True,
    )


if __name__ == "__main__":
    main()
