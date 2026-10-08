"""Codex CLI implementation of the model-gateway port."""

from __future__ import annotations

import shutil
import subprocess
import time

from ..core.ports import ModelRequest, ModelResponse


class CodexCliModelGateway:
    name = "codex-cli"

    def invoke(self, request: ModelRequest) -> ModelResponse:
        executable = shutil.which("codex")
        if not executable:
            return ModelResponse(
                status="failed",
                exit_code=127,
                reason="codex executable not found",
            )
        final_message = request.run_dir / f"{request.label}-final-message.md"
        log_path = request.run_dir / f"{request.label}.log"
        command = [
            executable,
            "exec",
            "--sandbox",
            "workspace-write",
            "--ephemeral",
            "--cd",
            request.repo_root.as_posix(),
            "--output-last-message",
            final_message.as_posix(),
            "-",
        ]
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                input=request.prompt,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=request.timeout_seconds,
                check=False,
            )
            output = completed.stdout
            exit_code = completed.returncode
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or "") + (
                f"\nTIMEOUT after {request.timeout_seconds} seconds\n"
            )
            exit_code = 124
        log_path.write_text(output, encoding="utf-8")
        return ModelResponse(
            status="passed" if exit_code == 0 else "failed",
            exit_code=exit_code,
            duration_seconds=round(time.monotonic() - started, 3),
            log_path=log_path.as_posix(),
            final_message_path=(
                final_message.as_posix() if final_message.exists() else None
            ),
        )
