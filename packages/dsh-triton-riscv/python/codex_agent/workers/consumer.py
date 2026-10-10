"""Bounded child execution, manual ACK and MySQL-owned recovery."""
import asyncio
import json
import os
import signal
import sys
import time

from codex_agent.tasks.store import LeaseLost


async def execute_delivery(jobs, delivery, kind):
    try:
        if len(delivery.body) > 4096:
            raise ValueError("oversized task envelope")
        envelope = json.loads(delivery.body)
        job = await asyncio.to_thread(jobs.claim, envelope, kind)
    except (ValueError, KeyError):
        await delivery.reject(requeue=False)
        return
    if job is None:
        await delivery.ack()
        return
    process = None
    try:
        # No credentials, code or shell command arrive in the queue message.
        process = await asyncio.create_subprocess_exec(sys.executable, '-I', '-m',
            'codex_agent.workers', 'execute', '--workspace', str(jobs.db.root),
            '--job', job['id'], '--owner', job['owner'], start_new_session=True,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        deadline = time.monotonic() + jobs.config.jobTimeoutSeconds
        while process.returncode is None:
            cancelled = await asyncio.to_thread(jobs.heartbeat, job['id'], job['owner'])
            if cancelled or time.monotonic() > deadline:
                break
            try:
                await asyncio.wait_for(process.wait(), timeout=min(5, jobs.config.leaseSeconds / 3,
                    max(.01, deadline - time.monotonic())))
            except asyncio.TimeoutError:
                pass
        if process.returncode is None:
            await stop_child(process)
        current = await asyncio.to_thread(jobs.get, job['id'])
        if current['status'] == 'running':
            status = 'needs-reconciliation' if current['effects_started'] else (
                'cancelled' if current['cancel_requested'] else 'failed')
            await asyncio.to_thread(jobs.finish, job['id'], job['owner'], status,
                {'reason': 'child stopped without a durable result', 'exit_code': process.returncode})
        await delivery.ack()
    except LeaseLost:
        if process is not None:
            await stop_child(process)
        # Durable recovery owns the task now; do not replay it just because AMQP redelivers.
        await delivery.ack()
    except BaseException:
        if process is not None:
            await stop_child(process)
        raise  # Connection closes without ACK; recovery handles the persisted claim.


async def stop_child(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        await asyncio.wait_for(process.wait(), 3)
    except asyncio.TimeoutError:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    await process.wait()
